// Voice in the browser: record a push-to-talk message, play spoken answers. The backend does the
// speech recognition and synthesis (POST /api/voice/transcribe, /api/voice/speak). Browser APIs
// are only touched when a method is called, so the pure helpers run under node --test.

import { Endpointer, rms } from "./handsfree.ts";
import type { Conversation } from "./state.ts";
import type { ChatEvent } from "./types.ts";

const MIME_TYPES = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus"];
export const MAX_RECORDING_MS = 60_000;

/** The first recording format the browser supports; "" lets the browser choose. */
export function pickMimeType(isSupported: (type: string) => boolean): string {
  return MIME_TYPES.find((type) => isSupported(type)) ?? "";
}

/** The answer to read aloud when `event` ends a turn live (not a replay of history: ids up to
 * `replayEnd` were already in the log when the session was opened). */
export function replyToSpeak(conv: Conversation, event: ChatEvent, replayEnd: number): string | null {
  if (event.type !== "done" || event.id <= replayEnd) {
    return null;
  }

  let start = conv.items.length;

  while (start > 0 && conv.items[start - 1]?.kind !== "user" && conv.items[start - 1]?.kind !== "notice") {
    start -= 1;
  }

  const text = conv.items
    .slice(start)
    .filter((item) => item.kind === "assistant")
    .map((item) => item.text)
    .join("\n")
    .trim();

  return text || null;
}

/** Whether this browser can record (a secure context such as http://127.0.0.1 is needed). */
export function canRecord(): boolean {
  return typeof MediaRecorder !== "undefined" && typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
}

/** One recording at a time: start() asks for the microphone, stop() resolves with the audio. */
export class Recorder {
  #recorder: MediaRecorder | null = null;
  #stream: MediaStream | null = null;
  #chunks: Blob[] = [];
  #timer = 0;
  #done: ((blob: Blob) => void) | null = null;
  #meter: LevelMeter | null = null;
  onAutoStop: (() => void) | null = null; // called when MAX_RECORDING_MS ends a recording
  onLevel: ((level: number) => void) | null = null; // microphone RMS, about 16 times a second

  get recording(): boolean {
    return this.#recorder !== null;
  }

  async start(): Promise<void> {
    if (this.#recorder) return;

    this.#stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    const mimeType = pickMimeType((type) => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(this.#stream, mimeType ? { mimeType } : {});
    this.#chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.#chunks.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(this.#chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
      this.#release();
      this.#done?.(blob);
      this.#done = null;
    };
    this.#recorder = recorder;
    recorder.start();
    this.#timer = window.setTimeout(() => this.onAutoStop?.(), MAX_RECORDING_MS);
    this.#meter = LevelMeter.open(this.#stream, (level) => this.onLevel?.(level));
  }

  /** Finish and return the recording (an empty blob if nothing was recording). */
  stop(): Promise<Blob> {
    const recorder = this.#recorder;

    if (!recorder) return Promise.resolve(new Blob([]));

    return new Promise((resolve) => {
      this.#done = resolve;
      recorder.stop();
    });
  }

  /** Drop the recording (switching sessions, page errors). */
  cancel(): void {
    if (!this.#recorder) return;
    this.#done = null;
    this.#recorder.stop();
  }

  #release(): void {
    window.clearTimeout(this.#timer);
    this.#meter?.close();
    this.#meter = null;
    this.#stream?.getTracks().forEach((track) => track.stop());
    this.#stream = null;
    this.#recorder = null;
  }
}

const LEVEL_MS = 60;

/** Live microphone level for the meter (push-to-talk); nothing if the browser has no Web Audio. */
class LevelMeter {
  #context: AudioContext;
  #timer: number;

  private constructor(stream: MediaStream, onLevel: (level: number) => void) {
    this.#context = new AudioContext();
    const analyser = this.#context.createAnalyser();
    analyser.fftSize = 1024;
    const samples = new Float32Array(analyser.fftSize);
    this.#context.createMediaStreamSource(stream).connect(analyser);
    void this.#context.resume().catch(() => undefined);
    this.#timer = window.setInterval(() => {
      analyser.getFloatTimeDomainData(samples);
      onLevel(rms(samples));
    }, LEVEL_MS);
  }

  static open(stream: MediaStream, onLevel: (level: number) => void): LevelMeter | null {
    if (typeof AudioContext === "undefined") return null;

    try {
      return new LevelMeter(stream, onLevel);
    } catch {
      return null; // the recording still works without a meter
    }
  }

  close(): void {
    window.clearInterval(this.#timer);
    void this.#context.close().catch(() => undefined);
  }
}

/** Plays one spoken answer at a time; a new one replaces the old. `onChange` tells when sound
 * starts and stops (hands-free listening pauses meanwhile, so it does not hear itself). */
export class Player {
  #audio: HTMLAudioElement | null = null;
  #url = "";
  onChange: ((playing: boolean) => void) | null = null;

  play(wav: Blob): void {
    this.stop();
    this.#url = URL.createObjectURL(wav);
    const audio = new Audio(this.#url);
    audio.onended = () => this.stop();
    this.#audio = audio;
    this.onChange?.(true);
    void audio.play().catch(() => this.stop()); // autoplay refused: stay silent
  }

  stop(): void {
    const wasPlaying = this.#audio !== null;
    this.#audio?.pause();
    this.#audio = null;

    if (this.#url) {
      URL.revokeObjectURL(this.#url);
      this.#url = "";
    }

    if (wasPlaying) this.onChange?.(false);
  }
}

/** Whether this browser can listen hands-free (recording plus live microphone levels). */
export function canListen(): boolean {
  return canRecord() && typeof AudioContext !== "undefined";
}

export const HANDS_FREE_IDLE_MS = 120_000; // as HANDS_FREE_IDLE_TIMEOUT in the CLI
const TICK_MS = 50;
const QUIET_RESTART_MS = 10_000; // nobody speaking: start a fresh recording so it stays small

/** Hands-free listening: one open microphone cut into utterances by an Endpointer. Each utterance
 * is its own MediaRecorder file (a webm file needs its header), started before the speech so the
 * first syllable is kept. `onUtterance` gets each finished recording; `onIdle` fires after
 * HANDS_FREE_IDLE_MS without speech. */
export class HandsFreeListener {
  onUtterance: ((recording: Blob) => void) | null = null;
  onIdle: (() => void) | null = null;
  onLevel: ((level: number, speaking: boolean) => void) | null = null; // each tick while not paused
  #stream: MediaStream | null = null;
  #context: AudioContext | null = null;
  #analyser: AnalyserNode | null = null;
  #samples = new Float32Array(0);
  #recorder: MediaRecorder | null = null;
  #chunks: Blob[] = [];
  #timer = 0;
  #endpointer = new Endpointer();
  #recordingSince = 0;
  #lastSpeech = 0;
  #paused = false;

  get listening(): boolean {
    return this.#stream !== null;
  }

  async start(): Promise<void> {
    if (this.#stream) return;

    this.#stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    this.#context = new AudioContext();
    await this.#context.resume();
    this.#analyser = this.#context.createAnalyser();
    this.#analyser.fftSize = 1024;
    this.#samples = new Float32Array(this.#analyser.fftSize);
    this.#context.createMediaStreamSource(this.#stream).connect(this.#analyser);
    this.#endpointer = new Endpointer();
    this.#paused = false;
    this.#lastSpeech = performance.now();
    this.#begin();
    this.#timer = window.setInterval(() => this.#tick(), TICK_MS);
  }

  /** Paused while an answer plays: what the microphone hears then is dropped. */
  pause(paused: boolean): void {
    if (paused === this.#paused || !this.#stream) return;

    this.#paused = paused;
    this.#endpointer.reset();
    this.#discard();

    if (!paused) {
      this.#lastSpeech = performance.now(); // the idle clock restarts after each answer
      this.#begin();
    }
  }

  stop(): void {
    window.clearInterval(this.#timer);
    this.#discard();
    this.#stream?.getTracks().forEach((track) => track.stop());
    void this.#context?.close().catch(() => undefined);
    this.#stream = null;
    this.#context = null;
    this.#analyser = null;
  }

  #begin(): void {
    if (!this.#stream) return;

    const mimeType = pickMimeType((type) => MediaRecorder.isTypeSupported(type));
    const recorder = new MediaRecorder(this.#stream, mimeType ? { mimeType } : {});
    const chunks: Blob[] = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    this.#recorder = recorder;
    this.#chunks = chunks;
    this.#recordingSince = performance.now();
    recorder.start();
  }

  /** The utterance is over: hand its recording on and start the next one. */
  #finish(): void {
    const recorder = this.#recorder;
    const chunks = this.#chunks; // the last data arrives just before onstop
    this.#recorder = null;

    if (recorder) {
      recorder.onstop = () => {
        if (chunks.length > 0) {
          this.onUtterance?.(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
        }
      };
      recorder.stop();
    }

    this.#begin();
  }

  #discard(): void {
    const recorder = this.#recorder;
    this.#recorder = null;

    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      recorder.stop();
    }
  }

  #tick(): void {
    if (!this.#analyser || this.#paused) return;

    const now = performance.now();
    this.#analyser.getFloatTimeDomainData(this.#samples);
    const level = rms(this.#samples);
    const event = this.#endpointer.update(level, now);
    this.onLevel?.(level, this.#endpointer.speaking);

    if (event !== null || this.#endpointer.speaking) {
      this.#lastSpeech = now;
    }

    if (event === "end") {
      this.#finish();
    } else if (this.#endpointer.speaking && now - this.#recordingSince > MAX_RECORDING_MS) {
      this.#endpointer.reset(); // a minute without a pause: send what there is
      this.#finish();
    } else if (!this.#endpointer.speaking && now - this.#recordingSince > QUIET_RESTART_MS) {
      this.#discard();
      this.#begin();
    }

    if (!this.#endpointer.speaking && now - this.#lastSpeech > HANDS_FREE_IDLE_MS) {
      this.onIdle?.();
    }
  }
}
