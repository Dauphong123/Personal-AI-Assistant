// Voice in the browser: record a push-to-talk message, play spoken answers. The backend does the
// speech recognition and synthesis (POST /api/voice/transcribe, /api/voice/speak). Browser APIs
// are only touched when a method is called, so the pure helpers run under node --test.

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
  onAutoStop: (() => void) | null = null; // called when MAX_RECORDING_MS ends a recording

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
    this.#stream?.getTracks().forEach((track) => track.stop());
    this.#stream = null;
    this.#recorder = null;
  }
}

/** Plays one spoken answer at a time; a new one replaces the old. */
export class Player {
  #audio: HTMLAudioElement | null = null;
  #url = "";

  play(wav: Blob): void {
    this.stop();
    this.#url = URL.createObjectURL(wav);
    const audio = new Audio(this.#url);
    audio.onended = () => this.stop();
    this.#audio = audio;
    void audio.play().catch(() => this.stop()); // autoplay refused: stay silent
  }

  stop(): void {
    this.#audio?.pause();
    this.#audio = null;

    if (this.#url) {
      URL.revokeObjectURL(this.#url);
      this.#url = "";
    }
  }
}
