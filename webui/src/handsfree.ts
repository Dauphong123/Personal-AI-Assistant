// Wispr Flow-style voice, the pure parts: telling a hold from a tap, finding where speech starts
// and ends in microphone levels, and deciding what a hands-free utterance does. No browser APIs
// here, so all of it runs under node --test.

/** What the backend heard in a dictated recording (POST /api/voice/transcribe?dictate=1). */
export interface Dictation {
  text: string; // raw transcript
  prompt: string; // cleaned up, "" when nothing usable was said; "stop" / "exit" kept plain
  hands_free: "on" | "off" | null; // "hands free" / "stop listening"
  answer: string; // as an approval answer: "y", "n" or "stop"
}

export const TAP_MAX_MS = 300; // a shorter press is a tap, a longer one a hold
export const DOUBLE_TAP_GAP_MS = 400; // the second tap must start this soon after the first ended

export type Release = "hold" | "tap" | "double_tap";

/** The push-to-talk key or button: hold to talk, double tap for hands-free (as in the CLI). */
export class PressTracker {
  #pressedAt = 0;
  #lastTapEnd: number | null = null;
  #second = false;

  down(now: number): void {
    this.#pressedAt = now;
    this.#second = this.#lastTapEnd !== null && now - this.#lastTapEnd <= DOUBLE_TAP_GAP_MS;
  }

  up(now: number): Release {
    if (now - this.#pressedAt > TAP_MAX_MS) {
      this.#lastTapEnd = null;
      return "hold";
    }

    if (this.#second) {
      this.#lastTapEnd = null;
      return "double_tap";
    }

    this.#lastTapEnd = now;
    return "tap";
  }
}

/** Root mean square of audio samples in [-1, 1]. */
export function rms(samples: ArrayLike<number>): number {
  if (samples.length === 0) return 0;

  let sum = 0;

  for (let i = 0; i < samples.length; i += 1) {
    const value = samples[i] ?? 0;
    sum += value * value;
  }

  return Math.sqrt(sum / samples.length);
}

export interface EndpointerOptions {
  startMs?: number; // loud this long = speech started (a click or a cough is shorter)
  endSilenceMs?: number; // quiet this long = the utterance is over
  minLevel?: number; // never treat quieter than this as speech
}

const MAX_START_FLOOR = 0.05; // the first level seen may be speech: do not trust it fully
const FLOOR_RISE = 0.05; // the noise floor follows a louder room slowly...
const FLOOR_FALL = 0.3; // ...and a quieter one quickly
const OVER_FLOOR = 3; // speech is this many times louder than the background

/** Splits a stream of microphone levels into utterances: "start" when speech begins, "end"
 * after a pause. The threshold follows the room's noise floor, so a fan is not speech. */
export class Endpointer {
  speaking = false;
  #startMs: number;
  #endSilenceMs: number;
  #minLevel: number;
  #floor: number | null = null;
  #loudSince: number | null = null;
  #quietSince: number | null = null;

  constructor(options: EndpointerOptions = {}) {
    this.#startMs = options.startMs ?? 150;
    this.#endSilenceMs = options.endSilenceMs ?? 900;
    this.#minLevel = options.minLevel ?? 0.02;
  }

  get threshold(): number {
    return Math.max(this.#minLevel, (this.#floor ?? 0) * OVER_FLOOR);
  }

  update(level: number, now: number): "start" | "end" | null {
    if (this.#floor === null) {
      this.#floor = Math.min(level, MAX_START_FLOOR);
    }

    const loud = level >= this.threshold;

    if (!this.speaking) {
      if (!loud) {
        this.#loudSince = null;
        const rate = level > this.#floor ? FLOOR_RISE : FLOOR_FALL;
        this.#floor += (level - this.#floor) * rate;
        return null;
      }

      this.#loudSince ??= now;

      if (now - this.#loudSince >= this.#startMs) {
        this.speaking = true;
        this.#quietSince = null;
        return "start";
      }

      return null;
    }

    if (loud) {
      this.#quietSince = null;
      return null;
    }

    this.#quietSince ??= now;

    if (now - this.#quietSince >= this.#endSilenceMs) {
      this.speaking = false;
      this.#loudSince = null;
      return "end";
    }

    return null;
  }

  reset(): void {
    this.speaking = false;
    this.#loudSince = null;
    this.#quietSince = null;
  }
}

export type HandsFreeAction =
  | { kind: "send"; text: string }
  | { kind: "approve"; allow: boolean }
  | { kind: "stop" }
  | { kind: "off" }
  | { kind: "ignore" };

/** What one hands-free utterance does. While a tool call waits for approval, speech answers it
 * (doubt or a long answer means no, as in the CLI); silence and noise never count as an answer. */
export function handsFreeAction(said: Dictation, approvalPending: boolean): HandsFreeAction {
  if (said.hands_free === "off") return { kind: "off" };
  if (!said.prompt) return { kind: "ignore" };

  if (approvalPending) {
    if (said.answer === "stop") return { kind: "stop" };
    return { kind: "approve", allow: said.answer === "y" };
  }

  if (said.prompt === "stop") return { kind: "stop" };
  if (said.hands_free === "on") return { kind: "ignore" }; // already on

  return { kind: "send", text: said.prompt };
}

/** Put dictated text into the message box at the cursor (replacing a selection), with a space
 * on either side where it would otherwise run into a word. */
export function insertDictation(
  value: string,
  start: number,
  end: number,
  text: string,
): { value: string; cursor: number } {
  const before = value.slice(0, start);
  const after = value.slice(end);
  const lead = before && !/\s$/.test(before) ? " " : "";
  const tail = after && !/^\s/.test(after) ? " " : "";
  const inserted = `${lead}${text}${tail}`;
  return { value: before + inserted + after, cursor: before.length + lead.length + text.length };
}

// --- voice meter (what the composer shows while the microphone is open) ----------------------

/** "0:03", "1:00". */
export function formatDuration(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

/** Push-to-talk time; in the last quarter the one-minute cap is shown too. */
export function recordingLabel(ms: number, maxMs: number): string {
  const shown = Math.min(ms, maxMs);
  return shown >= maxMs * 0.75 ? `${formatDuration(shown)} of ${formatDuration(maxMs)}` : formatDuration(shown);
}

/** Microphone RMS to 0..1 on a decibel scale: -60 dB and below is 0, -15 dB and above is 1. */
export function levelToUnit(level: number): number {
  if (!(level > 0)) return 0;

  const db = 20 * Math.log10(level);
  return Math.min(1, Math.max(0, (db + 60) / 45));
}

const BAR_SHAPE = [0.55, 0.8, 1, 0.8, 0.55];
const BAR_FLOOR = 0.12;

/** Heights (0..1) of the meter's five bars for one level; never below a small floor. */
export function meterBars(level: number): number[] {
  const unit = levelToUnit(level);
  return BAR_SHAPE.map((shape) => Math.max(BAR_FLOOR, Math.round(unit * shape * 100) / 100));
}

export interface HandsFreeState {
  speaking: boolean; // the endpointer hears speech now
  paused: boolean; // an answer is playing (listening is paused)
  transcribing: boolean; // an utterance is at the backend
  queued: number; // prompts waiting to be sent
}

export function handsFreeLabel(state: HandsFreeState): string {
  const base = state.paused
    ? "Paused while the answer plays"
    : state.speaking
      ? "Hearing you"
      : state.transcribing
        ? "Transcribing"
        : "Listening";

  if (!state.queued) return base;

  return `${base}, ${state.queued} ${state.queued === 1 ? "prompt" : "prompts"} queued`;
}
