import assert from "node:assert/strict";
import { test } from "node:test";

import {
  Endpointer,
  PressTracker,
  formatDuration,
  handsFreeLabel,
  handsFreeAction,
  insertDictation,
  levelToUnit,
  meterBars,
  recordingLabel,
  rms,
} from "../src/handsfree.ts";

test("a long press is a hold, a short one a tap, two quick taps a double tap", () => {
  const key = new PressTracker();
  key.down(0);
  assert.equal(key.up(500), "hold");

  key.down(1000);
  assert.equal(key.up(1100), "tap");
  key.down(1250);
  assert.equal(key.up(1350), "double_tap");

  key.down(1400); // a third quick tap starts over
  assert.equal(key.up(1450), "tap");
});

test("slow taps stay single and a hold breaks a double tap", () => {
  const key = new PressTracker();
  key.down(0);
  key.up(100);
  key.down(2000);
  assert.equal(key.up(2100), "tap");
  key.down(2200);
  assert.equal(key.up(2900), "hold");
});

test("rms of silence is 0 and of a full-scale square wave 1", () => {
  assert.equal(rms(new Float32Array(8)), 0);
  assert.equal(rms(Float32Array.from([1, -1, 1, -1])), 1);
  assert.equal(rms(new Float32Array(0)), 0);
});

function feed(endpointer: Endpointer, steps: [number, number][]): string[] {
  return steps.map(([level, now]) => endpointer.update(level, now)).filter((e): e is "start" | "end" => e !== null);
}

test("speech starts after a short loud stretch and ends after a pause", () => {
  const ep = new Endpointer({ startMs: 100, endSilenceMs: 500 });
  const quiet: [number, number][] = [0, 50, 100, 150].map((t) => [0.001, t]);
  const loud: [number, number][] = [200, 250, 300, 350, 400].map((t) => [0.2, t]);
  const pause: [number, number][] = [450, 600, 800, 950].map((t) => [0.001, t]);
  assert.deepEqual(feed(ep, [...quiet, ...loud, ...pause]), ["start", "end"]);
  assert.equal(ep.speaking, false);
});

test("a click is too short to count as speech", () => {
  const ep = new Endpointer({ startMs: 100, endSilenceMs: 500 });
  assert.deepEqual(feed(ep, [[0.001, 0], [0.3, 50], [0.001, 100], [0.001, 200]]), []);
});

test("a short breath inside speech does not end it", () => {
  const ep = new Endpointer({ startMs: 100, endSilenceMs: 500 });
  const steps: [number, number][] = [
    [0.2, 0], [0.2, 100], [0.2, 200], // start
    [0.001, 300], [0.001, 500], // 200 ms breath
    [0.2, 600], [0.2, 700],
  ];
  assert.deepEqual(feed(ep, steps), ["start"]);
  assert.equal(ep.speaking, true);
});

test("the threshold follows the room's noise floor", () => {
  const ep = new Endpointer({ startMs: 100, endSilenceMs: 500 });
  // A steady fan at 0.05 is background, not speech.
  const fan: [number, number][] = Array.from({ length: 40 }, (_, i) => [0.05, i * 50]);
  assert.deepEqual(feed(ep, fan), []);
  // Speech well above it still counts.
  const voice: [number, number][] = Array.from({ length: 5 }, (_, i) => [0.4, 2000 + i * 50]);
  assert.deepEqual(feed(ep, voice), ["start"]);
});

const said = (prompt: string, extra: Partial<{ hands_free: "on" | "off" | null; answer: string }> = {}) => ({
  text: prompt,
  prompt,
  hands_free: null,
  answer: "n",
  ...extra,
});

test("hands-free: a prompt is sent, noise ignored", () => {
  assert.deepEqual(handsFreeAction(said("Run the tests."), false), { kind: "send", text: "Run the tests." });
  assert.deepEqual(handsFreeAction(said(""), false), { kind: "ignore" });
});

test("hands-free: stop listening, stop the turn", () => {
  assert.deepEqual(handsFreeAction(said("Stop listening.", { hands_free: "off" }), false), { kind: "off" });
  assert.deepEqual(handsFreeAction(said("stop"), false), { kind: "stop" });
});

test("hands-free: while an approval waits, speech answers it", () => {
  assert.deepEqual(handsFreeAction(said("Yes.", { answer: "y" }), true), { kind: "approve", allow: true });
  assert.deepEqual(handsFreeAction(said("No.", { answer: "n" }), true), { kind: "approve", allow: false });
  assert.deepEqual(handsFreeAction(said("stop", { answer: "stop" }), true), { kind: "stop" });
  assert.deepEqual(handsFreeAction(said("", { answer: "n" }), true), { kind: "ignore" }); // silence is not a no
});

test("dictation is inserted at the cursor with spacing", () => {
  assert.deepEqual(insertDictation("", 0, 0, "Open it."), { value: "Open it.", cursor: 8 });
  assert.deepEqual(insertDictation("Please", 6, 6, "open it."), { value: "Please open it.", cursor: 15 });
  assert.deepEqual(insertDictation("A  B", 2, 2, "x"), { value: "A x B", cursor: 3 });
  assert.deepEqual(insertDictation("old text", 0, 8, "New."), { value: "New.", cursor: 4 });
});

// --- voice meter ---------------------------------------------------------------------------

test("formatDuration: minutes and seconds", () => {
  assert.equal(formatDuration(0), "0:00");
  assert.equal(formatDuration(3_400), "0:03");
  assert.equal(formatDuration(59_999), "0:59");
  assert.equal(formatDuration(60_000), "1:00");
  assert.equal(formatDuration(125_000), "2:05");
  assert.equal(formatDuration(-5), "0:00");
});

test("recordingLabel: the cap is shown only near the end", () => {
  assert.equal(recordingLabel(5_000, 60_000), "0:05");
  assert.equal(recordingLabel(44_000, 60_000), "0:44");
  assert.equal(recordingLabel(50_000, 60_000), "0:50 of 1:00");
  assert.equal(recordingLabel(70_000, 60_000), "1:00 of 1:00");
});

test("levelToUnit: silence is 0, loud speech is 1, louder is never less", () => {
  assert.equal(levelToUnit(0), 0);
  assert.equal(levelToUnit(0.0005), 0); // below -60 dB
  assert.equal(levelToUnit(1), 1);
  let previous = -1;
  for (const level of [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]) {
    const unit = levelToUnit(level);
    assert.ok(unit >= previous && unit >= 0 && unit <= 1, `${level} -> ${unit}`);
    previous = unit;
  }
  assert.ok(levelToUnit(0.05) > 0.5); // normal speech fills most of the meter
});

test("meterBars: five bars, tallest in the middle, all small when silent", () => {
  const loud = meterBars(0.2);
  assert.equal(loud.length, 5);
  assert.ok((loud[2] ?? 0) >= (loud[0] ?? 0));
  assert.ok(loud.every((bar) => bar >= 0.12 && bar <= 1));
  assert.deepEqual(meterBars(0), [0.12, 0.12, 0.12, 0.12, 0.12]);
});

test("handsFreeLabel: what the hands-free strip says", () => {
  const quiet = { speaking: false, paused: false, transcribing: false, queued: 0 };
  assert.equal(handsFreeLabel(quiet), "Listening");
  assert.equal(handsFreeLabel({ ...quiet, speaking: true }), "Hearing you");
  assert.equal(handsFreeLabel({ ...quiet, paused: true, speaking: true }), "Paused while the answer plays");
  assert.equal(handsFreeLabel({ ...quiet, transcribing: true }), "Transcribing");
  assert.equal(handsFreeLabel({ ...quiet, queued: 2 }), "Listening, 2 prompts queued");
  assert.equal(handsFreeLabel({ ...quiet, speaking: true, queued: 1 }), "Hearing you, 1 prompt queued");
});
