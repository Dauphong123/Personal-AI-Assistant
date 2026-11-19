// Live progress: how long the turn has been running, and how long an answer took.
import assert from "node:assert/strict";
import { test } from "node:test";

import { applyAll, elapsedLabel, emptyConversation, isChatEvent, statusOf } from "../src/state.ts";
import type { ChatEvent } from "../src/types.ts";

const idle = { connection: "", voiceStatus: "", handsFree: false, transcribing: false, queued: 0 };
const T = 1_800_000_000; // seconds since the epoch

const timed: ChatEvent[] = [
  { id: 1, type: "user", text: "q", at: T },
  { id: 2, type: "token", text: "Hel", at: T + 3 },
  { id: 3, type: "token", text: "lo", at: T + 4 },
  { id: 4, type: "done", at: T + 42.4 },
];

test("elapsedLabel: seconds, then minutes and seconds, then minutes", () => {
  assert.equal(elapsedLabel(0), "0 s");
  assert.equal(elapsedLabel(-3), "0 s");
  assert.equal(elapsedLabel(9.7), "9 s");
  assert.equal(elapsedLabel(59.9), "59 s");
  assert.equal(elapsedLabel(60), "1 min");
  assert.equal(elapsedLabel(65), "1 min 5 s");
  assert.equal(elapsedLabel(600), "10 min");
  assert.equal(elapsedLabel(725), "12 min");
});

test("a turn remembers when it started; the answer gets its duration when it is done", () => {
  const running = applyAll(emptyConversation(), timed.slice(0, 3));
  assert.equal(running.turnStartedAt, T);
  const done = applyAll(emptyConversation(), timed);
  assert.equal(done.turnStartedAt, null);
  assert.deepEqual(done.items.at(-1), { kind: "assistant", text: "Hello", seconds: 42.4 });
});

test("without times (an older log) nothing is measured", () => {
  const conv = applyAll(
    emptyConversation(),
    timed.map(({ at: _at, ...event }) => event as ChatEvent),
  );
  assert.equal(conv.turnStartedAt, null);
  assert.deepEqual(conv.items.at(-1), { kind: "assistant", text: "Hello" });
});

test("only the turn's final answer is timed; a stopped turn is not", () => {
  const conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "q", at: T },
    { id: 2, type: "token", text: "Looking.", at: T + 1 },
    { id: 3, type: "tool", call: "c", name: "read_file", args: {}, status: "done", text: "x", at: T + 2 },
    { id: 4, type: "token", text: "Answer.", at: T + 3 },
    { id: 5, type: "done", at: T + 5 },
    { id: 6, type: "user", text: "again", at: T + 10 },
    { id: 7, type: "token", text: "Part", at: T + 11 },
    { id: 8, type: "cancelled", at: T + 12 },
  ]);
  const answers = conv.items.filter((item) => item.kind === "assistant");
  assert.deepEqual(
    answers.map((item) => item.seconds),
    [undefined, 5, undefined],
  );
});

test("status: the time the turn has been running, apart from the text (not announced each second)", () => {
  const running = applyAll(emptyConversation(), timed.slice(0, 1));
  assert.deepEqual(statusOf({ ...idle, conv: running, now: (T + 12.5) * 1000 }), { text: "Working…", tone: "busy", elapsed: "12 s" });
  assert.deepEqual(statusOf({ ...idle, conv: running }), { text: "Working…", tone: "busy" });
  const done = applyAll(emptyConversation(), timed);
  assert.deepEqual(statusOf({ ...idle, conv: done, now: (T + 50) * 1000 }), { text: "", tone: "" });
});

test("isChatEvent: a time, when given, is a number", () => {
  assert.equal(isChatEvent({ id: 1, type: "done", at: T }), true);
  assert.equal(isChatEvent({ id: 1, type: "done", at: "now" }), false);
});
