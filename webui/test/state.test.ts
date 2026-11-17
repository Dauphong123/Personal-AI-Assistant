import assert from "node:assert/strict";
import { test } from "node:test";

import {
  apply,
  applyAll,
  blocks,
  emptyConversation,
  formatCalls,
  isChatEvent,
  isThinking,
  statusOf,
  sortSessions,
  stateLabel,
} from "../src/state.ts";
import type { ChatEvent, SessionInfo } from "../src/types.ts";

const turn: ChatEvent[] = [
  { id: 1, type: "user", text: "hi" },
  { id: 2, type: "token", text: "Hel" },
  { id: 3, type: "token", text: "lo" },
  { id: 4, type: "done" },
];

test("a turn becomes a user and an assistant message", () => {
  const conv = applyAll(emptyConversation(), turn);
  assert.deepEqual(conv.items, [
    { kind: "user", text: "hi" },
    { kind: "assistant", text: "Hello" },
  ]);
  assert.equal(conv.busy, false);
  assert.equal(conv.lastEventId, 4);
});

test("user message marks the conversation busy until done", () => {
  const conv = apply(emptyConversation(), { id: 1, type: "user", text: "hi" });
  assert.equal(conv.busy, true);
});

test("apply does not change its input", () => {
  const start = emptyConversation();
  const next = apply(start, { id: 1, type: "user", text: "hi" });
  assert.notEqual(next, start);
  assert.deepEqual(start.items, []);
  const more = apply(next, { id: 2, type: "token", text: "a" });
  apply(more, { id: 3, type: "token", text: "b" });
  assert.deepEqual(more.items.at(-1), { kind: "assistant", text: "a" });
});

test("tokens after a new user message start a new assistant message", () => {
  const conv = applyAll(emptyConversation(), [
    ...turn,
    { id: 5, type: "user", text: "again" },
    { id: 6, type: "token", text: "Sure" },
  ]);
  assert.deepEqual(
    conv.items.map((item) => item.kind),
    ["user", "assistant", "user", "assistant"],
  );
  assert.equal(conv.items.at(-1)?.text, "Sure");
});

test("events already applied are ignored (reconnect replay)", () => {
  const once = applyAll(emptyConversation(), turn);
  const twice = applyAll(once, turn);
  assert.deepEqual(twice, once);
});

test("waiting is shown until the turn starts streaming", () => {
  let conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "hi" },
    { id: 2, type: "waiting" },
  ]);
  assert.equal(conv.waiting, true);
  conv = apply(conv, { id: 3, type: "token", text: "x" });
  assert.equal(conv.waiting, false);
});

test("approval is pending until the next event", () => {
  const calls = [{ name: "write_file", args: { path: "a.txt" } }];
  let conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "go" },
    { id: 2, type: "approval", calls },
  ]);
  assert.deepEqual(conv.approval, calls);
  conv = apply(conv, { id: 3, type: "token", text: "ok" });
  assert.equal(conv.approval, null);
});

test("cancelled and error add notes and end the turn", () => {
  const conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "a" },
    { id: 2, type: "cancelled" },
    { id: 3, type: "user", text: "b" },
    { id: 4, type: "error", text: "model down" },
  ]);
  assert.deepEqual(conv.items.filter((item) => item.kind === "note"), [
    { kind: "note", text: "Stopped.", error: false },
    { kind: "note", text: "model down", error: true },
  ]);
  assert.equal(conv.busy, false);
});

test("reset clears the conversation but keeps the event cursor", () => {
  const conv = applyAll(emptyConversation(), [...turn, { id: 5, type: "reset" }]);
  assert.deepEqual(conv.items, []);
  assert.equal(conv.lastEventId, 5);
  assert.equal(conv.busy, false);
});

test("gone marks the conversation as deleted", () => {
  const conv = apply(applyAll(emptyConversation(), turn), { type: "gone" });
  assert.equal(conv.gone, true);
  assert.equal(conv.lastEventId, 4);
});

test("a notice starts a turn of its own, shown as a notice item", () => {
  const conv = applyAll(emptyConversation(), [
    ...turn,
    { id: 5, type: "notice", text: "Background job 2 ended: pytest" },
    { id: 6, type: "token", text: "Tests passed." },
  ]);
  assert.deepEqual(conv.items.slice(2), [
    { kind: "notice", text: "Background job 2 ended: pytest" },
    { kind: "assistant", text: "Tests passed." },
  ]);
  assert.equal(conv.busy, true);
  assert.equal(apply(conv, { id: 7, type: "done" }).busy, false);
});

test("isChatEvent accepts a notice with text only", () => {
  assert.equal(isChatEvent({ id: 1, type: "notice", text: "x" }), true);
  assert.equal(isChatEvent({ id: 1, type: "notice" }), false);
});

test("blocks split fenced code from prose", () => {
  assert.deepEqual(blocks("Look:\n```python\nprint(1)\n```\nDone."), [
    { kind: "prose", text: "Look:" },
    { kind: "code", text: "print(1)\n" },
    { kind: "prose", text: "Done." },
  ]);
});

test("blocks keep an unfinished code fence as code while streaming", () => {
  assert.deepEqual(blocks("a\n```js\nlet x"), [
    { kind: "prose", text: "a" },
    { kind: "code", text: "let x" },
  ]);
});

test("blocks drop blank prose", () => {
  assert.deepEqual(blocks("   "), []);
});

test("formatCalls shows one call per line", () => {
  assert.equal(
    formatCalls([
      { name: "terminal", args: { command: "ls" } },
      { name: "delete_file", args: { path: "x" } },
    ]),
    'terminal({"command":"ls"})\ndelete_file({"path":"x"})',
  );
});

test("isChatEvent accepts known events and rejects junk", () => {
  assert.equal(isChatEvent({ id: 1, type: "token", text: "a" }), true);
  assert.equal(isChatEvent({ type: "gone" }), true);
  assert.equal(isChatEvent({ id: 1, type: "token" }), false);
  assert.equal(isChatEvent({ id: 1, type: "approval", calls: "x" }), false);
  assert.equal(isChatEvent({ id: 1, type: "nope" }), false);
  assert.equal(isChatEvent({ type: "done" }), false);
  assert.equal(isChatEvent(null), false);
  assert.equal(isChatEvent("token"), false);
});

test("sortSessions puts the newest first", () => {
  const make = (id: string, created: number): SessionInfo => ({
    id,
    title: id,
    state: "idle",
    created,
    last_event: 0,
  });
  const sorted = sortSessions([make("a", 1), make("c", 3), make("b", 2)]);
  assert.deepEqual(
    sorted.map((s) => s.id),
    ["c", "b", "a"],
  );
});

test("stateLabel names every busy state and nothing for idle", () => {
  assert.equal(stateLabel("idle"), "");
  assert.equal(stateLabel("running"), "Working");
  assert.equal(stateLabel("waiting"), "Waiting");
  assert.equal(stateLabel("approval"), "Needs approval");
});

// --- status line and thinking indicator -------------------------------------------------

const idle = { connection: "", voiceStatus: "", handsFree: false, transcribing: false, queued: 0 };

test("status: nothing to say when idle", () => {
  assert.deepEqual(statusOf({ ...idle, conv: emptyConversation() }), { text: "", tone: "" });
});

test("status: a running turn, a waiting turn and an approval", () => {
  const busy = apply(emptyConversation(), { id: 1, type: "user", text: "hi" });
  assert.deepEqual(statusOf({ ...idle, conv: busy }), { text: "Working…", tone: "busy" });
  const waiting = apply(busy, { id: 2, type: "waiting" });
  assert.deepEqual(statusOf({ ...idle, conv: waiting }), { text: "Waiting for another session", tone: "busy" });
  const approval = apply(busy, { id: 2, type: "approval", calls: [{ name: "terminal", args: {} }] });
  assert.deepEqual(statusOf({ ...idle, conv: approval }), { text: "Needs approval", tone: "approval" });
  assert.deepEqual(statusOf({ ...idle, conv: approval, handsFree: true }), {
    text: "Hands-free: Needs approval: say yes or no",
    tone: "approval",
  });
});

test("status: connection trouble wins over everything", () => {
  const busy = apply(emptyConversation(), { id: 1, type: "user", text: "hi" });
  assert.deepEqual(statusOf({ ...idle, conv: busy, connection: "Reconnecting…", handsFree: true, queued: 2 }), {
    text: "Reconnecting…",
    tone: "error",
  });
});

test("status: voice states and the hands-free queue", () => {
  const conv = emptyConversation();
  assert.deepEqual(statusOf({ ...idle, conv, voiceStatus: "Listening…" }), { text: "Listening…", tone: "voice" });
  assert.deepEqual(statusOf({ ...idle, conv, handsFree: true }), { text: "Hands-free: Listening…", tone: "voice" });
  assert.deepEqual(statusOf({ ...idle, conv, handsFree: true, transcribing: true, queued: 2 }), {
    text: "Hands-free: Transcribing… · 2 queued",
    tone: "voice",
  });
  assert.deepEqual(statusOf({ ...idle, conv, queued: 1 }), { text: "Sending… · 1 queued", tone: "busy" });
});

test("thinking: a running turn with no answer text yet", () => {
  const asked = apply(emptyConversation(), { id: 1, type: "user", text: "hi" });
  assert.equal(isThinking(emptyConversation()), false);
  assert.equal(isThinking(asked), true);
  assert.equal(isThinking(apply(asked, { id: 2, type: "token", text: "He" })), false);
  assert.equal(isThinking(apply(asked, { id: 2, type: "waiting" })), false);
  assert.equal(isThinking(apply(asked, { id: 2, type: "approval", calls: [] })), false);
  assert.equal(isThinking(apply(asked, { id: 2, type: "done" })), false);
});

test("thinking: after a tool approval the answer goes on, so no new placeholder", () => {
  const conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "run it" },
    { id: 2, type: "token", text: "Running." },
    { id: 3, type: "approval", calls: [] },
  ]);
  // Approved: the event that follows clears the approval while the turn goes on.
  const resumed = apply(conv, { id: 4, type: "token", text: " Done." });
  assert.equal(isThinking(resumed), false);
});
