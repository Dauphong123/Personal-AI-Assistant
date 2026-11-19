// Message actions: Retry / Try again resend the prompt of the turn that just ended.
import assert from "node:assert/strict";
import { test } from "node:test";

import { applyAll, emptyConversation, retryPrompt } from "../src/state.ts";
import type { ChatEvent } from "../src/types.ts";

const asked: ChatEvent[] = [
  { id: 1, type: "user", text: "first question" },
  { id: 2, type: "token", text: "first answer" },
  { id: 3, type: "done" },
  { id: 4, type: "user", text: "  run the tests  " },
];

const conv = (events: ChatEvent[]) => applyAll(emptyConversation(), events);

test("retry: the prompt of the turn that just answered", () => {
  assert.equal(retryPrompt(conv([...asked, { id: 5, type: "token", text: "done" }, { id: 6, type: "done" }])), "run the tests");
});

test("retry: after a stopped or failed turn too (Try again)", () => {
  assert.equal(retryPrompt(conv([...asked, { id: 5, type: "cancelled" }])), "run the tests");
  assert.equal(retryPrompt(conv([...asked, { id: 5, type: "error", text: "boom" }])), "run the tests");
});

test("retry: not while a turn runs, waits or needs approval", () => {
  assert.equal(retryPrompt(conv(asked)), null);
  assert.equal(retryPrompt(conv([...asked, { id: 5, type: "waiting" }])), null);
  assert.equal(retryPrompt(conv([...asked, { id: 5, type: "approval", calls: [], request: "r" }])), null);
});

test("retry: nothing to retry in an empty, cleared or deleted session", () => {
  assert.equal(retryPrompt(emptyConversation()), null);
  assert.equal(retryPrompt(conv([...asked.slice(0, 3), { id: 4, type: "reset" }])), null);
  assert.equal(retryPrompt(conv([...asked.slice(0, 3), { type: "gone" }])), null);
});

test("retry: a background job's notice turn is not the user's prompt", () => {
  const events: ChatEvent[] = [
    ...asked.slice(0, 3),
    { id: 4, type: "notice", text: "[job 1 finished]" },
    { id: 5, type: "token", text: "The job finished." },
    { id: 6, type: "done" },
  ];
  assert.equal(retryPrompt(conv(events)), null);
});

test("retry: a turn that ended on a tool card has no answer to retry from", () => {
  const events: ChatEvent[] = [
    ...asked,
    { id: 5, type: "tool", call: "c", name: "terminal", args: { command: "x" }, status: "done", text: "ok" },
    { id: 6, type: "done" },
  ];
  assert.equal(retryPrompt(conv(events)), null);
});
