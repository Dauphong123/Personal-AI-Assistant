import assert from "node:assert/strict";
import { test } from "node:test";

import { applyAll, emptyConversation } from "../src/state.ts";
import type { ChatEvent } from "../src/types.ts";
import { pickMimeType, replyToSpeak } from "../src/voice.ts";

test("pickMimeType prefers webm/opus, then ogg/opus, else the browser default", () => {
  assert.equal(pickMimeType(() => true), "audio/webm;codecs=opus");
  assert.equal(pickMimeType((t) => t.startsWith("audio/ogg")), "audio/ogg;codecs=opus");
  assert.equal(pickMimeType(() => false), "");
});

const turn: ChatEvent[] = [
  { id: 1, type: "user", text: "run the tests" },
  { id: 2, type: "token", text: "Running " },
  { id: 3, type: "token", text: "them now." },
  { id: 4, type: "token", text: " All passed." },
  { id: 5, type: "done" },
];

test("a live done event speaks the turn's answer", () => {
  const conv = applyAll(emptyConversation(), turn);
  assert.equal(replyToSpeak(conv, { id: 5, type: "done" }, 0), "Running them now. All passed.");
});

test("replayed history is never spoken", () => {
  const conv = applyAll(emptyConversation(), turn);
  assert.equal(replyToSpeak(conv, { id: 5, type: "done" }, 5), null);
});

test("only done events speak", () => {
  const conv = applyAll(emptyConversation(), turn.slice(0, 3));
  assert.equal(replyToSpeak(conv, { id: 3, type: "token", text: "x" }, 0), null);
  assert.equal(replyToSpeak(conv, { id: 9, type: "cancelled" }, 0), null);
});

test("a notice turn's answer is spoken too", () => {
  const conv = applyAll(emptyConversation(), [
    ...turn,
    { id: 6, type: "notice", text: "Background job 1 ended: build" },
    { id: 7, type: "token", text: "The build finished." },
    { id: 8, type: "done" },
  ]);
  assert.equal(replyToSpeak(conv, { id: 8, type: "done" }, 5), "The build finished.");
});

test("a turn with no answer text says nothing", () => {
  const conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "hi" },
    { id: 2, type: "done" },
  ]);
  assert.equal(replyToSpeak(conv, { id: 2, type: "done" }, 0), null);
});
