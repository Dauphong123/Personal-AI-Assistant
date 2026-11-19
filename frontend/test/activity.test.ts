// Tool activity: `tool` events become cards in the conversation, apart from the answer.
import assert from "node:assert/strict";
import { test } from "node:test";

import { activeTool, apply, applyAll, describeTool, emptyConversation, isChatEvent, isThinking, statusOf } from "../src/state.ts";
import type { ChatEvent } from "../src/types.ts";

const read = { name: "read_file", args: { path: "main.py" } };
const idle = { connection: "", voiceStatus: "", handsFree: false, transcribing: false, queued: 0 };

const turn: ChatEvent[] = [
  { id: 1, type: "user", text: "what does main do?" },
  { id: 2, type: "token", text: "Let me look." },
  { id: 3, type: "tool", call: "c1", ...read, status: "running" },
  { id: 4, type: "tool", call: "c1", ...read, status: "done", text: "print('hi')" },
  { id: 5, type: "token", text: "It prints hi." },
  { id: 6, type: "done" },
];

test("a tool call is one card, updated in place, between the narration and the answer", () => {
  const conv = applyAll(emptyConversation(), turn);
  assert.deepEqual(
    conv.items.map((item) => item.kind),
    ["user", "assistant", "tool", "assistant"],
  );
  assert.deepEqual(conv.items[2], { kind: "tool", call: "c1", ...read, status: "done", text: "print('hi')" });
  assert.equal(conv.items[1]?.kind === "assistant" && conv.items[1].text, "Let me look.");
  assert.equal(conv.items[3]?.kind === "assistant" && conv.items[3].text, "It prints hi.");
});

test("a running card has no output yet; several calls keep their order", () => {
  const conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "q" },
    { id: 2, type: "tool", call: "a", name: "read_file", args: { path: "a.py" }, status: "running" },
    { id: 3, type: "tool", call: "b", name: "read_file", args: { path: "b.py" }, status: "running" },
    { id: 4, type: "tool", call: "a", name: "read_file", args: { path: "a.py" }, status: "error", text: "ERROR: no such file" },
  ]);
  assert.deepEqual(
    conv.items.slice(1).map((item) => item.kind === "tool" && [item.call, item.status, item.text]),
    [
      ["a", "error", "ERROR: no such file"],
      ["b", "running", ""],
    ],
  );
});

test("a finished call whose start was missed still shows", () => {
  const conv = applyAll(emptyConversation(), [
    { id: 1, type: "user", text: "q" },
    { id: 2, type: "tool", call: "c1", ...read, status: "done", text: "x" },
  ]);
  assert.equal(conv.items.at(-1)?.kind, "tool");
});

test("the same call id in a later turn is a new card", () => {
  const conv = applyAll(emptyConversation(), [
    ...turn,
    { id: 7, type: "user", text: "again" },
    { id: 8, type: "tool", call: "c1", ...read, status: "running" },
  ]);
  assert.equal(conv.items.filter((item) => item.kind === "tool").length, 2);
});

test("stopping or failing a turn marks its running calls as stopped", () => {
  for (const end of [{ id: 3, type: "cancelled" }, { id: 3, type: "error", text: "boom" }] as ChatEvent[]) {
    const conv = applyAll(emptyConversation(), [
      { id: 1, type: "user", text: "q" },
      { id: 2, type: "tool", call: "c1", ...read, status: "running" },
      end,
    ]);
    const card = conv.items[1];
    assert.equal(card?.kind === "tool" && card.status, "stopped");
  }
});

test("activeTool: the call running now, if any", () => {
  const running = applyAll(emptyConversation(), turn.slice(0, 3));
  assert.equal(activeTool(running)?.call, "c1");
  assert.equal(activeTool(applyAll(emptyConversation(), turn.slice(0, 4))), null);
});

test("status: names the running tool", () => {
  const running = applyAll(emptyConversation(), turn.slice(0, 3));
  assert.deepEqual(statusOf({ ...idle, conv: running }), { text: "Reading main.py…", tone: "busy" });
  const approval = apply(running, { id: 4, type: "approval", calls: [{ name: "terminal", args: {} }], request: "r" });
  assert.equal(statusOf({ ...idle, conv: approval }).text, "Needs approval");
});

test("thinking: not while a tool card shows its own progress, again once it finished", () => {
  assert.equal(isThinking(applyAll(emptyConversation(), turn.slice(0, 3))), false);
  assert.equal(isThinking(applyAll(emptyConversation(), turn.slice(0, 4))), true);
});

test("isChatEvent: tool events need a call, a name, object args, a known status and text if any", () => {
  const ok = { id: 1, type: "tool", call: "c", name: "read_file", args: {}, status: "running" };
  assert.equal(isChatEvent(ok), true);
  assert.equal(isChatEvent({ ...ok, status: "done", text: "x" }), true);
  assert.equal(isChatEvent({ ...ok, call: 1 }), false);
  assert.equal(isChatEvent({ ...ok, name: undefined }), false);
  assert.equal(isChatEvent({ ...ok, args: "x" }), false);
  assert.equal(isChatEvent({ ...ok, args: null }), false);
  assert.equal(isChatEvent({ ...ok, args: [] }), false);
  assert.equal(isChatEvent({ ...ok, status: "maybe" }), false);
  assert.equal(isChatEvent({ ...ok, text: 3 }), false);
});

test("describeTool: what each tool does, in words, and what it works on", () => {
  const cases: [string, Record<string, unknown>, string, string, string | null][] = [
    ["read_file", { path: "graph/node.py" }, "Reading", "Read", "graph/node.py"],
    ["list_files", {}, "Listing files in", "Listed files in", "."],
    ["search_code", { query: "route" }, "Searching the code for", "Searched the code for", "route"],
    ["terminal", { command: "git status" }, "Running", "Ran", "git status"],
    ["job_status", {}, "Checking background jobs", "Checked background jobs", null],
    ["cancel_job", { job_id: 3 }, "Cancelling job", "Cancelled job", "3"],
    ["write_file", { path: "a.txt", content: "x" }, "Writing", "Wrote", "a.txt"],
    ["edit_file", { path: "a.txt" }, "Editing", "Edited", "a.txt"],
    ["delete_file", { path: "a.txt" }, "Deleting", "Deleted", "a.txt"],
    ["move_file", { source: "a", destination: "b" }, "Moving", "Moved", "a → b"],
    ["git_status", { dir: "." }, "Checking git status", "Checked git status", null],
    ["git_diff", {}, "Reading the git diff", "Read the git diff", null],
    ["git_log", {}, "Reading the git history", "Read the git history", null],
    ["git_show", { dir: ".", commit: "abc123" }, "Showing commit", "Showed commit", "abc123"],
    ["search_document", { query: "memory" }, "Searching the documents for", "Searched the documents for", "memory"],
    ["web_search", { query: "pgvector" }, "Searching the web for", "Searched the web for", "pgvector"],
    ["read_webpage", { url: "https://example.com" }, "Reading", "Read", "https://example.com"],
    ["frobnicate", { x: 1 }, "Using frobnicate", "Used frobnicate", null],
  ];
  for (const [name, args, doing, done, target] of cases) {
    assert.deepEqual(describeTool(name, args), { doing, done, target }, name);
  }
});

test("describeTool: a long or multi-line target becomes one short line", () => {
  const { target } = describeTool("terminal", { command: `python -c "${"x".repeat(200)}"\necho hi` });
  assert.ok(target !== null && target.length <= 80 && !target.includes("\n") && target.endsWith("…"));
});
