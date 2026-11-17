// Pure view logic: the backend's event log folded into what the page shows. No DOM here, so
// it runs under node --test.

import type { ChatEvent, SessionInfo, SessionState, ToolCall } from "./types.ts";

export type Item =
  | { kind: "user"; text: string }
  | { kind: "notice"; text: string } // what a finished background job reported
  | { kind: "assistant"; text: string }
  | { kind: "note"; text: string; error: boolean };

export interface Conversation {
  readonly items: readonly Item[];
  readonly busy: boolean; // a turn is running (or waiting for the model)
  readonly waiting: boolean; // queued behind another session's turn
  readonly approval: readonly ToolCall[] | null; // tool calls waiting for allow/deny
  readonly lastEventId: number; // resume point for the event stream
  readonly gone: boolean; // the session was deleted
}

export function emptyConversation(): Conversation {
  return { items: [], busy: false, waiting: false, approval: null, lastEventId: 0, gone: false };
}

/** The conversation after one event. Events at or before lastEventId are ignored, so a
 * replay after a reconnect changes nothing. */
export function apply(conv: Conversation, event: ChatEvent): Conversation {
  if (event.type === "gone") {
    return { ...conv, gone: true, busy: false, waiting: false, approval: null };
  }

  if (event.id <= conv.lastEventId) {
    return conv;
  }

  const next = { ...conv, lastEventId: event.id, approval: null };

  switch (event.type) {
    case "user":
      return { ...next, items: [...conv.items, { kind: "user", text: event.text }], busy: true, waiting: false };
    case "notice":
      return { ...next, items: [...conv.items, { kind: "notice", text: event.text }], busy: true, waiting: false };
    case "waiting":
      return { ...next, waiting: true };
    case "token": {
      const last = conv.items.at(-1);
      const items =
        last?.kind === "assistant"
          ? [...conv.items.slice(0, -1), { kind: "assistant" as const, text: last.text + event.text }]
          : [...conv.items, { kind: "assistant" as const, text: event.text }];
      return { ...next, items, waiting: false };
    }
    case "approval":
      return { ...next, approval: event.calls, waiting: false };
    case "done":
      return { ...next, busy: false, waiting: false };
    case "cancelled":
      return { ...endTurn(next, "Stopped.", false) };
    case "error":
      return { ...endTurn(next, event.text, true) };
    case "reset":
      return { ...next, items: [], busy: false, waiting: false };
  }
}

function endTurn(conv: Conversation, text: string, error: boolean): Conversation {
  return { ...conv, items: [...conv.items, { kind: "note", text, error }], busy: false, waiting: false };
}

export function applyAll(conv: Conversation, events: readonly ChatEvent[]): Conversation {
  return events.reduce(apply, conv);
}

export type Block = { kind: "prose"; text: string } | { kind: "code"; text: string };

/** Split a message into prose and fenced code (```lang ... ```). An unclosed fence counts as
 * code, so a code block looks right while it streams. */
export function blocks(text: string): Block[] {
  const result: Block[] = [];

  text.split("```").forEach((part, index) => {
    if (index % 2 === 1) {
      result.push({ kind: "code", text: part.includes("\n") ? part.replace(/^[^\n]*\n/, "") : "" });
    } else if (part.trim() !== "") {
      result.push({ kind: "prose", text: part.trim() });
    }
  });

  return result;
}

export function formatCalls(calls: readonly ToolCall[]): string {
  return calls.map((call) => `${call.name}(${JSON.stringify(call.args)})`).join("\n");
}

const STATE_LABELS: Record<SessionState, string> = {
  idle: "",
  running: "Working",
  waiting: "Waiting",
  approval: "Needs approval",
};

export function stateLabel(state: SessionState): string {
  return STATE_LABELS[state] ?? "";
}

export function sortSessions(sessions: readonly SessionInfo[]): SessionInfo[] {
  return [...sessions].sort((a, b) => b.created - a.created);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** Runtime check of a parsed stream message (the stream is ours, but a bad message must not
 * break the page). */
export function isChatEvent(value: unknown): value is ChatEvent {
  if (!isRecord(value)) return false;
  if (value["type"] === "gone") return true;
  if (typeof value["id"] !== "number") return false;

  switch (value["type"]) {
    case "user":
    case "notice":
    case "token":
    case "error":
      return typeof value["text"] === "string";
    case "approval":
      return Array.isArray(value["calls"]);
    case "waiting":
    case "done":
    case "cancelled":
    case "reset":
      return true;
    default:
      return false;
  }
}
