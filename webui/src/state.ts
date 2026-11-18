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

// --- status line ----------------------------------------------------------------------------

export type StatusTone = "" | "busy" | "approval" | "error" | "voice";

export interface StatusInput {
  conv: Conversation;
  connection: string; // stream trouble ("Reconnecting…")
  voiceStatus: string; // push-to-talk ("Listening…" / "Transcribing…")
  handsFree: boolean;
  transcribing: boolean; // a hands-free recording is at the backend
  queued: number; // spoken prompts waiting to be sent
}

/** The header's status text and how it should look. */
export function statusOf(input: StatusInput): { text: string; tone: StatusTone } {
  const { conv, connection, voiceStatus, handsFree, transcribing, queued } = input;

  if (connection) return { text: connection, tone: "error" };

  let text = "";
  let tone: StatusTone = "";

  if (voiceStatus) {
    [text, tone] = [voiceStatus, "voice"];
  } else if (conv.waiting) {
    [text, tone] = ["Waiting for another session", "busy"];
  } else if (conv.approval) {
    [text, tone] = [handsFree ? "Needs approval: say yes or no" : "Needs approval", "approval"];
  } else if (conv.busy) {
    [text, tone] = ["Working…", "busy"];
  }

  const suffix = queued ? ` · ${queued} queued` : "";

  if (handsFree) {
    return { text: `Hands-free: ${text || (transcribing ? "Transcribing…" : "Listening…")}${suffix}`, tone: tone || "voice" };
  }

  if (suffix) return { text: `${text || "Sending…"}${suffix}`, tone: tone || "busy" };

  return { text, tone };
}

/** A turn is running and no answer text has arrived yet (show the thinking placeholder). */
export function isThinking(conv: Conversation): boolean {
  return conv.busy && !conv.waiting && conv.approval === null && conv.items.at(-1)?.kind !== "assistant";
}

// --- relative times ---------------------------------------------------------------------------

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "just now", "5 min ago", "3 h ago", "yesterday", "4 days ago", then a date ("2 Oct"). */
export function relativeTime(epochSeconds: number, nowMs: number = Date.now()): string {
  const seconds = Math.max(0, nowMs / 1000 - epochSeconds);

  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 2 * 86400) return "yesterday";
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)} days ago`;

  const then = new Date(epochSeconds * 1000);
  const day = `${then.getDate()} ${MONTHS[then.getMonth()]}`;
  return then.getFullYear() === new Date(nowMs).getFullYear() ? day : `${day} ${then.getFullYear()}`;
}

// --- approval card ----------------------------------------------------------------------------

export interface CallField {
  label: string;
  text: string;
  code: boolean; // shown as code (commands, file text, raw values)
}

export interface CallView {
  name: string;
  title: string; // "Run a command", "Edit a file"...
  target: string | null; // the file it touches, if any
  danger: boolean; // cannot be undone from the chat (delete)
  fields: CallField[];
}

const asText = (value: unknown): string => (typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? String(value));

/** Tool calls waiting for approval, in words: what each does and to which file. */
export function describeCalls(calls: readonly ToolCall[]): CallView[] {
  return calls.map((call) => {
    const args = call.args;

    if (typeof args !== "object" || args === null || Array.isArray(args)) {
      return { name: call.name, title: `Use ${call.name}`, target: null, danger: false, fields: [{ label: "Arguments", text: asText(args), code: true }] };
    }

    const rest = new Map(Object.entries(args as Record<string, unknown>));
    const take = (key: string): string | null => {
      if (!rest.has(key)) return null;

      const value = rest.get(key);
      rest.delete(key);
      return asText(value);
    };
    const field = (label: string, key: string, code = true): CallField[] => {
      const text = take(key);
      return text === null ? [] : [{ label, text, code }];
    };

    let view: CallView;

    switch (call.name) {
      case "terminal":
        view = { name: call.name, title: "Run a command", target: null, danger: false, fields: field("Command", "command") };
        break;
      case "write_file":
        view = { name: call.name, title: "Write a file", target: take("path"), danger: false, fields: field("Content", "content") };
        break;
      case "edit_file": {
        const target = take("path");
        const fields = [...field("Replace", "old"), ...field("With", "new"), ...field("Occurrence", "occurrence", false)];
        view = { name: call.name, title: "Edit a file", target, danger: false, fields };
        break;
      }
      case "delete_file":
        view = { name: call.name, title: "Delete a file", target: take("path"), danger: true, fields: [] };
        break;
      case "move_file": {
        const source = take("source");
        const destination = take("destination");
        const target = source !== null && destination !== null ? `${source} → ${destination}` : (source ?? destination);
        view = { name: call.name, title: "Move a file", target, danger: false, fields: [] };
        break;
      }
      default:
        view = { name: call.name, title: `Use ${call.name}`, target: null, danger: false, fields: [] };
    }

    for (const [label, value] of rest) {
      view.fields.push({ label, text: asText(value), code: true });
    }

    return view;
  });
}
