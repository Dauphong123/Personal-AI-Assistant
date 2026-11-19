// Pure view logic: the backend's event log folded into what the page shows. No DOM here, so
// it runs under node --test.

import type { ChatEvent, SessionInfo, SessionState, ToolCall, ToolStatus } from "./types.ts";

/** One tool call the agent made: a card in the conversation. `stopped` = the turn ended
 * (stop or error) before its result came. */
export interface ToolItem {
  kind: "tool";
  call: string;
  name: string;
  args: Record<string, unknown>;
  status: ToolStatus | "stopped";
  text: string; // the tool's output ("" while running)
}

export type Item =
  | { kind: "user"; text: string }
  | { kind: "notice"; text: string } // what a finished background job reported
  | { kind: "assistant"; text: string }
  | { kind: "note"; text: string; error: boolean }
  | ToolItem;

export interface Conversation {
  readonly items: readonly Item[];
  readonly busy: boolean; // a turn is running (or waiting for the model)
  readonly waiting: boolean; // queued behind another session's turn
  readonly approval: readonly ToolCall[] | null; // tool calls waiting for allow/deny
  readonly approvalRequest: string | null; // id of that request; the answer names it
  readonly lastEventId: number; // resume point for the event stream
  readonly gone: boolean; // the session was deleted
}

export function emptyConversation(): Conversation {
  return { items: [], busy: false, waiting: false, approval: null, approvalRequest: null, lastEventId: 0, gone: false };
}

/** The conversation after one event. Events at or before lastEventId are ignored, so a
 * replay after a reconnect changes nothing. */
export function apply(conv: Conversation, event: ChatEvent): Conversation {
  if (event.type === "gone") {
    return { ...conv, gone: true, busy: false, waiting: false, approval: null, approvalRequest: null };
  }

  if (event.id <= conv.lastEventId) {
    return conv;
  }

  const next = { ...conv, lastEventId: event.id, approval: null, approvalRequest: null };

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
    case "tool": {
      const card: ToolItem = { kind: "tool", call: event.call, name: event.name, args: event.args, status: event.status, text: event.text ?? "" };
      const at = toolIndex(conv.items, event.call);
      const items = at === -1 ? [...conv.items, card] : conv.items.map((item, index) => (index === at ? card : item));
      return { ...next, items, waiting: false };
    }
    case "approval":
      return { ...next, approval: event.calls, approvalRequest: event.request, waiting: false };
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

/** Where the current turn begins: just after its user message or job notice. */
function turnStart(items: readonly Item[]): number {
  for (let index = items.length - 1; index >= 0; index--) {
    const kind = items[index]?.kind;
    if (kind === "user" || kind === "notice") return index + 1;
  }

  return 0;
}

/** The card of this call in the current turn, or -1 (ids may repeat across turns). */
function toolIndex(items: readonly Item[], call: string): number {
  const start = turnStart(items);

  for (let index = items.length - 1; index >= start; index--) {
    const item = items[index];
    if (item?.kind === "tool" && item.call === call) return index;
  }

  return -1;
}

function endTurn(conv: Conversation, text: string, error: boolean): Conversation {
  const start = turnStart(conv.items);
  const items = conv.items.map((item, index): Item =>
    index >= start && item.kind === "tool" && item.status === "running" ? { ...item, status: "stopped" } : item,
  );
  return { ...conv, items: [...items, { kind: "note", text, error }], busy: false, waiting: false };
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
      return Array.isArray(value["calls"]) && typeof value["request"] === "string";
    case "tool":
      return (
        typeof value["call"] === "string" &&
        typeof value["name"] === "string" &&
        isRecord(value["args"]) &&
        !Array.isArray(value["args"]) &&
        (value["status"] === "running" || value["status"] === "done" || value["status"] === "error") &&
        (value["text"] === undefined || typeof value["text"] === "string")
      );
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

export type StatusTone = "" | "busy" | "approval" | "error" | "voice" | "listening";

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
    const tool = activeTool(conv);
    const view = tool && describeTool(tool.name, tool.args);
    [text, tone] = [view ? `${view.doing}${view.target === null ? "" : ` ${view.target}`}…` : "Working…", "busy"];
  }

  const suffix = queued ? ` · ${queued} queued` : "";

  if (handsFree) {
    return { text: `Hands-free: ${text || (transcribing ? "Transcribing…" : "Listening…")}${suffix}`, tone: tone || "listening" };
  }

  if (suffix) return { text: `${text || "Sending…"}${suffix}`, tone: tone || "busy" };

  return { text, tone };
}

/** A turn is running and nothing shows progress yet: no answer text, no running tool card
 * (show the thinking placeholder). */
export function isThinking(conv: Conversation): boolean {
  const last = conv.items.at(-1);
  return conv.busy && !conv.waiting && conv.approval === null && last?.kind !== "assistant" && !(last?.kind === "tool" && last.status === "running");
}

/** The tool call running now in the current turn, if any (the newest one). */
export function activeTool(conv: Conversation): ToolItem | null {
  const start = turnStart(conv.items);

  for (let index = conv.items.length - 1; index >= start; index--) {
    const item = conv.items[index];
    if (item?.kind === "tool" && item.status === "running") return item;
  }

  return null;
}

/** What Retry / Try again would send: the prompt of the turn that just ended with an answer or
 * a stop/error note. Null while a turn is running or waiting, and for a background job's
 * notice turn (not something the user asked). */
export function retryPrompt(conv: Conversation): string | null {
  const last = conv.items.at(-1);

  if (conv.busy || conv.waiting || conv.gone || conv.approval !== null) return null;
  if (last?.kind !== "assistant" && last?.kind !== "note") return null;

  const opener = conv.items[turnStart(conv.items) - 1];
  return opener?.kind === "user" && opener.text.trim() ? opener.text.trim() : null;
}

// --- tool cards -------------------------------------------------------------------------------

export interface ToolView {
  doing: string; // "Reading", while it runs
  done: string; // "Read", once it finished
  target: string | null; // what it works on, one short line
}

const TARGET_MAX = 80;

function shortTarget(value: unknown): string | null {
  if (value === undefined || value === null || value === "") return null;

  const text = typeof value === "string" ? value : JSON.stringify(value) ?? String(value);
  const line = text.split("\n", 1)[0] ?? "";

  if (line === text && line.length <= TARGET_MAX) return line;

  return `${line.slice(0, TARGET_MAX - 1).trimEnd()}…`;
}

// [doing, done, argument shown as the target]; tool names as in tools.py TOOLS (search_document
// is the RAG tool).
const TOOL_WORDS: Record<string, [string, string, string | null]> = {
  read_file: ["Reading", "Read", "path"],
  list_files: ["Listing files in", "Listed files in", "path"],
  search_code: ["Searching the code for", "Searched the code for", "query"],
  terminal: ["Running", "Ran", "command"],
  job_status: ["Checking background jobs", "Checked background jobs", null],
  cancel_job: ["Cancelling job", "Cancelled job", "job_id"],
  write_file: ["Writing", "Wrote", "path"],
  edit_file: ["Editing", "Edited", "path"],
  delete_file: ["Deleting", "Deleted", "path"],
  move_file: ["Moving", "Moved", null],
  git_status: ["Checking git status", "Checked git status", null],
  git_diff: ["Reading the git diff", "Read the git diff", null],
  git_log: ["Reading the git history", "Read the git history", null],
  git_show: ["Showing commit", "Showed commit", "commit"],
  search_document: ["Searching the documents for", "Searched the documents for", "query"],
  web_search: ["Searching the web for", "Searched the web for", "query"],
  read_webpage: ["Reading", "Read", "url"],
};

/** A tool call in words: "Reading" / "Read" + "graph/node.py". */
export function describeTool(name: string, args: Record<string, unknown>): ToolView {
  const words = TOOL_WORDS[name];

  if (!words) return { doing: `Using ${name}`, done: `Used ${name}`, target: null };

  const [doing, done, key] = words;
  let target: string | null;

  if (name === "move_file") {
    target = shortTarget(`${asText(args["source"] ?? "?")} → ${asText(args["destination"] ?? "?")}`);
  } else if (name === "list_files") {
    target = shortTarget(args["path"]) ?? ".";
  } else {
    target = key === null ? null : shortTarget(args[key]);
  }

  return { doing, done, target };
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
