// The page: a sessions sidebar and the active session's conversation.
//
// Everything the model or a web page wrote is untrusted: it is only ever inserted as text
// (textContent / createTextNode), never as HTML.

import { Api } from "./api.ts";
import {
  apply,
  blocks,
  emptyConversation,
  formatCalls,
  isChatEvent,
  sortSessions,
  stateLabel,
} from "./state.ts";
import type { Conversation, Item } from "./state.ts";
import type { ServerInfo, SessionInfo } from "./types.ts";
import { sessionFromHash, tokenFrom } from "./url.ts";

const TOKEN_KEY = "coding-assistant-token";
const POLL_MS = 2000;
const NEAR_BOTTOM_PX = 80;

function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);

  if (found === null) {
    throw new Error(`The page is missing #${id}`);
  }

  return found as T;
}

const ui = {
  sidebar: byId("sidebar"),
  sessions: byId<HTMLUListElement>("sessions"),
  newSession: byId<HTMLButtonElement>("new-session"),
  toggleSidebar: byId<HTMLButtonElement>("toggle-sidebar"),
  title: byId("session-title"),
  status: byId("status"),
  rename: byId<HTMLButtonElement>("rename"),
  clear: byId<HTMLButtonElement>("clear-chat"),
  remove: byId<HTMLButtonElement>("delete-session"),
  scroll: byId("scroll"),
  empty: byId("empty"),
  log: byId("log"),
  alert: byId("alert"),
  approval: byId("approval"),
  approvalCalls: byId("approval-calls"),
  allow: byId<HTMLButtonElement>("allow"),
  deny: byId<HTMLButtonElement>("deny"),
  form: byId<HTMLFormElement>("form"),
  input: byId<HTMLTextAreaElement>("input"),
  send: byId<HTMLButtonElement>("send"),
  stop: byId<HTMLButtonElement>("stop"),
};

// --- token ---------------------------------------------------------------------

function storedToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function storeToken(value: string): void {
  try {
    sessionStorage.setItem(TOKEN_KEY, value);
  } catch {
    // Storage blocked: the token then lasts until the page is reloaded.
  }
}

const launch = tokenFrom(location.search, storedToken());

if (launch.token) {
  storeToken(launch.token);
}

// Keep the token out of the address bar and the history (the hash names the session).
history.replaceState(null, "", location.pathname + launch.search + location.hash);

const api = new Api(launch.token);

// --- state ---------------------------------------------------------------------

let server: ServerInfo | null = null; // from the handshake
let sessions: SessionInfo[] = [];
let activeId: string | null = null;
let conv: Conversation = emptyConversation();
let source: EventSource | null = null;
let connection = ""; // stream trouble shown in the status line
let rendered: { item: Item; node: HTMLElement }[] = [];
let shownApproval: Conversation["approval"] = null;
const drafts = new Map<string, string>();

function active(): SessionInfo | undefined {
  return sessions.find((session) => session.id === activeId);
}

// --- rendering -------------------------------------------------------------------

function showAlert(text: string | null): void {
  ui.alert.textContent = text ?? "";
  ui.alert.hidden = text === null;
}

function renderSidebar(): void {
  const items = sortSessions(sessions).map((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-current", String(session.id === activeId));
    button.addEventListener("click", () => select(session.id));

    const name = document.createElement("span");
    name.className = "session-name";
    name.textContent = session.title;
    button.append(name);

    const label = stateLabel(session.state);

    if (label) {
      const state = document.createElement("span");
      state.className = `session-state state-${session.state}`;
      state.textContent = label;
      button.append(state);
    }

    const li = document.createElement("li");
    li.append(button);
    return li;
  });

  ui.sessions.replaceChildren(...items);
}

function renderHeader(): void {
  const session = active();
  ui.title.textContent = session?.title ?? "";
  document.title = session ? `${session.title} · Coding assistant` : "Coding assistant";

  const none = session === undefined || !launch.token;
  ui.rename.disabled = none;
  ui.clear.disabled = none || conv.busy;
  ui.remove.disabled = none || conv.busy;
  ui.send.disabled = none || conv.busy;
  ui.stop.disabled = none || !conv.busy;

  if (connection) {
    ui.status.textContent = connection;
  } else if (conv.waiting) {
    ui.status.textContent = "Waiting for another session";
  } else if (conv.approval) {
    ui.status.textContent = "Needs approval";
  } else if (conv.busy) {
    ui.status.textContent = "Working…";
  } else {
    ui.status.textContent = "";
  }
}

function renderMessage(item: Item): HTMLElement {
  if (item.kind === "note") {
    const note = document.createElement("p");
    note.className = item.error ? "note error" : "note";
    note.textContent = item.text;
    return note;
  }

  const message = document.createElement("div");
  message.className = `msg ${item.kind}`;

  for (const block of blocks(item.text)) {
    if (block.kind === "code") {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = block.text;
      pre.append(code);
      message.append(pre);
    } else {
      const paragraph = document.createElement("p");
      paragraph.textContent = block.text;
      message.append(paragraph);
    }
  }

  return message;
}

function nearBottom(): boolean {
  return ui.scroll.scrollHeight - ui.scroll.scrollTop - ui.scroll.clientHeight < NEAR_BOTTOM_PX;
}

/** Bring the log in line with `conv`, redrawing only items that changed (a streamed token
 * replaces just the last message). */
function renderConversation(): void {
  const stick = nearBottom();

  conv.items.forEach((item, index) => {
    const existing = rendered[index];

    if (existing?.item === item) {
      return;
    }

    const node = renderMessage(item);

    if (existing) {
      existing.node.replaceWith(node);
    } else {
      ui.log.append(node);
    }

    rendered[index] = { item, node };
  });

  for (const stale of rendered.splice(conv.items.length)) {
    stale.node.remove();
  }

  ui.empty.hidden = conv.items.length > 0;

  if (conv.approval !== shownApproval) {
    shownApproval = conv.approval;

    if (conv.approval) {
      ui.approvalCalls.textContent = formatCalls(conv.approval);
      ui.approval.hidden = false;
      ui.deny.focus(); // Enter on the focused button refuses, never allows
    } else if (!ui.approval.hidden) {
      ui.approval.hidden = true;
      ui.input.focus();
    }
  }

  renderHeader();

  if (stick) {
    ui.scroll.scrollTop = ui.scroll.scrollHeight;
  }
}

// --- the active session's event stream -----------------------------------------------

function closeStream(): void {
  source?.close();
  source = null;
  connection = "";
}

function openStream(id: string): void {
  const stream = new EventSource(api.eventsUrl(id, conv.lastEventId));
  source = stream;

  stream.onmessage = (message: MessageEvent<string>) => {
    if (stream !== source) return; // a stream from a session we left

    let data: unknown;

    try {
      data = JSON.parse(message.data);
    } catch {
      return;
    }

    if (!isChatEvent(data)) return;

    conv = apply(conv, data);
    renderConversation();
    syncActiveState();

    if (conv.gone) {
      closeStream();
      void sessionGone();
    }
  };

  stream.onopen = () => {
    if (stream !== source) return;
    connection = "";
    renderHeader();
  };

  stream.onerror = () => {
    if (stream !== source) return;
    // The browser retries by itself and resumes from the last event it saw (Last-Event-ID).
    connection = stream.readyState === EventSource.CLOSED ? "Disconnected" : "Reconnecting…";
    renderHeader();
  };
}

/** Show the active session's state in the sidebar right away instead of at the next poll. */
function syncActiveState(): void {
  const session = active();

  if (!session) return;

  const state = conv.approval ? "approval" : conv.waiting ? "waiting" : conv.busy ? "running" : "idle";

  if (session.state !== state) {
    sessions = sessions.map((s) => (s === session ? { ...s, state } : s));
    renderSidebar();
  }
}

// --- sessions ---------------------------------------------------------------------

function select(id: string): void {
  setDrawer(false);

  if (id === activeId && source !== null) {
    return;
  }

  if (activeId !== null) {
    drafts.set(activeId, ui.input.value);
  }

  closeStream();
  activeId = id;
  conv = emptyConversation();
  rendered = [];
  shownApproval = null;
  ui.log.replaceChildren();
  ui.approval.hidden = true;
  ui.input.value = drafts.get(id) ?? "";
  autosize();
  showAlert(null);
  history.replaceState(null, "", `${location.pathname}${launch.search}#${id}`);
  renderSidebar();
  renderConversation();
  openStream(id);
  ui.input.focus();
}

let refreshing = false;

/** Reload the session list. False if the server refused or could not be reached. */
async function refreshSessions(): Promise<boolean> {
  if (refreshing) return true;

  refreshing = true;

  try {
    const result = await api.listSessions();

    if (!result.ok) {
      if (result.status === 401) {
        needLink();
      } else {
        connection = result.status === 0 ? "Server unreachable" : connection;
        renderHeader();
      }

      return false;
    }

    sessions = result.data;

    if (connection === "Server unreachable") {
      connection = "";
    }

    renderSidebar();
    renderHeader();
    return true;
  } finally {
    refreshing = false;
  }
}

async function createSession(): Promise<void> {
  const result = await api.createSession();

  if (!result.ok) {
    showAlert(result.error);
    return;
  }

  sessions = [...sessions, result.data];
  select(result.data.id);
}

/** Open the session in the address, else the newest one, else a new one. */
async function openDefault(): Promise<void> {
  const wanted = sessionFromHash(location.hash);
  const choice = sessions.find((s) => s.id === wanted) ?? sortSessions(sessions)[0];

  if (choice) {
    select(choice.id);
  } else {
    await createSession();
  }
}

async function sessionGone(): Promise<void> {
  activeId = null;

  if (await refreshSessions()) {
    await openDefault();
  }
}

// --- actions ---------------------------------------------------------------------

async function send(): Promise<void> {
  const text = ui.input.value.trim();

  if (!text || !activeId || conv.busy) return;

  const id = activeId;
  const result = await api.send(id, text);

  if (result.ok) {
    if (activeId === id) {
      ui.input.value = "";
      autosize();
    }

    drafts.delete(id);
    showAlert(null);
  } else {
    showAlert(result.error);
  }
}

async function run(action: (id: string) => Promise<{ ok: boolean; error?: string }>): Promise<void> {
  if (!activeId) return;

  const result = await action(activeId);
  showAlert(result.ok ? null : (result.error ?? "Request failed."));
  await refreshSessions();
}

function autosize(): void {
  ui.input.style.height = "auto";
  // scrollHeight leaves out the border, and the box is border-box sized.
  const border = ui.input.offsetHeight - ui.input.clientHeight;
  ui.input.style.height = `${Math.min(ui.input.scrollHeight + border, 160)}px`;
}

function setDrawer(open: boolean): void {
  ui.sidebar.classList.toggle("open", open);
  ui.toggleSidebar.setAttribute("aria-expanded", String(open));
}

function needLink(): void {
  stopPage("This page needs the link printed in the terminal, including its token.");
}

/** The page cannot work (no token, wrong protocol, no server): say why and disable it. */
function stopPage(message: string): void {
  closeStream();
  ui.empty.textContent = message;
  ui.empty.hidden = false;
  ui.input.disabled = true;
  ui.newSession.disabled = true;
  launch.token = "";
  renderHeader();
}

ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  void send();
});

ui.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    void send();
  }
});

ui.input.addEventListener("input", autosize);
ui.stop.addEventListener("click", () => void run((id) => api.stop(id)));
ui.allow.addEventListener("click", () => void run((id) => api.approve(id, true)));
ui.deny.addEventListener("click", () => void run((id) => api.approve(id, false)));
ui.clear.addEventListener("click", () => void run((id) => api.reset(id)));
ui.newSession.addEventListener("click", () => void createSession());

ui.rename.addEventListener("click", () => {
  const title = window.prompt("Session name", active()?.title ?? "");

  if (title !== null && title.trim()) {
    void run((id) => api.rename(id, title));
  }
});

ui.remove.addEventListener("click", () => {
  const session = active();

  if (session && window.confirm(`Delete "${session.title}"? Its conversation is lost.`)) {
    void (async () => {
      const result = await api.remove(session.id);

      if (!result.ok) {
        showAlert(result.error);
        return;
      }

      drafts.delete(session.id);
      closeStream();
      await sessionGone();
    })();
  }
});

ui.toggleSidebar.addEventListener("click", () => setDrawer(!ui.sidebar.classList.contains("open")));

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setDrawer(false);
});

window.addEventListener("hashchange", () => {
  const id = sessionFromHash(location.hash);

  if (id && id !== activeId && sessions.some((s) => s.id === id)) {
    select(id);
  }
});

// --- start ------------------------------------------------------------------------

async function start(): Promise<void> {
  if (!launch.token) {
    needLink();
    return;
  }

  renderHeader();

  // Handshake first: agree on the protocol and learn which features this backend offers.
  const hello = await api.handshake();

  if (!hello.ok) {
    if (hello.status === 401) {
      needLink();
    } else {
      stopPage(hello.error);
    }

    return;
  }

  server = hello.data;

  if (await refreshSessions()) {
    await openDefault();
  }

  window.setInterval(() => {
    void (async () => {
      if (!launch.token) return;
      await refreshSessions();

      // Deleted from another tab while we were looking at it.
      if (activeId && !sessions.some((s) => s.id === activeId)) {
        closeStream();
        await sessionGone();
      }
    })();
  }, POLL_MS);
}

void start();
