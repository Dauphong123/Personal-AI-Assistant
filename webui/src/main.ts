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
import { PressTracker, TAP_MAX_MS, handsFreeAction, insertDictation } from "./handsfree.ts";
import { HandsFreeListener, Player, Recorder, canListen, canRecord, replyToSpeak } from "./voice.ts";

const TOKEN_KEY = "coding-assistant-token";
const SPEAK_KEY = "coding-assistant-speak";
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
  mic: byId<HTMLButtonElement>("mic"),
  handsFree: byId<HTMLButtonElement>("hands-free"),
  speakLabel: byId("speak-label"),
  speakAnswers: byId<HTMLInputElement>("speak-answers"),
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
const recorder = new Recorder();
const player = new Player();
let voiceStatus = ""; // "Listening…" / "Transcribing…"
// Push-to-talk: hold the Talk button or Right Ctrl (a short click on Talk records until the next).
const rightCtrl = new PressTracker();
let ctrlDown = false; // Right Ctrl is held and its press started a recording
let buttonPressedAt: number | null = null; // the Talk button is held down
let talkAttempt = 0; // bumped by cancelTalking, so a microphone that opens late is closed again
// Hands-free: every pause ends a prompt; prompts wait here until the session is idle.
const listener = new HandsFreeListener();
let handsFree = false;
const handsFreeQueue: string[] = [];
let handsFreeWork: Promise<void> = Promise.resolve(); // utterances are handled in order
let transcribing = 0; // hands-free recordings at the backend
let sending = false; // a queued prompt was sent and its turn has not shown up yet
let replayEnd = 0; // events up to here were history when the session was opened: never spoken

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
  // Dictation goes into the message box, so it works while a turn runs.
  ui.mic.disabled = none || handsFree || voiceStatus === "Transcribing…";
  ui.handsFree.disabled = none;

  let status = "";

  if (connection) {
    status = connection;
  } else if (voiceStatus) {
    status = voiceStatus;
  } else if (conv.waiting) {
    status = "Waiting for another session";
  } else if (conv.approval) {
    status = handsFree ? "Needs approval: say yes or no" : "Needs approval";
  } else if (conv.busy) {
    status = "Working…";
  }

  const queued = handsFreeQueue.length ? ` · ${handsFreeQueue.length} queued` : "";

  if (handsFree && !connection) {
    const hearing = transcribing ? "Transcribing…" : "Listening…";
    status = `Hands-free: ${status || hearing}${queued}`;
  } else if (queued && !connection) {
    status = `${status || "Sending…"}${queued}`;
  }

  ui.status.textContent = status;
}

function renderMessage(item: Item): HTMLElement {
  if (item.kind === "notice") {
    const notice = document.createElement("p");
    notice.className = "job-notice";
    notice.textContent = item.text;
    return notice;
  }

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

    const reply = ui.speakAnswers.checked ? replyToSpeak(conv, data, replayEnd) : null;

    if (reply) {
      void speakReply(id, reply);
    }

    if (data.type === "user") sending = false; // a sent prompt's turn has begun

    if (handsFree && data.type === "approval" && data.id > replayEnd && ui.speakAnswers.checked && conv.approval) {
      void speakReply(id, `May I run ${formatCalls(conv.approval)}? Say yes or no.`);
    }

    void sendQueued();

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
  stopHandsFree();
  handsFreeQueue.length = 0; // spoken for the session we are leaving
  recorder.cancel();
  ctrlDown = false;
  buttonPressedAt = null;
  player.stop();
  voiceStatus = "";
  ui.mic.setAttribute("aria-pressed", "false");
  ui.mic.textContent = "Talk";
  activeId = id;
  replayEnd = active()?.last_event ?? 0;
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

// --- voice ------------------------------------------------------------------------

async function speakReply(sessionId: string, text: string): Promise<void> {
  const result = await api.speak(text);

  // The user may have moved on (another session, a new recording) while it was synthesized.
  if (!result.ok || !result.data || sessionId !== activeId || recorder.recording) return;

  player.play(result.data);
}

async function startTalking(): Promise<void> {
  player.stop();
  showAlert(null);
  const attempt = ++talkAttempt;

  try {
    await recorder.start();

    if (attempt !== talkAttempt) {
      recorder.cancel(); // let go (or cancelled) before the microphone was open
      return;
    }
  } catch (error) {
    const denied = error instanceof DOMException && error.name === "NotAllowedError";
    showAlert(denied ? "Microphone access was refused." : "Could not use the microphone.");
    return;
  }

  ui.mic.setAttribute("aria-pressed", "true");
  ui.mic.textContent = "Done";
  voiceStatus = "Listening…";
  renderHeader();
}

function cancelTalking(): void {
  talkAttempt += 1;
  recorder.cancel();
  ui.mic.setAttribute("aria-pressed", "false");
  ui.mic.textContent = "Talk";
  voiceStatus = "";
  renderHeader();
}

function tooLong(recording: Blob): boolean {
  const maxAudio = server?.limits.max_audio ?? 0;
  return maxAudio > 0 && recording.size > maxAudio;
}

/** Push-to-talk ended: the cleaned-up text goes into the message box (as Wispr Flow types it
 * where you are), so it can be checked or added to before Enter sends it. */
async function finishTalking(): Promise<void> {
  if (!recorder.recording) return;

  const sessionId = activeId;
  const recording = await recorder.stop();
  ui.mic.setAttribute("aria-pressed", "false");
  ui.mic.textContent = "Talk";
  voiceStatus = recording.size > 0 ? "Transcribing…" : "";
  renderHeader();

  if (recording.size === 0) return;

  const result = tooLong(recording)
    ? { ok: false as const, status: 413, error: "That recording is too long." }
    : await api.dictate(recording);
  voiceStatus = "";
  renderHeader();

  if (sessionId !== activeId) return; // switched sessions meanwhile: drop it

  if (!result.ok) {
    showAlert(result.error);
  } else if (result.data.hands_free === "on") {
    await startHandsFree();
  } else if (result.data.hands_free === "off") {
    stopHandsFree();
  } else if (!result.data.prompt) {
    showAlert("Didn't catch that. Try again.");
  } else {
    const box = ui.input;
    const next = insertDictation(box.value, box.selectionStart ?? box.value.length, box.selectionEnd ?? box.value.length, result.data.prompt);
    box.value = next.value;
    autosize();
    box.focus();
    box.setSelectionRange(next.cursor, next.cursor);
  }
}

// --- hands-free -------------------------------------------------------------------

async function startHandsFree(): Promise<void> {
  if (handsFree || !activeId || ui.handsFree.hidden) return;

  cancelTalking();
  player.stop();

  try {
    await listener.start();
  } catch (error) {
    const denied = error instanceof DOMException && error.name === "NotAllowedError";
    showAlert(denied ? "Microphone access was refused." : "Could not use the microphone.");
    return;
  }

  handsFree = true;
  showAlert(null);
  ui.handsFree.setAttribute("aria-pressed", "true");
  ui.handsFree.textContent = "Stop hands-free";
  renderHeader();
}

function stopHandsFree(message: string | null = null): void {
  if (!handsFree) return;

  // Prompts already spoken stay queued and are still sent (as the CLI's ChatLoop does).
  listener.stop();
  handsFree = false;
  transcribing = 0;
  ui.handsFree.setAttribute("aria-pressed", "false");
  ui.handsFree.textContent = "Hands-free";

  if (message) showAlert(message);

  renderHeader();
}

async function handleUtterance(sessionId: string | null, recording: Blob): Promise<void> {
  if (!handsFree || sessionId !== activeId || !sessionId || tooLong(recording)) return;

  transcribing += 1;
  renderHeader();
  const result = await api.dictate(recording);
  transcribing = Math.max(0, transcribing - 1);

  if (!handsFree || sessionId !== activeId) return;

  if (!result.ok) {
    showAlert(result.error);
    renderHeader();
    return;
  }

  const action = handsFreeAction(result.data, conv.approval !== null);

  switch (action.kind) {
    case "off":
      stopHandsFree();
      return;
    case "stop":
      handsFreeQueue.length = 0;
      await run((id) => api.stop(id));
      break;
    case "approve":
      await run((id) => api.approve(id, action.allow));
      break;
    case "send":
      handsFreeQueue.push(action.text);
      break;
    case "ignore":
      break;
  }

  renderHeader();
  await sendQueued();
}

/** Send the next spoken prompt once the session can take it (one turn at a time). */
async function sendQueued(): Promise<void> {
  if (sending || !activeId || conv.busy || conv.approval || handsFreeQueue.length === 0) return;

  const id = activeId;
  const text = handsFreeQueue.shift() ?? "";
  sending = true;
  renderHeader();
  const result = await api.send(id, text);

  if (!result.ok) {
    sending = false;

    if (result.status === 409) {
      handsFreeQueue.unshift(text); // a turn started meanwhile: try again when it ends
    } else {
      showAlert(result.error);
    }
  }

  renderHeader();
}

// --- push-to-talk keys --------------------------------------------------------------

function voiceUsable(): boolean {
  return !ui.mic.hidden && !ui.mic.disabled;
}

function onKeyDown(event: KeyboardEvent): void {
  if (event.code === "ControlRight") {
    if (event.repeat) return;

    rightCtrl.down(performance.now());

    if (!handsFree && voiceUsable() && !recorder.recording) {
      ctrlDown = true;
      void startTalking();
    }

    return;
  }

  if (ctrlDown && event.ctrlKey) {
    // Right Ctrl + another key is a shortcut (copy, paste...), not talking.
    ctrlDown = false;
    cancelTalking();
  }
}

function onKeyUp(event: KeyboardEvent): void {
  if (event.code !== "ControlRight") return;

  const release = rightCtrl.up(performance.now());
  const wasTalking = ctrlDown;
  ctrlDown = false;

  if (release === "hold") {
    if (wasTalking) void finishTalking();
    return;
  }

  if (wasTalking) cancelTalking(); // a tap is a switch, not a recording

  if (release === "double_tap") {
    if (handsFree) stopHandsFree();
    else void startHandsFree();
  } else if (handsFree) {
    stopHandsFree();
  }
}

function setUpVoice(info: ServerInfo): void {
  ui.mic.hidden = !(info.features.voice.transcribe && canRecord());
  ui.handsFree.hidden = !(info.features.voice.transcribe && canListen());
  ui.speakLabel.hidden = !info.features.voice.speak;
  player.onChange = (playing) => listener.pause(playing);
  listener.onUtterance = (recording) => {
    const sessionId = activeId;
    handsFreeWork = handsFreeWork.then(() => handleUtterance(sessionId, recording));
  };
  listener.onIdle = () => stopHandsFree("Hands-free stopped after two minutes without speech.");

  try {
    ui.speakAnswers.checked = localStorage.getItem(SPEAK_KEY) === "on";
  } catch {
    ui.speakAnswers.checked = false;
  }

  recorder.onAutoStop = () => void finishTalking();
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
  stopHandsFree();
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
// Talk: hold to talk (released = into the message box); a short click records until the next.
ui.mic.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || ui.mic.disabled) return;

  if (recorder.recording) {
    void finishTalking();
    return;
  }

  buttonPressedAt = performance.now();
  void startTalking();
});

const releaseTalk = (): void => {
  if (buttonPressedAt === null) return;

  const held = performance.now() - buttonPressedAt > TAP_MAX_MS;
  buttonPressedAt = null;

  if (held) void finishTalking();
};

ui.mic.addEventListener("pointerup", releaseTalk);
ui.mic.addEventListener("pointercancel", releaseTalk);
ui.mic.addEventListener("click", (event) => {
  // Pointer presses are handled above; this is Enter/Space on the focused button.
  if (event.detail === 0) void (recorder.recording ? finishTalking() : startTalking());
});
ui.handsFree.addEventListener("click", () => void (handsFree ? stopHandsFree() : startHandsFree()));
document.addEventListener("keydown", onKeyDown);
document.addEventListener("keyup", onKeyUp);
window.addEventListener("blur", () => {
  // The key-up would be lost while another window has focus.
  if (ctrlDown) {
    ctrlDown = false;
    cancelTalking();
  }
});

ui.speakAnswers.addEventListener("change", () => {
  if (!ui.speakAnswers.checked) player.stop();

  try {
    localStorage.setItem(SPEAK_KEY, ui.speakAnswers.checked ? "on" : "off");
  } catch {
    // storage blocked: the choice lasts until reload
  }
});
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
  setUpVoice(server);

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
