// The page: a sessions sidebar and the active session's conversation.
//
// Everything the model or a web page wrote is untrusted: it is only ever inserted as text
// (textContent / createTextNode), never as HTML.

import { Api } from "./api.ts";
import {
  apply,
  blocks,
  describeCalls,
  emptyConversation,
  formatCalls,
  isChatEvent,
  isThinking,
  relativeTime,
  sortSessions,
  stateLabel,
  statusOf,
} from "./state.ts";
import type { Conversation, Item } from "./state.ts";
import type { ServerInfo, SessionInfo } from "./types.ts";
import { apiFrom, sessionFromHash, tokenFrom } from "./url.ts";
import { parseMarkdown } from "./markdown.ts";
import { copyButton, renderBlocks, renderCalls, renderTool } from "./render.ts";
import { shortcut } from "./keys.ts";
import { nextTheme, parseTheme, themeLabel } from "./theme.ts";
import type { Theme } from "./theme.ts";
import {
  PressTracker,
  TAP_MAX_MS,
  handsFreeAction,
  handsFreeLabel,
  insertDictation,
  meterBars,
  recordingLabel,
} from "./handsfree.ts";
import { HandsFreeListener, MAX_RECORDING_MS, Player, Recorder, canListen, canRecord, replyToSpeak } from "./voice.ts";

const TOKEN_KEY = "coding-assistant-token";
const API_KEY = "coding-assistant-api";
const SPEAK_KEY = "coding-assistant-speak";
const THEME_KEY = "coding-assistant-theme";
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
  titleInput: byId<HTMLInputElement>("title-input"),
  menuButton: byId<HTMLButtonElement>("session-menu"),
  menu: byId("session-menu-list"),
  confirm: byId<HTMLDialogElement>("confirm"),
  confirmTitle: byId("confirm-title"),
  confirmText: byId("confirm-text"),
  confirmOk: byId<HTMLButtonElement>("confirm-ok"),
  confirmCancel: byId<HTMLButtonElement>("confirm-cancel"),
  jump: byId<HTMLButtonElement>("jump"),
  theme: byId<HTMLButtonElement>("theme"),
  status: byId("status"),
  rename: byId<HTMLButtonElement>("rename"),
  clear: byId<HTMLButtonElement>("clear-chat"),
  remove: byId<HTMLButtonElement>("delete-session"),
  scroll: byId("scroll"),
  empty: byId("empty"),
  emptyText: byId("empty-text"),
  examples: byId("examples"),
  log: byId("log"),
  thinking: byId("thinking"),
  alert: byId("alert"),
  approval: byId("approval"),
  approvalCalls: byId("approval-calls"),
  approvalTitle: byId("approval-title"),
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
  voiceMeter: byId("voice-meter"),
  voiceState: byId("voice-state"),
  voiceTime: byId("voice-time"),
};

// --- token ---------------------------------------------------------------------

function stored(key: string): string | null {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key: string, value: string): void {
  try {
    sessionStorage.setItem(key, value);
  } catch {
    // Storage blocked: the value then lasts until the page is reloaded.
  }
}

const launch = tokenFrom(location.search, stored(TOKEN_KEY));

if (launch.token) {
  store(TOKEN_KEY, launch.token);
}

// The agent API this page talks to: `?api=` from the backend's launch link ("" = this origin).
const target = apiFrom(launch.search, stored(API_KEY));
launch.search = target.search;

if (target.api) {
  store(API_KEY, target.api);
}

// Keep the token and API address out of the address bar and the history (the hash names the session).
history.replaceState(null, "", location.pathname + launch.search + location.hash);

const api = new Api(launch.token, undefined, target.api);

// --- state ---------------------------------------------------------------------

let server: ServerInfo | null = null; // from the handshake
let sessions: SessionInfo[] = [];
let activeId: string | null = null;
let conv: Conversation = emptyConversation();
let source: EventSource | null = null;
let connection = ""; // stream trouble shown in the status line
let rendered: { item: Item; node: HTMLElement }[] = [];
let writingNode: HTMLElement | null = null; // the answer still being written (its spine moves)
const messageText = new WeakMap<HTMLElement, string>(); // what a message's Copy button copies
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
// The meter strip in the composer while the microphone is open.
let talkStartedAt: number | null = null; // push-to-talk recording began
let hearing = false; // hands-free: speech is coming in now
let answerPlaying = false; // hands-free listening is paused while an answer plays
let meterTimer = 0;
const bars = [...ui.voiceMeter.querySelectorAll<HTMLElement>(".meter span")];

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

    const meta = document.createElement("span");
    meta.className = "session-meta";
    const label = stateLabel(session.state);

    if (label) {
      const state = document.createElement("span");
      state.className = `session-state state-${session.state}`;
      state.textContent = label;
      meta.append(state);
    } else {
      const time = document.createElement("span");
      time.className = "session-time";
      time.textContent = relativeTime(session.created);
      meta.append(time);
    }

    button.append(meta);

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
  ui.menuButton.disabled = none;
  ui.rename.disabled = none;
  ui.clear.disabled = none || conv.busy;
  ui.remove.disabled = none || conv.busy;
  ui.stop.disabled = none || !conv.busy;
  // Dictation goes into the message box, so it works while a turn runs.
  ui.mic.disabled = none || handsFree || voiceStatus === "Transcribing…";
  ui.handsFree.disabled = none;

  ui.send.disabled = none || conv.busy || ui.input.value.trim() === "";

  const status = statusOf({
    conv,
    connection,
    voiceStatus,
    handsFree,
    transcribing: transcribing > 0,
    queued: handsFreeQueue.length,
  });
  ui.status.textContent = status.text;
  ui.status.dataset["tone"] = status.tone;
  renderMeter();
}

function setBars(level: number): void {
  meterBars(level).forEach((height, index) => {
    const bar = bars[index];

    if (bar) bar.style.transform = `scaleY(${height})`;
  });
}

/** The strip above the message box: live level, recording time or hands-free state. */
function renderMeter(): void {
  let state = "";
  let time = "";

  if (talkStartedAt !== null) {
    state = "Recording";
    time = recordingLabel(performance.now() - talkStartedAt, MAX_RECORDING_MS);
  } else if (handsFree) {
    state = handsFreeLabel({ speaking: hearing, paused: answerPlaying, transcribing: transcribing > 0, queued: handsFreeQueue.length });
  } else if (voiceStatus === "Transcribing…") {
    state = "Transcribing";
  }

  const shown = state !== "";
  ui.voiceMeter.hidden = !shown;
  ui.voiceMeter.dataset["mode"] = talkStartedAt !== null ? "talk" : handsFree ? "hands-free" : "busy";
  ui.voiceState.textContent = state;
  ui.voiceTime.textContent = time;

  if (shown && meterTimer === 0) {
    meterTimer = window.setInterval(renderMeter, 250); // the clock moves without other events
  } else if (!shown && meterTimer !== 0) {
    window.clearInterval(meterTimer);
    meterTimer = 0;
    setBars(0);
  }
}

/** Set the visible label of an icon button (its icon stays). */
function setLabel(button: HTMLElement, text: string): void {
  const label = button.querySelector(".label");

  if (label) label.textContent = text;
}

function renderMessage(item: Item): HTMLElement {
  if (item.kind === "notice") {
    const notice = document.createElement("p");
    notice.className = "job-notice";
    notice.textContent = item.text;
    return notice;
  }

  if (item.kind === "tool") {
    return renderTool(item);
  }

  if (item.kind === "note") {
    const note = document.createElement("p");
    note.className = item.error ? "note error" : "note";
    note.textContent = item.text;
    return note;
  }

  const message = document.createElement("div");
  message.className = `msg ${item.kind}`;

  if (item.kind === "assistant") {
    // The model writes markdown; it is parsed into a tree and built with textContent only.
    message.classList.add("md");
    message.append(...renderBlocks(parseMarkdown(item.text)));

    const actions = document.createElement("div");
    actions.className = "msg-actions";
    actions.append(copyButton("copy-msg", "Copy answer"));
    message.append(actions);
    messageText.set(message, item.text);
    return message;
  }

  // The user's own text is shown as typed (only ``` fences become code).
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

    if (existing?.node === writingNode) writingNode = null; // replaced: set again below

    rendered[index] = { item, node };
  });

  for (const stale of rendered.splice(conv.items.length)) {
    stale.node.remove();
  }

  ui.empty.hidden = conv.items.length > 0;
  ui.thinking.hidden = !isThinking(conv);

  const last = rendered.at(-1);
  const writing = conv.busy && last?.item.kind === "assistant" ? last.node : null;

  if (writing !== writingNode) {
    writingNode?.classList.remove("writing");
    writing?.classList.add("writing");
    writingNode = writing;
  }

  if (conv.approval !== shownApproval) {
    shownApproval = conv.approval;

    if (conv.approval) {
      const calls = describeCalls(conv.approval);
      ui.approvalCalls.replaceChildren(...renderCalls(calls));
      ui.approval.classList.toggle("danger", calls.some((call) => call.danger));
      ui.approvalTitle.textContent = calls.length > 1 ? `Allow these ${calls.length} actions?` : "Allow this action?";
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

  updateJump();
}

/** "Jump to latest" shows while the newest messages are out of view. */
function updateJump(): void {
  ui.jump.hidden = nearBottom() || conv.items.length === 0;
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
  closeMenu();
  endRename(false);
  stopHandsFree();
  handsFreeQueue.length = 0; // spoken for the session we are leaving
  recorder.cancel();
  ctrlDown = false;
  buttonPressedAt = null;
  talkStartedAt = null;
  player.stop();
  voiceStatus = "";
  ui.mic.setAttribute("aria-pressed", "false");
  setLabel(ui.mic, "Talk");
  activeId = id;
  replayEnd = active()?.last_event ?? 0;
  conv = emptyConversation();
  rendered = [];
  writingNode = null;
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

/** Allow or deny the approval shown on the card (named by its request id). */
function answerApproval(allow: boolean): void {
  const request = conv.approvalRequest;
  if (request !== null) void run((id) => api.approve(id, allow, request));
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
  setLabel(ui.mic, "Done");
  voiceStatus = "Listening…";
  talkStartedAt = performance.now();
  renderHeader();
}

function cancelTalking(): void {
  talkAttempt += 1;
  talkStartedAt = null;
  recorder.cancel();
  ui.mic.setAttribute("aria-pressed", "false");
  setLabel(ui.mic, "Talk");
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
  talkStartedAt = null;
  const recording = await recorder.stop();
  ui.mic.setAttribute("aria-pressed", "false");
  setLabel(ui.mic, "Talk");
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
  hearing = false;
  answerPlaying = false;
  showAlert(null);
  ui.handsFree.setAttribute("aria-pressed", "true");
  setLabel(ui.handsFree, "Stop hands-free");
  renderHeader();
}

function stopHandsFree(message: string | null = null): void {
  if (!handsFree) return;

  // Prompts already spoken stay queued and are still sent (as the CLI's ChatLoop does).
  listener.stop();
  handsFree = false;
  hearing = false;
  transcribing = 0;
  ui.handsFree.setAttribute("aria-pressed", "false");
  setLabel(ui.handsFree, "Hands-free");

  if (message) showAlert(message);

  renderHeader();
}

async function handleUtterance(sessionId: string | null, recording: Blob): Promise<void> {
  if (!handsFree || sessionId !== activeId || !sessionId || tooLong(recording)) return;

  const heardFor = conv.approvalRequest; // the approval waiting while the user spoke
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
  // An answer counts only for the request it was heard for, never for a newer one.
  const request = conv.approvalRequest !== null && conv.approvalRequest === heardFor ? conv.approvalRequest : null;

  switch (action.kind) {
    case "off":
      stopHandsFree();
      return;
    case "stop":
      handsFreeQueue.length = 0;
      await run((id) => api.stop(id));
      break;
    case "approve":
      if (request !== null) await run((id) => api.approve(id, action.allow, request));
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
  player.onChange = (playing) => {
    listener.pause(playing);
    answerPlaying = playing;
    setBars(0);
    renderMeter();
  };
  recorder.onLevel = setBars;
  listener.onLevel = (level, speaking) => {
    setBars(level);

    if (speaking !== hearing) {
      hearing = speaking;
      renderMeter();
    }
  };
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
  ui.input.style.height = `${Math.min(ui.input.scrollHeight + border, 192)}px`;
  renderHeader(); // Send is enabled only with something to send
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
  ui.emptyText.textContent = message;
  ui.examples.hidden = true;
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

// Example prompts on an empty session: put the text in the box, ready to edit or send.
ui.examples.addEventListener("click", (event) => {
  const example = event.target instanceof Element ? event.target.closest<HTMLElement>(".example") : null;
  const prompt = example?.dataset["prompt"];

  if (!prompt || ui.input.disabled) return;

  ui.input.value = prompt;
  autosize();
  ui.input.focus();
});
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
ui.allow.addEventListener("click", () => answerApproval(true));
ui.deny.addEventListener("click", () => answerApproval(false));
ui.newSession.addEventListener("click", () => void createSession());

// --- session menu, rename, confirm ---------------------------------------------------------

function menuItems(): HTMLButtonElement[] {
  return [...ui.menu.querySelectorAll<HTMLButtonElement>("[role=menuitem]")].filter((item) => !item.disabled);
}

function openMenu(): void {
  ui.menu.hidden = false;
  ui.menuButton.setAttribute("aria-expanded", "true");
  menuItems()[0]?.focus();
}

function closeMenu(focusButton = false): void {
  if (ui.menu.hidden) return;

  ui.menu.hidden = true;
  ui.menuButton.setAttribute("aria-expanded", "false");

  if (focusButton) ui.menuButton.focus();
}

ui.menuButton.addEventListener("click", () => (ui.menu.hidden ? openMenu() : closeMenu()));

ui.menu.addEventListener("keydown", (event) => {
  const items = menuItems();
  const at = items.indexOf(document.activeElement as HTMLButtonElement);
  const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;

  if (step) {
    event.preventDefault();
    items[(at + step + items.length) % items.length]?.focus();
  } else if (event.key === "Tab") {
    closeMenu();
  }
});

document.addEventListener("pointerdown", (event) => {
  if (!ui.menu.hidden && event.target instanceof Node && !ui.menu.parentElement?.contains(event.target)) {
    closeMenu();
  }
});

let renaming = false;

function startRename(): void {
  const session = active();

  if (!session || renaming) return;

  closeMenu();
  renaming = true;
  ui.titleInput.value = session.title;
  ui.title.hidden = true;
  ui.titleInput.hidden = false;
  ui.titleInput.focus();
  ui.titleInput.select();
}

/** Leave the rename box; `save` sends the new name if it changed. */
function endRename(save: boolean): void {
  if (!renaming) return;

  renaming = false;
  const title = ui.titleInput.value.trim();
  ui.titleInput.hidden = true;
  ui.title.hidden = false;

  if (save && title && title !== active()?.title) {
    ui.title.textContent = title; // shown at once; the next poll brings the server's version
    void run((id) => api.rename(id, title));
  }
}

ui.titleInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    endRename(true);
    ui.input.focus();
  } else if (event.key === "Escape") {
    event.preventDefault();
    endRename(false);
    ui.input.focus();
  }
});

ui.titleInput.addEventListener("blur", () => endRename(true));
ui.title.addEventListener("dblclick", startRename);
ui.rename.addEventListener("click", startRename);

/** Ask in a modal dialog; resolves true when the user confirms. */
function confirmAction(title: string, text: string, ok: string): Promise<boolean> {
  ui.confirmTitle.textContent = title;
  ui.confirmText.textContent = text;
  ui.confirmOk.textContent = ok;

  return new Promise((resolve) => {
    const finish = (answer: boolean): void => {
      ui.confirmOk.removeEventListener("click", yes);
      ui.confirmCancel.removeEventListener("click", no);
      ui.confirm.removeEventListener("close", no);

      if (ui.confirm.open) ui.confirm.close();

      resolve(answer);
    };
    const yes = (): void => finish(true);
    const no = (): void => finish(false);

    ui.confirmOk.addEventListener("click", yes);
    ui.confirmCancel.addEventListener("click", no);
    ui.confirm.addEventListener("close", no); // Escape closes the dialog
    ui.confirm.showModal();
    ui.confirmCancel.focus(); // Enter on the focused button keeps things as they are
  });
}

ui.clear.addEventListener("click", () => {
  closeMenu();

  void (async () => {
    const ok = await confirmAction(
      "Clear this conversation?",
      "The messages are removed and the assistant starts fresh in this session. Long-term memory is kept.",
      "Clear",
    );

    if (ok) await run((id) => api.reset(id));
  })();
});

ui.remove.addEventListener("click", () => {
  closeMenu();
  const session = active();

  if (!session) return;

  void (async () => {
    const ok = await confirmAction(`Delete "${session.title}"?`, "The session and its conversation are removed.", "Delete");

    if (!ok) return;

    const result = await api.remove(session.id);

    if (!result.ok) {
      showAlert(result.error);
      return;
    }

    drafts.delete(session.id);
    closeStream();
    await sessionGone();
  })();
});

// --- copy and jump --------------------------------------------------------------------------

async function copyText(button: HTMLElement, text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    showAlert("Could not copy: the browser blocked the clipboard.");
    return;
  }

  const label = button.querySelector(".label");

  if (!label) return;

  label.textContent = "Copied";
  button.classList.add("copied");
  window.setTimeout(() => {
    label.textContent = "Copy";
    button.classList.remove("copied");
  }, 1500);
}

ui.log.addEventListener("click", (event) => {
  const button = event.target instanceof Element ? event.target.closest<HTMLElement>(".copy") : null;

  if (!button) return;

  if (button.classList.contains("copy-code")) {
    const code = button.closest("figure")?.querySelector("code");
    void copyText(button, code?.textContent ?? "");
  } else {
    const message = button.closest<HTMLElement>(".msg");
    void copyText(button, (message && messageText.get(message)) ?? "");
  }
});

ui.scroll.addEventListener("scroll", updateJump, { passive: true });

ui.jump.addEventListener("click", () => {
  const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  ui.scroll.scrollTo({ top: ui.scroll.scrollHeight, behavior: smooth ? "smooth" : "auto" });
});

ui.toggleSidebar.addEventListener("click", () => setDrawer(!ui.sidebar.classList.contains("open")));

document.addEventListener("keydown", (event) => {
  const target = event.target;
  const typing =
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLInputElement ||
    (target instanceof HTMLElement && target.isContentEditable);

  const action = shortcut({
    key: event.key,
    ctrl: event.ctrlKey,
    alt: event.altKey,
    meta: event.metaKey,
    typing,
    menuOpen: !ui.menu.hidden,
    drawerOpen: ui.sidebar.classList.contains("open"),
    dialogOpen: ui.confirm.open,
    renaming,
    busy: activeId !== null && conv.busy,
  });

  switch (action) {
    case "close-menu":
      closeMenu(true);
      break;
    case "close-drawer":
      setDrawer(false);
      break;
    case "stop":
      void run((id) => api.stop(id));
      break;
    case "focus-input":
      ui.input.focus();
      break;
    case null:
      return;
  }

  event.preventDefault();
});

window.addEventListener("hashchange", () => {
  const id = sessionFromHash(location.hash);

  if (id && id !== activeId && sessions.some((s) => s.id === id)) {
    select(id);
  }
});

// --- theme --------------------------------------------------------------------------

function applyTheme(theme: Theme): void {
  const root = document.documentElement;

  if (theme === "system") {
    delete root.dataset["theme"];
    root.style.colorScheme = "";
  } else {
    root.dataset["theme"] = theme;
    root.style.colorScheme = theme; // native controls and scrollbars follow too
  }

  setLabel(ui.theme, themeLabel(theme));
}

function storedTheme(): Theme {
  try {
    return parseTheme(localStorage.getItem(THEME_KEY));
  } catch {
    return "system";
  }
}

let theme = storedTheme();
applyTheme(theme);

ui.theme.addEventListener("click", () => {
  theme = nextTheme(theme);
  applyTheme(theme);

  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // storage blocked: the choice lasts until reload
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
