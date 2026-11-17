import assert from "node:assert/strict";
import { test } from "node:test";

import { Api } from "../src/api.ts";
import { PROTOCOL } from "../src/types.ts";

interface Call {
  url: string;
  init: RequestInit | undefined;
}

function fakeFetch(status: number, body: unknown, calls: Call[]): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return new Response(body === undefined ? "" : JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
}

const info = { id: "abcdef012345", title: "New chat", state: "idle", created: 1, last_event: 0 };

test("listSessions sends the token and returns the sessions", async () => {
  const calls: Call[] = [];
  const api = new Api("tok", fakeFetch(200, { sessions: [info] }, calls));
  const result = await api.listSessions();
  assert.deepEqual(result, { ok: true, data: [info] });
  assert.equal(calls[0]?.url, "/api/sessions");
  assert.equal(new Headers(calls[0]?.init?.headers).get("X-Token"), "tok");
});

test("createSession posts JSON and returns the new session", async () => {
  const calls: Call[] = [];
  const api = new Api("tok", fakeFetch(201, { session: info }, calls));
  const result = await api.createSession("Plans");
  assert.deepEqual(result, { ok: true, data: info });
  assert.equal(calls[0]?.url, "/api/sessions");
  assert.equal(calls[0]?.init?.method, "POST");
  assert.equal(new Headers(calls[0]?.init?.headers).get("Content-Type"), "application/json");
  assert.deepEqual(JSON.parse(String(calls[0]?.init?.body)), { title: "Plans" });
});

test("createSession without a title sends an empty object", async () => {
  const calls: Call[] = [];
  await new Api("tok", fakeFetch(201, { session: info }, calls)).createSession();
  assert.deepEqual(JSON.parse(String(calls[0]?.init?.body)), {});
});

test("session actions post to their route", async () => {
  const calls: Call[] = [];
  const api = new Api("tok", fakeFetch(200, { ok: true, session: info }, calls));
  const id = "abcdef012345";
  await api.send(id, "hello");
  await api.approve(id, false);
  await api.stop(id);
  await api.reset(id);
  await api.rename(id, "Name");
  await api.remove(id);
  assert.deepEqual(
    calls.map((call) => call.url),
    ["send", "approve", "stop", "reset", "rename", "delete"].map((action) => `/api/sessions/${id}/${action}`),
  );
  assert.deepEqual(JSON.parse(String(calls[0]?.init?.body)), { text: "hello" });
  assert.deepEqual(JSON.parse(String(calls[1]?.init?.body)), { allow: false });
  assert.deepEqual(JSON.parse(String(calls[4]?.init?.body)), { title: "Name" });
});

test("session ids are escaped in URLs", async () => {
  const calls: Call[] = [];
  await new Api("tok", fakeFetch(200, {}, calls)).stop("../x?y");
  assert.equal(calls[0]?.url, "/api/sessions/..%2Fx%3Fy/stop");
});

test("an error response carries the server message and status", async () => {
  const api = new Api("tok", fakeFetch(409, { error: "A turn is already running" }, []));
  assert.deepEqual(await api.send("abcdef012345", "x"), {
    ok: false,
    status: 409,
    error: "A turn is already running",
  });
});

test("an error response without JSON gets a generic message", async () => {
  const fetchFn = (async () => new Response("boom", { status: 500 })) as typeof fetch;
  const result = await new Api("tok", fetchFn).listSessions();
  assert.equal(result.ok, false);
  assert.equal(!result.ok && result.status, 500);
  assert.match(!result.ok ? result.error : "", /500/);
});

test("a network failure is an error result, not a throw", async () => {
  const fetchFn = (async () => {
    throw new TypeError("Failed to fetch");
  }) as typeof fetch;
  const result = await new Api("tok", fetchFn).listSessions();
  assert.deepEqual(result, { ok: false, status: 0, error: "Cannot reach the server." });
});

test("handshake posts our protocol and returns the server description", async () => {
  const calls: Call[] = [];
  const hello = {
    protocol: PROTOCOL,
    server: "coding-assistant",
    features: { sessions: true, notices: true, voice: { transcribe: false, speak: true } },
    limits: { max_sessions: 50, max_turns: 1, max_body: 65536, max_audio: 0 },
  };
  const result = await new Api("tok", fakeFetch(200, hello, calls)).handshake();
  assert.deepEqual(result, { ok: true, data: hello });
  assert.equal(calls[0]?.url, "/api/handshake");
  assert.deepEqual(JSON.parse(String(calls[0]?.init?.body)), { protocol: PROTOCOL, client: "webui" });
});

test("handshake mismatch passes the server's message on", async () => {
  const api = new Api("tok", fakeFetch(409, { error: "rebuild with npm", protocol: 9 }, []));
  assert.deepEqual(await api.handshake(), { ok: false, status: 409, error: "rebuild with npm" });
});

test("transcribe uploads the recording with its audio type", async () => {
  const calls: Call[] = [];
  const api = new Api("tok", fakeFetch(200, { text: "hello" }, calls));
  const blob = new Blob([new Uint8Array([1, 2, 3])], { type: "audio/webm;codecs=opus" });
  assert.deepEqual(await api.transcribe(blob), { ok: true, data: "hello" });
  assert.equal(calls[0]?.url, "/api/voice/transcribe");
  assert.equal(calls[0]?.init?.method, "POST");
  const headers = new Headers(calls[0]?.init?.headers);
  assert.equal(headers.get("Content-Type"), "audio/webm;codecs=opus");
  assert.equal(headers.get("X-Token"), "tok");
  assert.equal(calls[0]?.init?.body, blob);
});

test("transcribe of a typeless blob says webm", async () => {
  const calls: Call[] = [];
  await new Api("tok", fakeFetch(200, { text: "" }, calls)).transcribe(new Blob([new Uint8Array([1])]));
  assert.equal(new Headers(calls[0]?.init?.headers).get("Content-Type"), "audio/webm");
});

test("speak returns the audio, or null when there is nothing to say", async () => {
  const wav = (async () =>
    new Response(new Uint8Array([82, 73, 70, 70]), { status: 200, headers: { "Content-Type": "audio/wav" } })) as typeof fetch;
  const result = await new Api("tok", wav).speak("Done.");
  assert.equal(result.ok, true);
  assert.equal(result.ok && result.data?.type, "audio/wav");
  assert.equal(result.ok && result.data?.size, 4);

  const empty = (async () => new Response(null, { status: 204 })) as typeof fetch;
  assert.deepEqual(await new Api("tok", empty).speak("***"), { ok: true, data: null });
});

test("speak posts JSON text and reports errors", async () => {
  const calls: Call[] = [];
  const result = await new Api("tok", fakeFetch(503, { error: "voice off" }, calls)).speak("hi");
  assert.deepEqual(result, { ok: false, status: 503, error: "voice off" });
  assert.equal(calls[0]?.url, "/api/voice/speak");
  assert.deepEqual(JSON.parse(String(calls[0]?.init?.body)), { text: "hi" });
});

test("eventsUrl carries token and resume point", () => {
  const api = new Api("t k", fakeFetch(200, {}, []));
  assert.equal(api.eventsUrl("abcdef012345", 7), "/api/sessions/abcdef012345/events?token=t+k&after=7");
});
