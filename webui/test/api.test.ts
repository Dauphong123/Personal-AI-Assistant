import assert from "node:assert/strict";
import { test } from "node:test";

import { Api } from "../src/api.ts";

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

test("eventsUrl carries token and resume point", () => {
  const api = new Api("t k", fakeFetch(200, {}, []));
  assert.equal(api.eventsUrl("abcdef012345", 7), "/api/sessions/abcdef012345/events?token=t+k&after=7");
});
