// Typed client for the backend API (backend/server.py). Calls never throw: failures come back
// as { ok: false, status, error } so the page can show them.

import { PROTOCOL } from "./types.ts";
import type { ServerInfo, SessionInfo } from "./types.ts";

export type Result<T> = { ok: true; data: T } | { ok: false; status: number; error: string };

export class Api {
  readonly #token: string;
  readonly #fetch: typeof fetch;

  constructor(token: string, fetchFn: typeof fetch = (input, init) => fetch(input, init)) {
    this.#token = token;
    this.#fetch = fetchFn;
  }

  /** First call of a page: agree on the protocol and learn what this backend offers. */
  handshake(): Promise<Result<ServerInfo>> {
    return this.#call("POST", "/api/handshake", { protocol: PROTOCOL, client: "webui" }, (body) => body as unknown as ServerInfo);
  }

  listSessions(): Promise<Result<SessionInfo[]>> {
    return this.#call("GET", "/api/sessions", undefined, (body) => body["sessions"] as SessionInfo[]);
  }

  createSession(title?: string): Promise<Result<SessionInfo>> {
    return this.#call("POST", "/api/sessions", title === undefined ? {} : { title }, sessionOf);
  }

  send(id: string, text: string): Promise<Result<SessionInfo>> {
    return this.#action(id, "send", { text });
  }

  approve(id: string, allow: boolean): Promise<Result<SessionInfo>> {
    return this.#action(id, "approve", { allow });
  }

  stop(id: string): Promise<Result<SessionInfo>> {
    return this.#action(id, "stop", {});
  }

  reset(id: string): Promise<Result<SessionInfo>> {
    return this.#action(id, "reset", {});
  }

  rename(id: string, title: string): Promise<Result<SessionInfo>> {
    return this.#action(id, "rename", { title });
  }

  remove(id: string): Promise<Result<null>> {
    return this.#call("POST", `${sessionPath(id)}/delete`, {}, () => null);
  }

  /** EventSource cannot send headers, so the token goes in the query string. */
  eventsUrl(id: string, after: number): string {
    const query = new URLSearchParams({ token: this.#token, after: String(after) });
    return `${sessionPath(id)}/events?${query.toString()}`;
  }

  /** Speech to text: one push-to-talk recording (needs the handshake's voice.transcribe). */
  async transcribe(recording: Blob): Promise<Result<string>> {
    const sent = await this.#request("POST", "/api/voice/transcribe", recording, recording.type || "audio/webm");

    if (!sent.ok) return sent;

    const record = await jsonRecord(sent.data);
    return { ok: true, data: typeof record["text"] === "string" ? record["text"] : "" };
  }

  /** Text to speech: a WAV to play, or null when nothing in the text is worth saying. */
  async speak(text: string): Promise<Result<Blob | null>> {
    const sent = await this.#request("POST", "/api/voice/speak", JSON.stringify({ text }), "application/json");

    if (!sent.ok) return sent;

    return { ok: true, data: sent.data.status === 204 ? null : await sent.data.blob() };
  }

  #action(id: string, action: string, body: object): Promise<Result<SessionInfo>> {
    return this.#call("POST", `${sessionPath(id)}/${action}`, body, sessionOf);
  }

  async #call<T>(
    method: "GET" | "POST",
    path: string,
    body: object | undefined,
    pick: (body: Record<string, unknown>) => T,
  ): Promise<Result<T>> {
    const sent = await this.#request(
      method,
      path,
      body === undefined ? undefined : JSON.stringify(body),
      body === undefined ? undefined : "application/json",
    );

    if (!sent.ok) return sent;

    return { ok: true, data: pick(await jsonRecord(sent.data)) };
  }

  /** Send with the token; a non-2xx answer or a network failure becomes an error result. */
  async #request(
    method: "GET" | "POST",
    path: string,
    body: BodyInit | undefined,
    contentType: string | undefined,
  ): Promise<Result<Response>> {
    const headers: Record<string, string> = { "X-Token": this.#token };
    const init: RequestInit = { method, headers };

    if (body !== undefined) {
      init.body = body;
    }

    if (contentType !== undefined) {
      headers["Content-Type"] = contentType;
    }

    let response: Response;

    try {
      response = await this.#fetch(path, init);
    } catch {
      return { ok: false, status: 0, error: "Cannot reach the server." };
    }

    if (!response.ok) {
      const record = await jsonRecord(response);
      const message = typeof record["error"] === "string" ? record["error"] : `Request failed (${response.status}).`;
      return { ok: false, status: response.status, error: message };
    }

    return { ok: true, data: response };
  }
}

/** The JSON object in a response, or {} when there is none. */
async function jsonRecord(response: Response): Promise<Record<string, unknown>> {
  try {
    const parsed: unknown = await response.json();
    return typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

function sessionPath(id: string): string {
  return `/api/sessions/${encodeURIComponent(id)}`;
}

function sessionOf(body: Record<string, unknown>): SessionInfo {
  return body["session"] as SessionInfo;
}
