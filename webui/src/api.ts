// Typed client for the backend API (backend/server.py). Calls never throw: failures come back
// as { ok: false, status, error } so the page can show them.

import type { SessionInfo } from "./types.ts";

export type Result<T> = { ok: true; data: T } | { ok: false; status: number; error: string };

export class Api {
  readonly #token: string;
  readonly #fetch: typeof fetch;

  constructor(token: string, fetchFn: typeof fetch = (input, init) => fetch(input, init)) {
    this.#token = token;
    this.#fetch = fetchFn;
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

  #action(id: string, action: string, body: object): Promise<Result<SessionInfo>> {
    return this.#call("POST", `${sessionPath(id)}/${action}`, body, sessionOf);
  }

  async #call<T>(
    method: "GET" | "POST",
    path: string,
    body: object | undefined,
    pick: (body: Record<string, unknown>) => T,
  ): Promise<Result<T>> {
    const headers: Record<string, string> = { "X-Token": this.#token };
    const init: RequestInit = { method, headers };

    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }

    let response: Response;

    try {
      response = await this.#fetch(path, init);
    } catch {
      return { ok: false, status: 0, error: "Cannot reach the server." };
    }

    let parsed: unknown = null;

    try {
      parsed = await response.json();
    } catch {
      parsed = null;
    }

    const record = typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : {};

    if (!response.ok) {
      const message = typeof record["error"] === "string" ? record["error"] : `Request failed (${response.status}).`;
      return { ok: false, status: response.status, error: message };
    }

    return { ok: true, data: pick(record) };
  }
}

function sessionPath(id: string): string {
  return `/api/sessions/${encodeURIComponent(id)}`;
}

function sessionOf(body: Record<string, unknown>): SessionInfo {
  return body["session"] as SessionInfo;
}
