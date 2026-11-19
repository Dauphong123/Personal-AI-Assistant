// Shapes of what the backend (backend/server.py, backend/session.py) sends.

/** API version this page speaks; must equal backend/server.py PROTOCOL (the handshake checks). */
export const PROTOCOL = 3; // 2: approve needs the approval event's request id; 3: `tool` events

/** The backend's answer to the handshake (POST /api/handshake). */
export interface ServerInfo {
  protocol: number;
  server: string;
  features: {
    sessions: boolean;
    notices: boolean; // background-job completion notices arrive as `notice` turns
    voice: { transcribe: boolean; speak: boolean };
  };
  limits: { max_sessions: number; max_turns: number; max_body: number; max_audio: number };
}

export type SessionState = "idle" | "waiting" | "running" | "approval";

export interface SessionInfo {
  id: string;
  title: string;
  state: SessionState;
  created: number; // seconds since the epoch
  last_event: number;
}

export type ToolStatus = "running" | "done" | "error";

export interface ToolCall {
  name: string;
  args: unknown;
  id?: string;
}

// One entry of a session's event log (GET /api/sessions/<id>/events). `gone` has no id:
// the session was deleted and the stream ends. `at`: when it happened, in seconds since the
// epoch (the backend stamps every logged event; optional so older logs and tests still fit).
export type ChatEvent = { at?: number } & (
  | { id: number; type: "user"; text: string }
  | { id: number; type: "notice"; text: string } // a finished background job starts a turn
  | { id: number; type: "waiting" }
  | { id: number; type: "token"; text: string }
  // A tool call started (`running`, no text yet) or finished; `call` pairs the two. Output and
  // long arguments are cut by the backend.
  | { id: number; type: "tool"; call: string; name: string; args: Record<string, unknown>; status: ToolStatus; text?: string }
  | { id: number; type: "approval"; calls: ToolCall[]; request: string } // request: id an answer must name
  | { id: number; type: "done" }
  | { id: number; type: "cancelled" }
  | { id: number; type: "error"; text: string }
  | { id: number; type: "reset" }
  | { type: "gone" }
);
