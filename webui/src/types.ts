// Shapes of what the backend (backend/server.py, backend/session.py) sends.

export type SessionState = "idle" | "waiting" | "running" | "approval";

export interface SessionInfo {
  id: string;
  title: string;
  state: SessionState;
  created: number; // seconds since the epoch
  last_event: number;
}

export interface ToolCall {
  name: string;
  args: unknown;
  id?: string;
}

// One entry of a session's event log (GET /api/sessions/<id>/events). `gone` has no id:
// the session was deleted and the stream ends.
export type ChatEvent =
  | { id: number; type: "user"; text: string }
  | { id: number; type: "waiting" }
  | { id: number; type: "token"; text: string }
  | { id: number; type: "approval"; calls: ToolCall[] }
  | { id: number; type: "done" }
  | { id: number; type: "cancelled" }
  | { id: number; type: "error"; text: string }
  | { id: number; type: "reset" }
  | { type: "gone" };
