// Shapes of what the backend (backend/server.py, backend/session.py) sends.

/** API version this page speaks; must equal backend/server.py PROTOCOL (the handshake checks). */
export const PROTOCOL = 2; // 2: approve needs the approval event's request id

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

export interface ToolCall {
  name: string;
  args: unknown;
  id?: string;
}

// One entry of a session's event log (GET /api/sessions/<id>/events). `gone` has no id:
// the session was deleted and the stream ends.
export type ChatEvent =
  | { id: number; type: "user"; text: string }
  | { id: number; type: "notice"; text: string } // a finished background job starts a turn
  | { id: number; type: "waiting" }
  | { id: number; type: "token"; text: string }
  | { id: number; type: "approval"; calls: ToolCall[]; request: string } // request: id an answer must name
  | { id: number; type: "done" }
  | { id: number; type: "cancelled" }
  | { id: number; type: "error"; text: string }
  | { id: number; type: "reset" }
  | { type: "gone" };
