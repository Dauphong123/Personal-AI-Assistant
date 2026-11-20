"""HTTP + Server-Sent Events API over SessionManager, standard library only.

The frontend (frontend/) is a separate app on its own origin that reaches the agent only through
this API; its origins are listed in frontend_origins (CORS). With static_dir set the server also
serves the built frontend on its own origin (--serve-frontend). The agent can run commands and
write files, so the server is locked down: it binds to loopback, every API call needs the launch
token, the Host header must be ours (DNS rebinding) and a browser Origin must be ours or a listed
frontend origin (cross-site requests). Pages are served with a CSP that forbids inline script
and style.

API (JSON bodies, token in X-Token or Authorization: Bearer):
    POST /api/handshake       {protocol}    {protocol, server, features, limits}; 409 if the
                                            frontend speaks another protocol version (rebuild it)
    GET  /api/sessions                      {"sessions": [info, ...]}
    POST /api/sessions        {title?}      201 {"session": info}; 429 at the session limit
    GET  /api/sessions/<id>                 {"session": info}
    GET  /api/sessions/<id>/events          SSE; ?after=N or Last-Event-ID; ?token= accepted
    POST /api/sessions/<id>/send    {text}
    POST /api/sessions/<id>/approve {allow, request}  request = id from the approval event;
                                            409 if that request is not the one waiting
    POST /api/sessions/<id>/stop
    POST /api/sessions/<id>/reset
    POST /api/sessions/<id>/rename  {title}
    POST /api/sessions/<id>/delete
    POST /api/sessions/<id>/model   {model, effort?}  switch this session's model (409 while busy)
    GET  /api/models                        {models: [...], efforts: [...], default: {model, effort}}
    POST /api/voice/transcribe  audio/*     {"text"}; 503 unless started with --voice;
                                            ?dictate=1 adds {"prompt", "hands_free", "answer"} (backend.voice.dictation)
    POST /api/voice/speak       {text}      audio/wav (204: nothing speakable)
    GET  /api/health                        {"status": "ok"}; no token (Host still checked)
    OPTIONS /api/...                        CORS preflight: 204 for a listed frontend origin, else 403

A handler that raises answers 500 {"error": "Internal server error"} and is logged.
"""

import hmac
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from model_choice import EFFORTS, available_models, parse_choice

from .session import ChatSession
from .sessions import SessionManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
BUILD_HINT = "The web frontend is not built. Run: npm --prefix frontend install && npm --prefix frontend run build"
# Bump when the API changes incompatibly; frontend/src/types.ts PROTOCOL must match.
PROTOCOL = 3  # 2: approve needs the approval event's request id; 3: `tool` events, tool output no longer in `token`
SERVER_NAME = "coding-assistant"
MAX_BODY = 64 * 1024
MAX_AUDIO = 4 * 1024 * 1024  # one push-to-talk recording (compressed browser audio)
# How much of a refused body is read so the client sees our answer (closing with unread data
# makes Windows reset the connection). Covers a recording just over MAX_AUDIO; anything larger
# is answered and closed without reading it all.
DISCARD_LIMIT = 2 * MAX_AUDIO
LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}
KEEPALIVE_SECONDS = 10
# A socket read or write that stalls this long ends the connection (idle keep-alive, slow
# clients). Longer than KEEPALIVE_SECONDS so an open event stream is never cut.
REQUEST_TIMEOUT = 30

# Only these file types are served from the build directory.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; media-src 'self' blob:; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
}

# CORS answers for a listed frontend origin (the page sends the token in X-Token).
CORS_METHODS = "GET, POST"
CORS_HEADERS = "Content-Type, X-Token, Authorization"
CORS_MAX_AGE = "600"
API_ONLY_HINT = (
    "Not found: this server is the agent API only. Start the frontend (npm --prefix frontend start) "
    "or run the backend with --serve-frontend."
)

HANDSHAKE_PATH = "/api/handshake"
HEALTH_PATH = "/api/health"  # no token: says only that the server answers
VOICE_PATHS = {"/api/voice/transcribe", "/api/voice/speak"}
API_ROUTE = re.compile(r"^/api/sessions(?:/(?P<id>[0-9a-f]{12})(?:/(?P<action>[a-z]+))?)?$")
POST_ACTIONS = {"send", "approve", "stop", "reset", "rename", "delete", "model"}
MODELS_PATH = "/api/models"


def new_token() -> str:
    return secrets.token_urlsafe(24)


def parse_origin(value: str) -> str:
    """A browser origin as the Origin header sends it ("http://127.0.0.1:5173"): scheme, host
    and optional port, lowercased, no path, query, fragment or user. ValueError otherwise."""
    problem = f"Not an origin: {value!r} (expected e.g. http://127.0.0.1:5173)"

    try:
        parts = urlsplit(value.strip())
        port = parts.port  # ValueError for a port that is not a number
    except ValueError:
        raise ValueError(problem) from None

    if (
        parts.scheme.lower() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.path not in {"", "/"}
        or "?" in value
        or "#" in value
    ):
        raise ValueError(problem)

    origin = f"{parts.scheme.lower()}://{parts.hostname.lower()}"
    return origin if port is None else f"{origin}:{port}"


class WebServer(ThreadingHTTPServer):
    daemon_threads = True  # an open event stream must not keep the process alive
    allow_reuse_address = False  # on Windows SO_REUSEADDR lets a second process share the port

    def __init__(
        self,
        address,
        manager: SessionManager,
        token: str,
        static_dir: Path | None,
        allow_remote: bool,
        notices: bool = False,
        voice=None,
        frontend_origins: Iterable[str] = (),
    ):
        super().__init__(address, Handler)
        self.manager = manager
        self.token = token
        self.static_dir = static_dir  # None: API only, the frontend runs on its own origin
        self.frontend_origins = frozenset(parse_origin(origin) for origin in frontend_origins)
        self.allow_remote = allow_remote
        self.notices = notices  # background-job notices are routed to sessions
        self.voice = voice  # backend.voice.WebVoice or None

    def allowed_hosts(self) -> set[str]:
        port = self.server_address[1]
        return {f"{name}:{port}" for name in LOOPBACK_HOSTS}

    def describe(self) -> dict:
        """What the handshake tells the frontend: protocol, features and limits."""
        transcribe = bool(self.voice is not None and self.voice.can_transcribe)
        speak = bool(self.voice is not None and self.voice.can_speak)
        return {
            "protocol": PROTOCOL,
            "server": SERVER_NAME,
            "features": {
                "sessions": True,
                "notices": self.notices,
                "voice": {"transcribe": transcribe, "speak": speak},
                "models": True,
            },
            "limits": {
                "max_sessions": self.manager.max_sessions,
                "max_turns": self.manager.max_turns,
                "max_body": MAX_BODY,
                "max_audio": MAX_AUDIO if transcribe else 0,
            },
        }


def make_server(
    manager: SessionManager,
    token: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    static_dir: Path | None = None,
    allow_remote: bool = False,
    notices: bool = False,
    voice=None,
    frontend_origins: Iterable[str] = (),
) -> WebServer:
    """static_dir=None: API only (the frontend runs on its own origin, listed in
    frontend_origins); a build directory also serves the page on this origin."""
    if host not in LOOPBACK_HOSTS and not allow_remote:
        raise ValueError(f"Refusing to listen on {host!r}: only loopback is allowed without allow_remote.")

    return WebServer(
        (host, port),
        manager,
        token,
        None if static_dir is None else Path(static_dir),
        allow_remote,
        notices,
        voice,
        frontend_origins,
    )


def static_file(static_dir: Path, url_path: str) -> Path | None:
    """The build file a URL path names, or None. Hidden names, '..', backslashes and
    unlisted file types are refused; the result must stay inside static_dir."""
    path = unquote(url_path)

    if "\\" in path or "\x00" in path:
        return None

    parts = (path.lstrip("/") or "index.html").split("/")

    if any(not part or part.startswith(".") for part in parts):
        return None

    if Path(parts[-1]).suffix.lower() not in CONTENT_TYPES:
        return None

    root = static_dir.resolve()
    file = root.joinpath(*parts).resolve()

    if not file.is_relative_to(root) or not file.is_file():
        return None

    return file


class Handler(BaseHTTPRequestHandler):
    server: WebServer  # pyright: ignore

    timeout = REQUEST_TIMEOUT
    _response_started = False

    def log_message(self, format, *args):  # noqa: A002  (request lines carry the token)
        pass

    def send_response(self, code, message=None):
        self._response_started = True
        super().send_response(code, message)

    def _guarded(self, handle) -> None:
        """Run one request; a bug becomes a logged 500 instead of a dropped connection."""
        self._response_started = False

        try:
            handle()
        except (ConnectionError, TimeoutError):
            self.close_connection = True  # the client went away; nothing to answer
        except Exception:
            # The path only: the query string can carry the token.
            logger.exception("Request failed: %s %s", self.command, urlsplit(self.path).path)
            self.close_connection = True

            if not self._response_started:
                try:
                    self._json(500, {"error": "Internal server error"})
                except OSError:
                    pass

    def do_GET(self) -> None:
        self._guarded(self._get)

    def do_POST(self) -> None:
        self._guarded(self._post)

    def do_OPTIONS(self) -> None:
        self._guarded(self._options)

    # --- responses ---------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        self.wfile.write(body)

    def _common_headers(self) -> None:
        """Security headers on every answer; CORS headers when a listed frontend origin asks."""
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)

        if self.server.frontend_origins:
            self.send_header("Vary", "Origin")

        origin = self._frontend_origin()

        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _fail(self, status: int, message: str) -> None:
        self.close_connection = True
        self._drain_body()
        self._json(status, {"error": message})

    def _drain_body(self) -> None:
        """Read a request body nobody read yet, so the error is not lost to a reset (closing a
        socket with unread data makes Windows reset the connection)."""
        if self.command != "POST" or getattr(self, "_body_read", False):
            return

        self._body_read = True

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return

        self._discard(length)

    # --- checks ------------------------------------------------------------------

    def _host_ok(self) -> bool:
        if self.server.allow_remote:
            return True

        return self.headers.get("Host", "") in self.server.allowed_hosts()

    def _frontend_origin(self) -> str | None:
        """The request's Origin when it is a listed frontend origin, else None."""
        origin = self.headers.get("Origin")
        return origin if origin is not None and origin in self.server.frontend_origins else None

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")

        if origin is None or self.server.allow_remote or self._frontend_origin() is not None:
            return True

        return origin in {f"http://{host}" for host in self.server.allowed_hosts()}

    def _token_ok(self, query: dict, allow_query: bool) -> bool:
        supplied = self.headers.get("X-Token", "")
        auth = self.headers.get("Authorization", "")

        if not supplied and auth.startswith("Bearer "):
            supplied = auth[len("Bearer "):]

        if not supplied and allow_query:
            supplied = query.get("token", [""])[0]  # EventSource cannot set headers

        return hmac.compare_digest(supplied.encode(), self.server.token.encode())

    def _discard(self, length: int) -> None:
        """Read and drop a refused body (up to DISCARD_LIMIT).

        Closing a socket that still has unread data makes the OS reset the connection, and the
        client then never sees our 413. Beyond the limit we just answer and close."""
        remaining = min(max(length, 0), DISCARD_LIMIT)
        self.connection.settimeout(2)

        try:
            while remaining > 0:
                data = self.rfile.read(min(remaining, 65536))

                if not data:
                    break

                remaining -= len(data)
        except OSError:
            pass

    def _read_json(self) -> dict | None:
        """The JSON object in the request body, or None after sending an error."""
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()

        if content_type != "application/json":
            self._fail(415, "Content-Type must be application/json")
            return None

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._fail(411, "Content-Length required")
            return None

        self._body_read = True

        if length < 0 or length > MAX_BODY:
            self._discard(length)
            self._fail(413, "Request body too large")
            return None

        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._fail(400, "Invalid JSON")
            return None

        if not isinstance(data, dict):
            self._fail(400, "JSON object expected")
            return None

        return data

    def _authorized(self, query: dict, allow_query_token: bool = False) -> bool:
        """Token and Origin checks for an API call; sends the error when they fail."""
        if not self._token_ok(query, allow_query_token):
            self._fail(401, "Missing or wrong token")
            return False

        if not self._origin_ok():
            self._fail(403, "Forbidden origin")
            return False

        return True

    def _session(self, session_id: str) -> ChatSession | None:
        session = self.server.manager.get(session_id)

        if session is None:
            self._fail(404, "No such session")

        return session

    # --- routing -----------------------------------------------------------------

    def _get(self) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)

        if not self._host_ok():
            return self._fail(403, "Forbidden host")

        if url.path == HEALTH_PATH:
            return self._json(200, {"status": "ok"})

        if not url.path.startswith("/api/"):
            return self._static(url.path)

        if url.path == HANDSHAKE_PATH or url.path in VOICE_PATHS:
            return self._fail(405, "Use POST")

        if url.path == MODELS_PATH:
            if not self._authorized(query, allow_query_token=False):
                return
            default = self.server.manager.default_choice()
            return self._json(
                200,
                {
                    "models": available_models(),
                    "efforts": list(EFFORTS),
                    "default": {
                        "model": default.model if default else os.environ.get("MODEL", "").strip(),
                        "effort": default.effort if default else None,
                    },
                },
            )

        route = API_ROUTE.match(url.path)

        if route is None:
            return self._fail(404, "Not found")

        sid, action = route.group("id"), route.group("action")

        if action is not None and action != "events" and action not in POST_ACTIONS:
            return self._fail(404, "Not found")

        if not self._authorized(query, allow_query_token=action == "events"):
            return

        if sid is None:
            return self._json(200, {"sessions": self.server.manager.list()})

        session = self._session(sid)

        if session is None:
            return

        if action is None:
            return self._json(200, {"session": session.info()})

        if action == "events":
            return self._events(session, query)

        self._fail(405, "Use POST")

    def _options(self) -> None:
        """CORS preflight: the frontend's origin may call the API with the token header. The
        preflight itself carries no token; the real request is still checked in full."""
        if not self._host_ok():
            return self._fail(403, "Forbidden host")

        if not urlsplit(self.path).path.startswith("/api/"):
            return self._fail(404, "Not found")

        if self._frontend_origin() is None:
            return self._fail(403, "Forbidden origin")

        self.send_response(204)
        self._common_headers()
        self.send_header("Access-Control-Allow-Methods", CORS_METHODS)
        self.send_header("Access-Control-Allow-Headers", CORS_HEADERS)
        self.send_header("Access-Control-Max-Age", CORS_MAX_AGE)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _post(self) -> None:
        url = urlsplit(self.path)
        self._body_read = False  # the handler serves every request of a keep-alive connection

        if not self._host_ok():
            return self._fail(403, "Forbidden host")

        if url.path == HEALTH_PATH:
            return self._fail(405, "Use GET")

        if not url.path.startswith("/api/"):
            return self._fail(405, "Use GET")

        if url.path == HANDSHAKE_PATH:
            if not self._authorized(parse_qs(url.query)):
                return

            data = self._read_json()
            return None if data is None else self._handshake(data)

        if url.path in VOICE_PATHS:
            if not self._authorized(parse_qs(url.query)):
                return

            if url.path.endswith("/transcribe"):
                return self._transcribe(dictate=parse_qs(url.query).get("dictate") == ["1"])

            return self._speak()

        route = API_ROUTE.match(url.path)

        if route is None:
            return self._fail(404, "Not found")

        sid, action = route.group("id"), route.group("action")

        if action is not None and action != "events" and action not in POST_ACTIONS:
            return self._fail(404, "Not found")

        if not self._authorized(parse_qs(url.query)):
            return

        if sid is not None and action in (None, "events"):
            return self._fail(405, "Use GET")

        data = self._read_json()

        if data is None:
            return

        if sid is None:
            return self._create(data)

        session = self._session(sid)

        if session is not None:
            self._action(session, action, data)  # pyright: ignore[reportArgumentType]

    # --- handlers ----------------------------------------------------------------

    def _static(self, path: str) -> None:
        static_dir = self.server.static_dir

        if static_dir is None:
            return self._fail(404, API_ONLY_HINT)

        if not static_dir.is_dir():
            return self._send(503, BUILD_HINT.encode("utf-8"), "text/plain; charset=utf-8")

        file = static_file(static_dir, path)

        if file is None:
            return self._fail(404, "Not found")

        try:
            body = file.read_bytes()
        except OSError:
            return self._fail(404, "Not found")

        self._send(200, body, CONTENT_TYPES[file.suffix.lower()])

    def _handshake(self, data: dict) -> None:
        protocol = data.get("protocol")

        if not isinstance(protocol, int) or isinstance(protocol, bool):
            return self._fail(400, "protocol must be an integer")

        if protocol != PROTOCOL:
            self.close_connection = True
            return self._json(
                409,
                {
                    "error": f"The page speaks protocol {protocol}, this backend speaks {PROTOCOL}. "
                    "Rebuild the frontend (npm --prefix frontend run build) and reload the page.",
                    "protocol": PROTOCOL,
                },
            )

        self._json(200, self.server.describe())

    def _transcribe(self, dictate: bool = False) -> None:
        """POST /api/voice/transcribe: raw browser audio in, {"text"} out (plus the cleaned
        prompt and hands-free command when dictating)."""
        voice = self.server.voice

        if voice is None or not voice.can_transcribe:
            return self._fail(503, "Speech recognition is not enabled (start the backend with --voice)")

        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()

        if not content_type.startswith("audio/"):
            return self._fail(415, "Content-Type must be audio/*")

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._fail(411, "Content-Length required")

        self._body_read = True

        if length > MAX_AUDIO or length < 0:
            self._discard(length)
            return self._fail(413, "Recording too long")

        data = self.rfile.read(length)

        if not data:
            return self._fail(400, "Empty recording")

        try:
            text = voice.transcribe(data)
        except Exception as exc:
            logger.warning("Could not transcribe a recording: %s", exc)
            return self._fail(422, "Could not understand the recording")

        if not dictate:
            return self._json(200, {"text": text})

        from .voice import dictation

        self._json(200, {"text": text, **dictation(text)})

    def _speak(self) -> None:
        """POST /api/voice/speak {text}: a WAV reading it aloud (204 when nothing is speakable)."""
        voice = self.server.voice

        if voice is None or not voice.can_speak:
            return self._fail(503, "Speech output is not enabled (start the backend with --voice)")

        data = self._read_json()

        if data is None:
            return

        text = data.get("text")

        if not isinstance(text, str) or not text.strip():
            return self._fail(400, "text must be a non-empty string")

        try:
            audio = voice.speak(text)
        except Exception:
            logger.exception("Speech synthesis failed")
            return self._fail(500, "Speech synthesis failed")

        if not audio:
            self.send_response(204)
            self._common_headers()
            self.end_headers()
            return

        self._send(200, audio, "audio/wav")

    def _create(self, data: dict) -> None:
        title = data.get("title")

        if title is not None and not isinstance(title, str):
            return self._fail(400, "title must be a string")

        session = self.server.manager.create(title=title)

        if session is None:
            return self._fail(429, "Too many sessions; delete one first")

        self._json(201, {"session": session.info()})

    def _action(self, session: ChatSession, action: str, data: dict) -> None:
        if action == "send":
            text = data.get("text")

            if not isinstance(text, str) or not text.strip():
                return self._fail(400, "text must be a non-empty string")

            if not session.send(text):
                return self._fail(409, "A turn is already running")
        elif action == "approve":
            allow = data.get("allow")

            request = data.get("request")

            if not isinstance(allow, bool):
                return self._fail(400, "allow must be true or false")

            if not isinstance(request, str) or not request:
                return self._fail(400, "request must be the id from the approval event")

            if not session.approve(allow, request):
                return self._fail(409, "That request is no longer waiting for approval")
        elif action == "model":
            model, effort = data.get("model"), data.get("effort")

            if not isinstance(model, str) or not (effort is None or isinstance(effort, str)):
                return self._fail(400, "model must be a model name and effort text or null")

            try:
                choice = parse_choice(f"{model} {effort}" if effort else model)
            except ValueError as exc:
                return self._fail(400, str(exc))

            if not session.set_model(choice):
                return self._fail(409, "Stop the running turn first")
        elif action == "stop":
            session.stop()
        elif action == "reset":
            if not session.new_chat():
                return self._fail(409, "Stop the running turn first")
        elif action == "rename":
            title = data.get("title")

            if not isinstance(title, str) or not session.rename(title):
                return self._fail(400, "title must be a non-empty string")
        elif action == "delete":
            result = self.server.manager.delete(session.id)

            if result == "busy":
                return self._fail(409, "Stop the running turn first")

            if result == "missing":
                return self._fail(404, "No such session")

            return self._json(200, {"ok": True})

        self._json(200, {"ok": True, "session": session.info()})

    def _events(self, session: ChatSession, query: dict) -> None:
        after = 0

        for raw in (query.get("after", [""])[0], self.headers.get("Last-Event-ID", "")):
            try:
                after = max(after, int(raw))
            except ValueError:
                pass

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self._common_headers()
        self.end_headers()
        last_write = time.monotonic()

        try:
            while True:
                if self.server.manager.get(session.id) is not session:
                    # Deleted: say so (no id, so a reconnect does not skip anything) and end.
                    self.wfile.write(b'data: {"type": "gone"}\n\n')
                    self.wfile.flush()
                    return

                events = session.events_after(after, timeout=1.0)

                for event in events:
                    chunk = f"id: {event['id']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    self.wfile.write(chunk.encode("utf-8"))
                    after = event["id"]

                if events:
                    self.wfile.flush()
                    last_write = time.monotonic()
                elif time.monotonic() - last_write >= KEEPALIVE_SECONDS:
                    # A comment line; also how a vanished client is noticed.
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_write = time.monotonic()
        except OSError:
            return
