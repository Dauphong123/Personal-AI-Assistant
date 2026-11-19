"""Start the agent's web API: python -m backend [--port N] [--max-turns N] [--serve-frontend]

By default this is the API only: the frontend (frontend/, `npm --prefix frontend start`) runs
on its own origin and reaches the agent through the API. --serve-frontend also serves the built
frontend (frontend/dist) on the API's origin.
"""

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

from log_setup import configure_logging

from .server import BUILD_HINT, LOOPBACK_HOSTS, STATIC_DIR, make_server, new_token, parse_origin
from .sessions import SessionManager

# Where `npm --prefix frontend start` serves the page (frontend/scripts/serve.mjs).
DEFAULT_FRONTEND_ORIGINS = ("http://127.0.0.1:5173", "http://localhost:5173")


def _origin_arg(value: str) -> str:
    try:
        return parse_origin(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m backend", description="Web chat backend for the coding assistant")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="listen on a non-loopback address; the assistant can run commands, so only on a network you trust",
    )
    parser.add_argument(
        "--voice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="talk to the assistant from the browser and hear its answers (on by default when "
        "requirements_voice.txt is installed; --no-voice turns it off)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1,
        help="how many sessions may run a turn at the same time; others wait (default: 1)",
    )
    parser.add_argument(
        "--frontend-origin",
        action="append",
        type=_origin_arg,
        metavar="ORIGIN",
        help="origin of the frontend allowed to call the API, e.g. http://127.0.0.1:5173; repeat for "
        f"several (default: {', '.join(DEFAULT_FRONTEND_ORIGINS)})",
    )
    parser.add_argument(
        "--serve-frontend",
        action="store_true",
        help="also serve the built frontend (frontend/dist) on the API's own address",
    )
    return parser.parse_args(argv)


def frontend_origins(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.frontend_origin or DEFAULT_FRONTEND_ORIGINS)


def _shown_host(host: str) -> str:
    return "localhost" if host in {"0.0.0.0", "::"} else host


def build_url(host: str, port: int, token: str) -> str:
    return f"http://{_shown_host(host)}:{port}/?token={token}"


def frontend_url(origin: str, host: str, port: int, token: str) -> str:
    """The separate frontend's link: it names the API to call and carries the token."""
    return f"{origin}/?api=http://{_shown_host(host)}:{port}&token={token}"


def frontend_built(static_dir: Path = STATIC_DIR) -> bool:
    return (static_dir / "index.html").is_file()


def build_app():
    """The compiled graph, as main.py builds it. Importing it loads the models (slow)."""
    from langgraph.checkpoint.memory import InMemorySaver

    from graph import ModelGraph

    return ModelGraph.compile(checkpointer=InMemorySaver())


def build_voice():
    from .voice import WebVoice

    return WebVoice.from_env()


def connect_jobs(manager: SessionManager) -> None:
    """Finished background jobs notify the session that started them."""
    from tools import JOBS

    JOBS.on_finish = manager.job_finished


def cleanup() -> None:
    from graph.node import wait_for_memory
    from tools import JOBS

    JOBS.close()  # background commands must not outlive the app
    wait_for_memory()  # a save that is still running should finish


def install_stop_signals(signals: Any = signal) -> None:
    """SIGTERM and Ctrl+Break (Windows SIGBREAK) stop the server like Ctrl+C, so the
    shutdown below still runs: turns stopped, background jobs killed, memory saved."""

    def stop(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGBREAK"):
        number = getattr(signals, name, None)

        if number is not None:
            signals.signal(number, stop)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.host not in LOOPBACK_HOSTS and not args.allow_remote:
        sys.exit(
            f"Refusing to listen on {args.host}: the assistant can run commands and edit files. "
            "Pass --allow-remote if you really want that."
        )

    if args.max_turns < 1:
        sys.exit("--max-turns must be at least 1.")

    if args.serve_frontend and not frontend_built():
        sys.exit(BUILD_HINT)

    try:
        configure_logging()
    except ValueError as exc:
        sys.exit(str(exc))
    print("Loading models...", flush=True)
    manager = SessionManager(build_app(), max_turns=args.max_turns)
    connect_jobs(manager)
    voice = build_voice() if args.voice else None

    if voice is not None and not (voice.can_transcribe and voice.can_speak):
        print(
            "Voice: some packages are missing (pip install -r requirements_voice.txt); "
            f"speech input {'on' if voice.can_transcribe else 'off'}, output {'on' if voice.can_speak else 'off'}.",
            flush=True,
        )

        if not (voice.can_transcribe or voice.can_speak):
            voice = None  # nothing installed: the page shows no voice controls

    token = new_token()
    origins = frontend_origins(args)
    server = None

    try:
        try:
            server = make_server(
                manager,
                token,
                host=args.host,
                port=args.port,
                allow_remote=args.allow_remote,
                notices=True,
                voice=voice,
                static_dir=STATIC_DIR if args.serve_frontend else None,
                frontend_origins=origins,
            )
        except OSError as exc:
            sys.exit(f"Could not listen on {args.host}:{args.port} ({exc}). Try another --port.")

        host, port = server.server_address[:2]

        if args.serve_frontend:
            print(f"Open {build_url(args.host, port, token)}")
        else:
            print(f"Agent API on http://{_shown_host(args.host)}:{port}")
            print("Start the frontend (npm --prefix frontend start), then open:")
            print(f"  {frontend_url(origins[0], args.host, port, token)}")
        print("Press Ctrl+C to stop.", flush=True)
        install_stop_signals()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        manager.stop_all()

        if server is not None:
            server.server_close()

        cleanup()


if __name__ == "__main__":
    main()
