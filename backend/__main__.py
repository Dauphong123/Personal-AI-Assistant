"""Start the web chat backend: python -m backend [--port N] [--max-turns N]

Serves the API and the built TypeScript frontend (webui/dist) on one origin.
"""

import argparse
import logging
import sys
from pathlib import Path

from .server import BUILD_HINT, LOOPBACK_HOSTS, STATIC_DIR, make_server, new_token
from .sessions import SessionManager


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
        "--max-turns",
        type=int,
        default=1,
        help="how many sessions may run a turn at the same time; others wait (default: 1)",
    )
    return parser.parse_args(argv)


def build_url(host: str, port: int, token: str) -> str:
    shown = "localhost" if host in {"0.0.0.0", "::"} else host
    return f"http://{shown}:{port}/?token={token}"


def frontend_built(static_dir: Path = STATIC_DIR) -> bool:
    return (static_dir / "index.html").is_file()


def build_app():
    """The compiled graph, as main.py builds it. Importing it loads the models (slow)."""
    from langgraph.checkpoint.memory import InMemorySaver

    from graph import ModelGraph

    return ModelGraph.compile(checkpointer=InMemorySaver())


def connect_jobs(manager: SessionManager) -> None:
    """Finished background jobs notify the session that started them."""
    from tools import JOBS

    JOBS.on_finish = manager.job_finished


def cleanup() -> None:
    from graph.node import wait_for_memory
    from tools import JOBS

    JOBS.close()  # background commands must not outlive the app
    wait_for_memory()  # a save that is still running should finish


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.host not in LOOPBACK_HOSTS and not args.allow_remote:
        sys.exit(
            f"Refusing to listen on {args.host}: the assistant can run commands and edit files. "
            "Pass --allow-remote if you really want that."
        )

    if args.max_turns < 1:
        sys.exit("--max-turns must be at least 1.")

    if not frontend_built():
        sys.exit(BUILD_HINT)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    print("Loading models...", flush=True)
    manager = SessionManager(build_app(), max_turns=args.max_turns)
    connect_jobs(manager)
    token = new_token()
    server = None

    try:
        try:
            server = make_server(
                manager, token, host=args.host, port=args.port, allow_remote=args.allow_remote, notices=True
            )
        except OSError as exc:
            sys.exit(f"Could not listen on {args.host}:{args.port} ({exc}). Try another --port.")

        host, port = server.server_address[:2]
        print(f"Open {build_url(args.host, port, token)}")
        print("Press Ctrl+C to stop.", flush=True)

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
