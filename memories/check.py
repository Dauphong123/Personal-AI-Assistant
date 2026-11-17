"""Check the Postgres connection and explain any failure: python -m memories.check"""

import csv
import io
import subprocess
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from .database import get_database_url
from .diagnose import (
    BAD_CREDENTIALS,
    OK,
    UNREACHABLE,
    WRONG_SERVER,
    classify,
    is_docker_process,
    parse_netstat,
    redact_url,
)

SUGGESTED_HOST_PORT = 15432  # host port compose.yaml maps to the container


def port_owners(port: int, run=subprocess.run) -> list[str]:
    """Names of the processes listening on `port` (Windows: netstat + tasklist). Empty if unknown."""
    try:
        netstat = run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10)
        names: list[str] = []
        for pid in parse_netstat(netstat.stdout, port):
            tasks = run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for row in csv.reader(io.StringIO(tasks.stdout)):
                if len(row) > 1 and row[1] == str(pid) and row[0] not in names:
                    names.append(row[0])
        return names
    except (OSError, subprocess.SubprocessError):
        return []


def try_connect(url: str) -> str | None:
    """Return None if `SELECT 1` works, otherwise the error text."""
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return None
    except Exception as exc:
        return str(getattr(exc, "orig", None) or exc)
    finally:
        engine.dispose()


def run_check(url: str, connect=try_connect, owners=port_owners) -> tuple[str, str]:
    """Return (status, human-readable report). The password is never printed."""
    parsed = make_url(url)
    port = parsed.port or 5432
    error = connect(url)
    names = owners(port) if error is not None else []
    status = classify(error, names)

    lines = [f"Database: {redact_url(url)}"]
    if status == OK:
        lines.append("OK: connected.")
    elif status == WRONG_SERVER:
        foreign = [name for name in names if not is_docker_process(name)]
        lines += [
            f"FAILED: port {port} is answered by {', '.join(foreign)}, not by the Docker container,",
            f'so it rejects user "{parsed.username}".',
            "Fix (keeps the other server untouched):",
        ]
        if port == SUGGESTED_HOST_PORT:
            lines += [
                "  1. Pick a free port (netstat -ano | findstr LISTENING) and map the container",
                "     to it in compose.yaml.",
                "  2. Set DB_URL in .env to use that port.",
            ]
        else:
            lines += [
                f"  1. compose.yaml maps the container to host port {SUGGESTED_HOST_PORT}.",
                f"  2. Set DB_URL in .env to use localhost:{SUGGESTED_HOST_PORT}.",
            ]
        lines.append("  3. docker compose up -d, then run this check again.")
    elif status == BAD_CREDENTIALS:
        lines += [
            "FAILED: the server rejected the user or password.",
            "Check DB_URL against POSTGRES_USER / POSTGRES_PASSWORD in compose.yaml.",
            "The password in an existing volume does not change when compose.yaml changes;",
            "run `docker compose down -v` (deletes stored memories) to start fresh.",
        ]
    elif status == UNREACHABLE:
        lines += [
            f"FAILED: nothing is accepting connections on port {port}.",
            "Start Docker Desktop, then: docker compose up -d",
        ]
    else:
        lines.append(f"FAILED: {error}")
    return status, "\n".join(lines)


def explain_failure(error: object, owners=port_owners) -> str:
    """Short hint for a startup failure already caught (no second connection). Never raises."""
    pointer = "Run `python -m memories.check` for details."
    try:
        status, report = run_check(get_database_url(), connect=lambda url: str(error), owners=owners)
    except Exception:
        return pointer
    return f"{report}\n{pointer}"


def main(argv: list[str] | None = None) -> int:
    try:
        url = get_database_url()
    except RuntimeError as exc:
        print(exc)
        return 1
    status, report = run_check(url)
    print(report)
    return 0 if status == OK else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
