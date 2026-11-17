"""Explain why the Postgres connection fails (pure helpers; no database access)."""

import re

OK = "ok"
WRONG_SERVER = "wrong_server"
BAD_CREDENTIALS = "bad_credentials"
UNREACHABLE = "unreachable"
UNKNOWN = "unknown"

# Processes that forward a port to a Docker container (Docker Desktop / WSL).
DOCKER_PROCESSES = ("docker", "wslrelay", "vpnkit")

_LISTEN_LINE = re.compile(r"^\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$", re.IGNORECASE)


def parse_netstat(text: str, port: int) -> list[int]:
    """PIDs listening (TCP) on `port`, from `netstat -ano` output."""
    pids: list[int] = []
    for line in text.splitlines():
        match = _LISTEN_LINE.match(line)
        if match and int(match.group(1)) == port:
            pid = int(match.group(2))
            if pid not in pids:
                pids.append(pid)
    return pids


def is_docker_process(name: str) -> bool:
    return any(part in name.lower() for part in DOCKER_PROCESSES)


def classify(error: str | None, owners: list[str]) -> str:
    """Name the cause. `owners` are the process names listening on the DB port."""
    if error is None:
        return OK

    lowered = error.lower()
    if "authentication failed" in lowered:
        # Right credentials, wrong server: something other than Docker owns the port. Windows
        # lets a native server bind the same port beside Docker, and it may be the one answering.
        if any(not is_docker_process(name) for name in owners):
            return WRONG_SERVER
        return BAD_CREDENTIALS
    if "refused" in lowered or "timeout" in lowered or "timed out" in lowered:
        return UNREACHABLE
    return UNKNOWN


def redact_url(url: str) -> str:
    """Hide the password in a database URL."""
    return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", url)
