"""Logging for the CLI (`main.py`) and the web backend (`python -m backend`).

LOG_LEVEL sets what the console shows (default WARNING, so chat output stays clean).
LOG_FILE adds a rotating log file that keeps INFO and up with timestamps, e.g. tool
failures the model was told about but the console does not show.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(threadName)s]: %(message)s"
FILE_LEVEL = logging.INFO
FILE_MAX_BYTES = 5 * 1024 * 1024
FILE_BACKUPS = 3


def parse_level(value: str | None) -> int:
    name = (value or "WARNING").strip().upper()
    if name not in LEVELS:
        raise ValueError(f"LOG_LEVEL must be one of {', '.join(LEVELS)} (got {value!r}).")
    return logging.getLevelNamesMapping()[name]


def configure_logging(env=None, root: logging.Logger | None = None) -> None:
    """Replace the root logger's handlers. Raises ValueError for an unknown LOG_LEVEL."""
    env = os.environ if env is None else env
    root = logging.getLogger() if root is None else root
    console_level = parse_level(env.get("LOG_LEVEL"))

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)
    lowest = console_level

    log_file = (env.get("LOG_FILE") or "").strip()
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file = RotatingFileHandler(path, maxBytes=FILE_MAX_BYTES, backupCount=FILE_BACKUPS, encoding="utf-8")
        file.setLevel(FILE_LEVEL)
        file.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(file)
        lowest = min(lowest, FILE_LEVEL)

    root.setLevel(lowest)
