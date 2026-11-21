"""Startup check of the .env settings, so a typo stops the app with one clear list instead of
a traceback (or, for flags, silently meaning something else).

`dependency.py` runs `check_settings` right after loading .env, before any model loads.
Voice settings are checked separately by `voice.config.VoiceConfig.from_env` when voice is on.
"""

import math
import os
import re
from pathlib import Path

from llm_options import FALSE, TRUE, model_name, ollama_options
from log_setup import parse_level

DEFAULT_MAX_HISTORY_TOKENS = 5000
DEFAULT_MEMORY_IDLE_SECONDS = 3.0
# Flags read with a default; a value outside TRUE/FALSE is a typo, not "off".
FLAGS = ("LLM_REASONING", "LOG_TIMING", "WEB_SEARCH_ENABLED", "VOICE_ENABLED")


def max_history_tokens(env=None) -> int:
    env = os.environ if env is None else env
    raw = (env.get("MAX_HISTORY_TOKENS") or "").strip()

    if not raw:
        return DEFAULT_MAX_HISTORY_TOKENS

    try:
        value = int(raw)
    except ValueError:
        value = 0

    if value < 1:
        raise ValueError(f"MAX_HISTORY_TOKENS must be a positive whole number, got {raw!r}")

    return value


def memory_idle_seconds(env=None) -> float:
    env = os.environ if env is None else env
    raw = (env.get("MEMORY_IDLE_SECONDS") or "").strip()

    if not raw:
        return DEFAULT_MEMORY_IDLE_SECONDS

    try:
        value = float(raw)
    except ValueError:
        value = -1.0

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"MEMORY_IDLE_SECONDS must be a number of seconds (0 or more), got {raw!r}")

    return value


DEFAULT_MODEL_RETRIES = 2
MAX_MODEL_RETRIES = 10


def model_retries(env=None) -> int:
    """How often a model call is tried again after a short Ollama failure (model_retry.py)."""
    env = os.environ if env is None else env
    raw = (env.get("MODEL_RETRIES") or "").strip()

    if not raw:
        return DEFAULT_MODEL_RETRIES

    try:
        value = int(raw)
    except ValueError:
        value = -1

    if not 0 <= value <= MAX_MODEL_RETRIES:
        raise ValueError(f"MODEL_RETRIES must be a whole number from 0 to {MAX_MODEL_RETRIES}, got {raw!r}")

    return value


DEFAULT_TERMINAL_MEMORY_MB = 2048
MIN_TERMINAL_MEMORY_MB = 64


def terminal_memory_mb(env=None) -> int:
    """Memory cap of one terminal command and everything it starts (sandbox.py, Windows); 0 = none."""
    env = os.environ if env is None else env
    raw = (env.get("TERMINAL_MEMORY_MB") or "").strip()

    if not raw:
        return DEFAULT_TERMINAL_MEMORY_MB

    try:
        value = int(raw)
    except ValueError:
        value = -1

    if value != 0 and value < MIN_TERMINAL_MEMORY_MB:
        raise ValueError(f"TERMINAL_MEMORY_MB must be 0 (no cap) or at least {MIN_TERMINAL_MEMORY_MB}, got {raw!r}")

    return value


_CLAUDE_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\[\]-]*")


def claude_fallback_model(env=None) -> str:
    """Claude Code's --fallback-model list (comma-separated, in order); empty = none."""
    env = os.environ if env is None else env
    raw = (env.get("CLAUDE_FALLBACK_MODEL") or "").strip()

    if not raw:
        return ""

    names = [name.strip() for name in raw.split(",")]

    if not all(_CLAUDE_MODEL.fullmatch(name) for name in names):
        raise ValueError(f"CLAUDE_FALLBACK_MODEL must be Claude model names separated by commas, e.g. haiku, got {raw!r}")

    return ",".join(names)


def _check_flag(env, name: str) -> None:
    raw = (env.get(name) or "").strip().lower()

    if raw not in TRUE | FALSE:
        raise ValueError(f"{name} must be true or false, got {raw!r}")


def _check_project_root(env) -> None:
    raw = (env.get("PROJECT_ROOT") or "").strip()

    if raw and not Path(raw).is_dir():
        raise ValueError(f"PROJECT_ROOT must be an existing directory, got {raw!r}")


def check_settings(env=None) -> list[str]:
    """Every problem found in the settings, one message each (empty when all is well)."""
    env = os.environ if env is None else env
    checks = [
        lambda: model_name(env),
        lambda: ollama_options(env),
        lambda: max_history_tokens(env),
        lambda: memory_idle_seconds(env),
        lambda: model_retries(env),
        lambda: claude_fallback_model(env),
        lambda: terminal_memory_mb(env),
        lambda: parse_level(env.get("LOG_LEVEL")),
        lambda: _check_project_root(env),
        *(lambda name=name: _check_flag(env, name) for name in FLAGS if name != "LLM_REASONING"),
    ]
    problems = []

    for check in checks:
        try:
            check()
        except ValueError as exc:
            problems.append(str(exc))

    return problems


def settings_error(problems: list[str]) -> str:
    lines = "\n".join(f"  - {problem}" for problem in problems)
    return f"Fix these settings in .env (see .env.example):\n{lines}"
