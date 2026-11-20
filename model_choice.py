"""Which model answers a chat (F036): installed Ollama models, or Claude Code run headless with the
user's own Claude login. The pick lives in model_settings.json (gitignored); .env is never written."""

import json
import logging
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CLAUDE_PREFIX = "claude-code:"
CLAUDE_ALIASES: tuple[str, ...] = ("opus", "sonnet", "haiku", "fable")
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "medium"
SETTINGS_FILE = Path("model_settings.json")
OLLAMA_URL = "http://127.0.0.1:11434"
USAGE = "/model <name> [effort]"

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class ModelChoice:
    model: str
    effort: str | None

    @property
    def is_claude(self) -> bool:
        return self.model.startswith(CLAUDE_PREFIX)

    @property
    def alias(self) -> str | None:
        return self.model[len(CLAUDE_PREFIX) :] if self.is_claude else None

    def label(self) -> str:
        return f"{self.model} ({self.effort})" if self.effort else self.model


def _usage(problem: str) -> ValueError:
    return ValueError(
        f"{problem}. Usage: {USAGE}; Claude models: {', '.join(CLAUDE_ALIASES)}; efforts: {', '.join(EFFORTS)}."
    )


def parse_choice(text: str) -> ModelChoice:
    """'claude-code:opus high', 'opus', 'qwen3:4b' -> a ModelChoice; anything else -> ValueError."""
    parts = text.split()

    if not 1 <= len(parts) <= 2:
        raise _usage("Name one model")

    name, effort = parts[0], (parts[1].lower() if len(parts) == 2 else None)

    if name.lower() in CLAUDE_ALIASES:
        name = CLAUDE_PREFIX + name.lower()

    if name.lower().startswith(CLAUDE_PREFIX):
        alias = name[len(CLAUDE_PREFIX) :].lower()

        if alias not in CLAUDE_ALIASES:
            raise _usage(f"Unknown Claude model {alias!r}")

        if effort is not None and effort not in EFFORTS:
            raise _usage(f"Unknown effort {effort!r}")

        return ModelChoice(CLAUDE_PREFIX + alias, effort or DEFAULT_EFFORT)

    if effort is not None:
        raise _usage(f"Only Claude models take an effort, not {name!r}")

    return ModelChoice(name, None)


def claude_models() -> list[str]:
    return [CLAUDE_PREFIX + alias for alias in CLAUDE_ALIASES]


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 (local Ollama address)
        return response.read()


def ollama_models(env=None, fetch: Fetch | None = None) -> list[str]:
    """Installed Ollama chat models, newest first (embedding models left out). OSError if unreachable."""
    env = os.environ if env is None else env
    base = (env.get("OLLAMA_HOST") or OLLAMA_URL).strip().rstrip("/")

    if "://" not in base:
        base = "http://" + base

    try:
        data = json.loads((fetch or _fetch)(f"{base}/api/tags"))
        models = [m for m in data.get("models", []) if isinstance(m, dict) and isinstance(m.get("name"), str)]
    except (ValueError, AttributeError) as exc:
        raise OSError(f"Ollama sent an unreadable model list: {exc}") from exc

    models = [m for m in models if "embed" not in m["name"].lower()]
    models.sort(key=lambda m: str(m.get("modified_at", "")), reverse=True)
    return [m["name"] for m in models]


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring %s: %s", path, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("Ignoring %s: not a JSON object", path)
        return {}

    return data


def save_settings(data: dict, path: Path = SETTINGS_FILE) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _base_model(env) -> str:
    return (env.get("MODEL") or "").strip()


def available_models(env=None, fetch: Fetch | None = None, path: Path = SETTINGS_FILE) -> list[str]:
    """Live Ollama list (remembered for when Ollama is down), MODEL always included, then Claude Code."""
    env = os.environ if env is None else env

    try:
        local = ollama_models(env, fetch)
        try:
            save_settings({**load_settings(path), "ollama_models": local}, path)
        except OSError:
            logger.warning("Could not save the model list to %s", path, exc_info=True)
    except OSError as exc:
        logger.info("Ollama model list unavailable: %s", exc)
        saved = load_settings(path).get("ollama_models")
        local = [m for m in saved if isinstance(m, str)] if isinstance(saved, list) else []

    base = _base_model(env)
    if base and base not in local:
        local.append(base)

    return local + claude_models()


def check_available(choice: ModelChoice, models: list[str]) -> None:
    """ValueError for an Ollama model that is not in `models` (Claude Code is always offered)."""
    if not choice.is_claude and choice.model not in models:
        raise _usage(f"Model {choice.model!r} is not installed in Ollama (see /model)")


def startup_choice(env=None, path: Path = SETTINGS_FILE, fetch: Fetch | None = None) -> ModelChoice:
    """default_choice, checked against the live Ollama list when Ollama answers."""
    try:
        installed: list[str] | None = ollama_models(env, fetch)
    except OSError:
        installed = None
    return default_choice(env, path, installed)


def default_choice(env=None, path: Path = SETTINGS_FILE, installed: list[str] | None = None) -> ModelChoice:
    """The saved pick, or MODEL when there is none or it no longer makes sense (with `installed`,
    an Ollama model that is no longer installed)."""
    env = os.environ if env is None else env
    fallback = ModelChoice(_base_model(env), None)
    settings = load_settings(path)
    name = settings.get("chat_model")

    if name is None:
        return fallback

    effort = settings.get("effort")

    try:
        if not isinstance(name, str) or not (effort is None or isinstance(effort, str)):
            raise ValueError("chat_model and effort must be text")
        choice = parse_choice(f"{name} {effort}" if effort else name)
        if installed is not None:
            check_available(choice, installed)
        return choice
    except ValueError as exc:
        logger.warning("Ignoring the saved model in %s (%s); using MODEL=%s", path, exc, fallback.model)
        return fallback


def save_default(choice: ModelChoice, path: Path = SETTINGS_FILE) -> None:
    save_settings({**load_settings(path), "chat_model": choice.model, "effort": choice.effort}, path)


def privacy_notice(choice: ModelChoice) -> str | None:
    if not choice.is_claude:
        return None

    return (
        "Claude Code runs this chat with your Claude login: your messages, long-term memory "
        "and the files it reads go to Anthropic."
    )
