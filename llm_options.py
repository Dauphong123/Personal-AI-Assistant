"""Settings shared by every ChatOllama in the project (read from the environment).

Measured on a laptop where qwen3:8b runs mostly on the CPU (about 4 tokens/s):
Qwen3's hidden "thinking" tokens made a two-sentence answer take 56 s instead of 13 s,
and a one-word routing call 113 s instead of 3 s. So thinking is off by default.
"""

import os

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off", ""}
MIN_CONTEXT = 256


def _flag(env, name: str) -> bool:
    raw = env.get(name, "false").strip().lower()

    if raw in TRUE:
        return True

    if raw in FALSE:
        return False

    raise ValueError(f"{name} must be true or false, got {raw!r}")


def ollama_options(env=None) -> dict:
    env = os.environ if env is None else env
    options: dict = {
        # False: the model answers directly. Set LLM_REASONING=true for harder tasks.
        "reasoning": _flag(env, "LLM_REASONING"),
        # Keep the model in memory between messages; Ollama's default unloads it after 5 minutes,
        # and reloading a multi-GB model from disk is slow.
        "keep_alive": env.get("LLM_KEEP_ALIVE", "30m").strip() or "30m",
    }

    num_ctx = (env.get("LLM_NUM_CTX") or "").strip()

    if num_ctx:
        try:
            value = int(num_ctx)
        except ValueError:
            value = 0

        if value < MIN_CONTEXT:
            raise ValueError(f"LLM_NUM_CTX must be a whole number of at least {MIN_CONTEXT}, got {num_ctx!r}")

        options["num_ctx"] = value

    return options
