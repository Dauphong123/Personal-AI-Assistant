"""Retry a model call after a short Ollama failure (F038).

Only errors that a second try can fix are retried (no connection, a 5xx or 429 answer, a dropped
or timed-out request), and only while no text has streamed: the user already sees streamed text,
so a retry would show the start of the answer twice. A stop (`configurable["cancel"]`) ends the wait.
"""

import logging
import threading
from typing import Any, Callable
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.runnables import RunnableConfig, ensure_config
from ollama import ResponseError

logger = logging.getLogger(__name__)

DELAYS = (1.0, 2.0, 4.0, 8.0)
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

Wait = Callable[[threading.Event | None, float], bool]


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ResponseError):
        return exc.status_code in TRANSIENT_STATUS
    # ollama wraps a refused connection as ConnectionError; a streamed request raises httpx's own.
    return isinstance(exc, (ConnectionError, httpx.TransportError))


def is_model_error(exc: BaseException) -> bool:
    """Any failure of the Ollama call itself (not a bug in our code): reason to try another model."""
    return isinstance(exc, (ResponseError, ConnectionError, httpx.TransportError))


class _Streamed(BaseCallbackHandler):
    def __init__(self) -> None:
        self.any = False

    def on_llm_new_token(self, token: str | list[str | dict[str, Any]], *, run_id: UUID, **kwargs: Any) -> None:
        self.any = True


def _with_handler(config: RunnableConfig | None, handler: BaseCallbackHandler) -> RunnableConfig:
    # ensure_config merges the caller's context (LangGraph's streaming callbacks); passing
    # `callbacks` alone would replace them and the answer would stop streaming.
    config = ensure_config(config)
    callbacks = config.get("callbacks")

    if callbacks is None:
        callbacks = [handler]
    elif isinstance(callbacks, list):
        callbacks = [*callbacks, handler]
    else:
        assert isinstance(callbacks, BaseCallbackManager)
        callbacks = callbacks.copy()
        callbacks.add_handler(handler, inherit=True)

    return {**config, "callbacks": callbacks}


def _wait(cancel: threading.Event | None, seconds: float) -> bool:
    """Sleep; True if stopped meanwhile."""
    return (cancel or threading.Event()).wait(seconds)


def invoke_with_retry(runnable, inputs, config: RunnableConfig | None = None, retries: int | None = None, wait: Wait = _wait):
    if retries is None:
        from settings import model_retries

        retries = model_retries()

    config = ensure_config(config)  # a node called without config still gets the turn's
    cancel = config.get("configurable", {}).get("cancel")
    tries = 0

    while True:
        streamed = _Streamed()
        try:
            return runnable.invoke(inputs, _with_handler(config, streamed))
        except Exception as exc:
            if tries >= retries or streamed.any or not is_transient(exc):
                raise
            if cancel is not None and cancel.is_set():
                raise

            delay = DELAYS[min(tries, len(DELAYS) - 1)]
            tries += 1
            logger.warning("Model call failed (%s); retry %d of %d in %.0f s", exc, tries, retries, delay)

            if wait(cancel, delay):
                raise
