"""F038 step 1: transient Ollama errors are retried, but never after text has streamed."""

import threading
import unittest
from typing import Any

import httpx
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from ollama import ResponseError

import model_retry
from model_retry import invoke_with_retry, is_transient
from settings import check_settings, model_retries


class FlakyModel(BaseChatModel):
    """Raises the queued errors one per call, then answers. `tokens` streams before each error."""

    errors: list = []
    tokens_before_error: list[str] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "flaky"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls += 1
        if self.errors:
            for token in self.tokens_before_error:
                if run_manager is not None:
                    run_manager.on_llm_new_token(token, chunk=ChatGenerationChunk(message=AIMessageChunk(content=token)))
            raise self.errors.pop(0)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="answer"))])


class Collect(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str | list[str | dict[str, Any]], **kwargs: Any) -> None:
        self.tokens.append(str(token))


def no_sleep(cancel, seconds) -> bool:
    return False


class TransientTests(unittest.TestCase):
    def test_connection_and_server_errors_are_transient(self):
        self.assertTrue(is_transient(ConnectionError("Failed to connect to Ollama")))
        self.assertTrue(is_transient(httpx.ConnectError("refused")))
        self.assertTrue(is_transient(httpx.ReadTimeout("slow")))
        self.assertTrue(is_transient(httpx.RemoteProtocolError("closed")))
        self.assertTrue(is_transient(ResponseError("server busy", 503)))
        self.assertTrue(is_transient(ResponseError("boom", 500)))
        self.assertTrue(is_transient(ResponseError("slow down", 429)))

    def test_request_and_program_errors_are_not(self):
        self.assertFalse(is_transient(ResponseError("model 'x' not found", 404)))
        self.assertFalse(is_transient(ResponseError("bad request", 400)))
        self.assertFalse(is_transient(ValueError("bug")))
        self.assertFalse(is_transient(KeyboardInterrupt()))


class RetryTests(unittest.TestCase):
    def test_a_transient_error_is_retried_then_answers(self):
        model = FlakyModel(errors=[ConnectionError("down"), ResponseError("busy", 503)])
        reply = invoke_with_retry(model, "hi", retries=2, wait=no_sleep)

        self.assertEqual(reply.content, "answer")
        self.assertEqual(model.calls, 3)

    def test_gives_up_after_the_retries(self):
        model = FlakyModel(errors=[ConnectionError("1"), ConnectionError("2"), ConnectionError("3")])

        with self.assertRaises(ConnectionError):
            invoke_with_retry(model, "hi", retries=2, wait=no_sleep)
        self.assertEqual(model.calls, 3)

    def test_a_permanent_error_is_not_retried(self):
        model = FlakyModel(errors=[ResponseError("not found", 404)])

        with self.assertRaises(ResponseError):
            invoke_with_retry(model, "hi", retries=2, wait=no_sleep)
        self.assertEqual(model.calls, 1)

    def test_no_retry_once_text_has_streamed(self):
        # A retry would show the user the start of the answer twice.
        model = FlakyModel(errors=[httpx.RemoteProtocolError("closed")], tokens_before_error=["Hel"])

        with self.assertRaises(httpx.RemoteProtocolError):
            invoke_with_retry(model, "hi", retries=2, wait=no_sleep)
        self.assertEqual(model.calls, 1)

    def test_callbacks_of_the_caller_still_see_the_tokens(self):
        collect = Collect()
        model = FlakyModel(errors=[httpx.RemoteProtocolError("closed")], tokens_before_error=["Hel"])

        with self.assertRaises(httpx.RemoteProtocolError):
            invoke_with_retry(model, "hi", {"callbacks": [collect]}, retries=2, wait=no_sleep)
        self.assertEqual(collect.tokens, ["Hel"])

    def test_waits_grow_between_tries(self):
        waits: list[float] = []
        model = FlakyModel(errors=[ConnectionError("1"), ConnectionError("2")])

        invoke_with_retry(model, "hi", retries=2, wait=lambda cancel, s: waits.append(s) or False)
        self.assertEqual(waits, list(model_retry.DELAYS[:2]))

    def test_stop_during_the_wait_ends_the_retries(self):
        cancel = threading.Event()
        cancel.set()
        model = FlakyModel(errors=[ConnectionError("down")])

        with self.assertRaises(ConnectionError):
            invoke_with_retry(model, "hi", {"configurable": {"cancel": cancel}}, retries=2)
        self.assertEqual(model.calls, 1)

    def test_zero_retries_is_one_try(self):
        model = FlakyModel(errors=[ConnectionError("down")])

        with self.assertRaises(ConnectionError):
            invoke_with_retry(model, "hi", retries=0, wait=no_sleep)
        self.assertEqual(model.calls, 1)


class SettingTests(unittest.TestCase):
    def test_default_and_values(self):
        self.assertEqual(model_retries({}), 2)
        self.assertEqual(model_retries({"MODEL_RETRIES": "0"}), 0)
        self.assertEqual(model_retries({"MODEL_RETRIES": " 5 "}), 5)

    def test_bad_values_are_listed_at_startup(self):
        for raw in ("-1", "two", "11"):
            with self.subTest(raw=raw):
                problems = check_settings({"MODEL": "qwen3:8b", "MODEL_RETRIES": raw})
                self.assertTrue(any("MODEL_RETRIES" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
