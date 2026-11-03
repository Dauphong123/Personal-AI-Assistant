"""Speech runs in parallel with the streamed answer: sentences are spoken as they are generated."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from tests import stubs

stubs.install()

import chat_loop  # noqa: E402
import turns  # noqa: E402
from chat_loop import ChatLoop  # noqa: E402
from graph import ModelGraph, node  # noqa: E402
from jobs import JobManager  # noqa: E402
from tests.test_chat_loop import wait_for  # noqa: E402
from tests.test_voice import SPEECH, WAKE, FakeChat, VoiceTestCase  # noqa: E402
from turns import SpeechStream, TurnCancelled  # noqa: E402


def chunk(text, id="m1", tool_call_chunks=()):
    return SimpleNamespace(id=id, content=text, tool_call_chunks=list(tool_call_chunks))


def stream_of(*texts, id="m1", hold=False):
    """Feed text pieces to a SpeechStream and return what it said."""
    said = []
    stream = SpeechStream(said.append)

    for text in texts:
        stream.feed(chunk(text, id), hold=hold)

    stream.end_message()
    return said


class SentenceStreamTests(unittest.TestCase):
    def test_sentences_are_spoken_as_soon_as_they_complete(self):
        said = []
        stream = SpeechStream(said.append)
        stream.feed(chunk("Hello th"))
        self.assertEqual(said, [])
        stream.feed(chunk("ere. How a"))
        self.assertEqual(said, ["Hello there."])
        stream.feed(chunk("re you? Fine"))
        self.assertEqual(said, ["Hello there.", "How are you?"])
        stream.end_message()
        self.assertEqual(said, ["Hello there.", "How are you?", "Fine"])

    def test_numbers_and_file_names_do_not_end_a_sentence(self):
        self.assertEqual(stream_of("Pi is 3.14 and the file is memories/store.py. Done."), ["Pi is 3.14 and the file is memories/store.py.", "Done."])

    def test_line_breaks_end_a_sentence(self):
        self.assertEqual(stream_of("- first item\n- second item\n"), ["- first item", "- second item"])

    def test_vietnamese_sentences(self):
        self.assertEqual(stream_of("Xin chào! Bạn khỏe không? Tôi khỏe."), ["Xin chào!", "Bạn khỏe không?", "Tôi khỏe."])

    def test_symbol_only_pieces_are_skipped(self):
        self.assertEqual(stream_of("Intro.\n---\nNext."), ["Intro.", "Next."])

    def test_messages_that_call_tools_are_not_spoken(self):
        said = []
        stream = SpeechStream(said.append)
        stream.feed(chunk("Let me look at the files. "), hold=True)
        stream.feed(chunk("", tool_call_chunks=[{"name": "list_files"}]), hold=True)
        stream.end_message()
        self.assertEqual(said, [], "narration before a tool call is a working note")

    def test_the_coding_agent_holds_back_the_newest_sentence_until_the_next_begins(self):
        said = []
        stream = SpeechStream(said.append)
        stream.feed(chunk("First sentence. "), hold=True)
        self.assertEqual(said, [], "the newest sentence waits in case a tool call follows")
        stream.feed(chunk("Second sentence. "), hold=True)
        self.assertEqual(said, ["First sentence."])
        stream.end_message()
        self.assertEqual(said, ["First sentence.", "Second sentence."])

    def test_a_held_sentence_is_spoken_when_the_message_ends(self):
        self.assertEqual(stream_of("Only one sentence. ", hold=True), ["Only one sentence."])

    def test_plain_chat_speaks_immediately(self):
        said = []
        SpeechStream(said.append).feed(chunk("Instant sentence. "), hold=False)
        self.assertEqual(said, ["Instant sentence."])

    def test_a_whole_message_with_tool_calls_is_not_spoken(self):
        said = []
        stream = SpeechStream(said.append)
        stream.feed(AIMessage(content="Checking now.", tool_calls=[{"name": "x", "args": {}, "id": "1"}], id="a"))
        stream.end_message()
        self.assertEqual(said, [])

    def test_the_answer_after_a_tool_round_is_spoken(self):
        said = []
        stream = SpeechStream(said.append)
        stream.feed(chunk("", id="m1", tool_call_chunks=[{"name": "list_files"}]))
        stream.feed(chunk("There are three files. ", id="m2"))
        stream.end_message()
        self.assertEqual(said, ["There are three files."])

    def test_a_message_without_an_id_is_reset_by_end_message(self):
        said = []
        stream = SpeechStream(said.append)
        stream.feed(chunk("", id=None, tool_call_chunks=[{"name": "t"}]))
        stream.end_message()  # what stream_turn calls when the node changes
        stream.feed(chunk("The answer.", id=None))
        stream.end_message()
        self.assertEqual(said, ["The answer."])

    def test_code_blocks_are_not_read_even_when_split_across_chunks(self):
        said = stream_of("Run this:\n```py", "thon\nprint(1)\n", "```\nThen it prints one.")
        self.assertEqual(said, ["Run this:", "(code omitted)", "Then it prints one."])

    def test_an_unclosed_code_block_is_dropped(self):
        self.assertEqual(stream_of("Here:\n```python\nprint(1)"), ["Here:"])

    def test_speech_stops_after_the_limit(self):
        said = []
        stream = SpeechStream(said.append, limit=30)

        for text in ["First sentence here. ", "Second sentence here. ", "Third sentence here. ", "Fourth one. "]:
            stream.feed(chunk(text))

        stream.end_message()
        self.assertEqual(said, ["First sentence here.", "The rest is on screen."])

    def test_the_limit_message_follows_the_language(self):
        said = []
        stream = SpeechStream(said.append, limit=10)
        stream.feed(chunk("Xin chào, đây là một câu rất dài. Câu thứ hai."))
        stream.end_message()
        self.assertEqual(said, ["Phần còn lại hiển thị trên màn hình."])

    def test_nothing_emitted_means_not_emitted(self):
        stream = SpeechStream(lambda s: None)
        stream.feed(chunk("", tool_call_chunks=[{"name": "t"}]))
        stream.end_message()
        self.assertFalse(stream.emitted)


class FakeApp:
    """Stands in for the compiled graph: streams the given (node, chunk) pairs."""

    def __init__(self, pairs):
        self.pairs = pairs

    def stream(self, graph_input, config=None, stream_mode=None):
        for node_name, piece in self.pairs:
            yield piece, {"langgraph_node": node_name}

    def get_state(self, config):
        return SimpleNamespace(tasks=[], values={"messages": []})


class StreamTurnTests(unittest.TestCase):
    def run_turn(self, pairs, cancel=None, on_text=None):
        said = []
        text = []
        spoke = turns.run_turn(
            FakeApp(pairs), "q", {}, on_text or text.append, lambda r: True, cancel, on_speech=said.append
        )
        return said, text, spoke

    def test_only_the_answer_is_spoken_not_tool_output(self):
        pairs = [
            ("code_tools", chunk("", id="a", tool_call_chunks=[{"name": "read_file"}])),
            ("tool_execute", chunk("SECRET FILE CONTENTS. more.", id="t")),
            ("code_tools", chunk("It reads a file. ", id="b")),
            ("code_tools", chunk("Done.", id="b")),
        ]
        said, text, spoke = self.run_turn(pairs)
        self.assertEqual(said, ["It reads a file.", "Done."])
        self.assertTrue(spoke)
        self.assertIn("SECRET FILE CONTENTS", "".join(text), "tool output still streams to the screen")

    def test_without_a_speech_callback_nothing_changes(self):
        text = []
        spoke = turns.run_turn(FakeApp([("direct", chunk("Hi there."))]), "q", {}, text.append, lambda r: True)
        self.assertFalse(spoke)
        self.assertEqual(text, ["Hi there."])

    def test_a_cancelled_turn_does_not_speak_its_unfinished_sentence(self):
        cancel = threading.Event()
        pairs = [
            ("direct", chunk("Complete one. Half a sen")),
            ("direct", chunk("tence that never finishes")),
        ]
        said = []

        with self.assertRaises(TurnCancelled):
            turns.run_turn(
                FakeApp(pairs), "q", {}, lambda t: cancel.set(), lambda r: True, cancel, on_speech=said.append
            )

        self.assertEqual(said, ["Complete one."])


class SlowModel(BaseChatModel):
    """A chat model that streams its answer piece by piece, slowly."""

    pieces: list
    delay: float = 0.15
    finished: threading.Event = None  # type: ignore[assignment]

    @property
    def _llm_type(self):
        return "slow-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="".join(self.pieces)))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for piece in self.pieces:
            time.sleep(self.delay)
            generation = ChatGenerationChunk(message=AIMessageChunk(content=piece))

            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=generation)

            yield generation

        self.finished.set()


class RealGraphSpeechTests(unittest.TestCase):
    def setUp(self):
        self.model_finished = threading.Event()
        self.replies = []

        def tools_chain(inputs):
            return self.replies.pop(0)

        router = RunnableLambda(lambda x: SimpleNamespace(destination="code"))
        for p in [
            patch.object(node, "router_chain", router),
            patch.object(node, "code_chain", RunnableLambda(tools_chain)),
        ]:
            p.start()
            self.addCleanup(p.stop)

        self.app = ModelGraph.compile(checkpointer=InMemorySaver())
        self.config = {"configurable": {"thread_id": "s"}}

    def test_speech_starts_while_the_model_is_still_generating(self):
        model = SlowModel(
            pieces=["First sentence. ", "Second ", "sentence. ", "Third sentence."],
            finished=self.model_finished,
        )
        node_chain = RunnableLambda(lambda inputs, config: model.invoke(inputs["messages"], config))

        with patch.object(node, "code_chain", node_chain):
            heard = []
            started_before_the_end = []

            def on_speech(sentence):
                heard.append(sentence)
                started_before_the_end.append(not self.model_finished.is_set())

            turns.run_turn(self.app, "go", self.config, lambda t: None, lambda r: True, on_speech=on_speech)

        self.assertEqual(heard, ["First sentence.", "Second sentence.", "Third sentence."])
        self.assertTrue(started_before_the_end[0], "the first sentence must be spoken before generation ends")

    def test_tool_round_then_answer_speaks_only_the_answer(self):
        self.replies = [
            AIMessage(content="I will list the files.", tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "c1"}]),
            AIMessage(content="The folder has several files."),
        ]
        said = []
        turns.run_turn(self.app, "what is here?", self.config, lambda t: None, lambda r: True, on_speech=said.append)
        self.assertEqual(said, ["The folder has several files."])


class ChatLoopHookTests(unittest.TestCase):
    def make_chat(self, run_turn_result):
        calls = []

        def fake_run_turn(app, text, config, on_text, approve, cancel, on_speech=None, speech_limit=1200, notice=False):
            calls.append((on_speech, speech_limit))
            return run_turn_result

        app = SimpleNamespace(
            get_state=lambda config: SimpleNamespace(values={"messages": [AIMessage(content="Final answer.")]}, tasks=[])
        )
        replies = []
        chat = ChatLoop(app, {}, JobManager(), write=lambda s: None, on_reply=replies.append)
        chat.on_speech = lambda s: None
        chat.speech_limit = 77
        patcher = patch.object(chat_loop, "run_turn", fake_run_turn)
        patcher.start()
        self.addCleanup(patcher.stop)
        chat.start()
        self.addCleanup(chat.close)
        return chat, calls, replies

    def test_the_speech_hook_and_limit_are_passed_to_the_turn(self):
        chat, calls, _ = self.make_chat(True)
        chat.submit("hello")
        self.assertTrue(wait_for(lambda: calls))
        self.assertIs(calls[0][0], chat.on_speech)
        self.assertEqual(calls[0][1], 77)

    def test_an_answer_that_was_spoken_while_streaming_is_not_repeated(self):
        chat, _, replies = self.make_chat(True)
        chat.submit("hello")
        self.assertTrue(wait_for(lambda: not chat.busy))
        self.assertEqual(replies, [])

    def test_an_answer_that_was_not_streamed_is_still_delivered(self):
        chat, _, replies = self.make_chat(False)
        chat.submit("hello")
        self.assertTrue(wait_for(lambda: replies))
        self.assertEqual(replies, ["Final answer."])

    def test_stopping_notifies_the_cancel_hook(self):
        chat, _, _ = self.make_chat(True)
        fired = []
        chat.on_cancel = lambda: fired.append(1)
        chat.stop()
        self.assertEqual(fired, [1])

    def test_a_failing_cancel_hook_does_not_break_stop(self):
        chat, _, _ = self.make_chat(True)

        def boom():
            raise RuntimeError("x")

        chat.on_cancel = boom
        chat.stop()  # must not raise


class VoiceStreamingTests(VoiceTestCase):
    def test_streamed_sentences_are_spoken_while_the_agent_is_still_working(self):
        self.build()
        self.chat.busy = True
        thread = threading.Thread(target=self.assistant.run, daemon=True)
        thread.start()
        self.chat.on_speech("First sentence.")
        self.assertTrue(wait_for(lambda: self.speaker.said))
        self.assertTrue(self.chat.busy, "the agent is still generating while the first sentence is spoken")
        self.chat.on_speech("Second sentence.")
        self.assertTrue(wait_for(lambda: len(self.speaker.said) == 2))
        self.chat.busy = False
        self.assertTrue(wait_for(lambda: self.assistant.cycles >= 1))
        self.assistant.request_stop()
        thread.join(5)
        self.assertEqual(" ".join(self.speaker.said), "First sentence. Second sentence.")

    def test_sentences_that_pile_up_are_spoken_together(self):
        self.build()
        for sentence in ["One.", "Two.", "Three."]:
            self.chat.on_speech(sentence)

        self.assertEqual(self.assistant._take_pending_speech(), "One. Two. Three.")
        self.assertEqual(self.assistant._take_pending_speech(), "")

    def test_cancelling_drops_unspoken_sentences_and_cuts_the_speaker(self):
        self.build()
        self.chat.on_speech("Never spoken.")
        self.chat.on_cancel()
        self.assertEqual(self.assistant._take_pending_speech(), "")
        self.assertEqual(self.speaker.stops, 1)

    def test_assistant_registers_all_the_hooks(self):
        self.build()
        self.assertIsNotNone(self.chat.on_speech)
        self.assertIsNotNone(self.chat.on_reply)
        self.assertIsNotNone(self.chat.on_cancel)
        self.assertEqual(self.chat.speech_limit, self.assistant.config.max_spoken_chars)

    def test_the_wake_word_is_still_heard_between_spoken_bursts(self):
        chat = FakeChat()
        chat.busy = True
        self.build(frames=[WAKE], audios=[SPEECH], texts=["Jarvis, stop"], chat=chat)
        chat.on_speech = self.assistant.chat.on_speech
        self.run_until(lambda: chat.submitted)
        self.assertEqual(chat.submitted, ["stop"])


class EndToEndSpeechTests(VoiceTestCase):
    """Real ChatLoop and graph feeding the voice assistant: nothing is spoken twice."""

    def test_the_answer_is_spoken_once(self):
        router = RunnableLambda(lambda x: SimpleNamespace(destination="direct"))
        reply = RunnableLambda(lambda x: AIMessage(content="Your project has three layers. Each has tests."))

        with patch.object(node, "router_chain", router), patch.object(node, "direct_chain", reply):
            app = ModelGraph.compile(checkpointer=InMemorySaver())
            chat = ChatLoop(app, {"configurable": {"thread_id": "e"}}, JobManager(), write=lambda s: None)
            chat.start()
            self.addCleanup(chat.close)
            self.build(frames=[WAKE], audios=[SPEECH], texts=["Jarvis, explain my project"], chat=chat)
            self.run_until(lambda: self.speaker.said and not chat.busy and self.assistant._replies.empty())

        self.assertEqual(" ".join(self.speaker.said), "Your project has three layers. Each has tests.")


if __name__ == "__main__":
    unittest.main()
