"""What a job notice is in a conversation, compared with what the user said (conversation.py)."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import conversation
from conversation import is_notice, latest_turn_start, notice_message, user_message, visible_talk


class NoticeSemanticsTests(unittest.TestCase):
    def test_messages(self):
        user = user_message("hi")
        notice = notice_message("[system notice] job 3 ended")
        flagless_ai = AIMessage(content="x", additional_kwargs={"notice": True})  # only human messages count

        cases = [
            # (message, is_notice)
            (user, False),
            (notice, True),
            (HumanMessage(content="hi"), False),
            (HumanMessage(content="old", additional_kwargs={"notice": True}), True),  # saved before F039
            (AIMessage(content="answer"), False),
            (flagless_ai, False),
            (ToolMessage(content="out", tool_call_id="1"), False),
            (SystemMessage(content="sys"), False),
        ]
        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(is_notice(message), expected)

        self.assertIsInstance(user, HumanMessage)
        self.assertIsInstance(notice, HumanMessage)
        self.assertEqual((user.content, notice.content), ("hi", "[system notice] job 3 ended"))
        self.assertEqual(user.additional_kwargs, {})  # unchanged shape for checkpoints already saved

    def test_latest_turn_start(self):
        call = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
        result = ToolMessage(content="out", tool_call_id="1")
        cases = [
            # (messages, index) — a notice starts a turn of its own (own tool-call budget,
            # summarize may split there)
            ([], None),
            ([AIMessage(content="summary")], None),
            ([user_message("a")], 0),
            ([user_message("a"), AIMessage(content="b")], 0),
            ([user_message("a"), AIMessage(content="b"), user_message("c"), call, result], 2),
            ([user_message("a"), AIMessage(content="b"), notice_message("job")], 2),
            ([user_message("a"), notice_message("job"), call, result], 1),
        ]
        for messages, expected in cases:
            with self.subTest(messages=[m.type for m in messages]):
                self.assertEqual(latest_turn_start(messages), expected)

    def test_visible_talk(self):
        call = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
        messages = [
            user_message("question"),
            call,
            ToolMessage(content="out", tool_call_id="1"),
            AIMessage(content="answer"),
            notice_message("job ended"),
            AIMessage(content="the job is done"),
            AIMessage(content="   "),
            AIMessage(content=[{"type": "text", "text": "x"}]),  # not plain text
            SystemMessage(content="sys"),
            user_message("next"),
        ]
        self.assertEqual(
            [(m.type, m.content) for m in visible_talk(messages)],
            [("human", "question"), ("ai", "answer"), ("ai", "the job is done"), ("human", "next")],
        )

    def test_module_states_the_turn_rule(self):
        self.assertIn("turn", (conversation.__doc__ or "").lower())


if __name__ == "__main__":
    unittest.main()
