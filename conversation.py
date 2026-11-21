"""User messages and job notices in a conversation.

A job notice is a message the app sends when a background `terminal` job ends
(`jobs.finished_notice`). It is a human-role message flagged `additional_kwargs["notice"]`,
so the model reads it as input, but it is not something the user said: memory neither recalls
for nor learns from it, the rule router and the Claude Code recap skip it.

It still starts a turn of its own: `latest_turn_start` counts it like a user message, so a
notice turn gets its own tool-call budget and `summarize` may split the history there.
Only this module reads or writes the flag.
"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_NOTICE = "notice"


def user_message(text: str) -> HumanMessage:
    """Something the user said."""
    return HumanMessage(content=text)


def notice_message(text: str) -> HumanMessage:
    """A job notice (see the module docstring)."""
    return HumanMessage(content=text, additional_kwargs={_NOTICE: True})


def is_notice(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and bool(message.additional_kwargs.get(_NOTICE))


def latest_turn_start(messages: Sequence[BaseMessage]) -> int | None:
    """Index of the message the latest turn began with (a user message or a job notice)."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index

    return None


def visible_talk(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """What the user and the assistant said to each other: user messages (not notices) and the
    assistant's plain-text answers. Tool calls, tool results and system messages are left out."""
    return [
        message for message in messages
        if (isinstance(message, HumanMessage) and not is_notice(message))
        or (isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip())
    ]
