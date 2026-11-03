import argparse
import logging
import os
import sys
import threading

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import RunnableConfig

from chat_loop import ChatLoop
from graph import ModelGraph
from graph.node import wait_for_memory
from tools import JOBS


def read_keyboard(chat: ChatLoop) -> str:
    """Feed typed lines to the chat until the user leaves.

    Returns "exit" (typed exit/quit), "eof" (the input ended, e.g. a pipe) or "interrupt" (Ctrl+C)."""
    while True:
        try:
            line = input("User: ")
        except EOFError:
            return "eof"
        except KeyboardInterrupt:
            return "interrupt"

        if not chat.submit(line):
            return "exit"


def voice_requested(args: argparse.Namespace) -> bool:
    enabled = os.environ.get("VOICE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    return args.voice or enabled


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LangGraph coding assistant")
    parser.add_argument("--voice", action="store_true", help="enable the wake-word voice interface")
    args = parser.parse_args()

    # Windows consoles default to cp1252, which can't print emoji from the model.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore

    if not sys.stdin.isatty():
        # Piped input is decoded with the console code page otherwise, which garbles Vietnamese.
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # pyright: ignore

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    voice_config = None

    if voice_requested(args):
        from voice.config import VoiceConfig, VoiceConfigError

        try:
            voice_config = VoiceConfig.from_env()
        except VoiceConfigError as e:
            sys.exit(f"Invalid voice configuration: {e}")

    app = ModelGraph.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}

    chat = ChatLoop(app, config, JOBS)

    assistant = None

    if voice_config is not None:
        from voice import build_assistant

        try:
            assistant = build_assistant(voice_config, chat, show=chat.say)
        except Exception as e:
            sys.exit(f"Could not start voice mode: {e}")

    chat.start()

    try:
        # Turns run on the chat loop's worker thread, so input stays live meanwhile.
        if assistant is None:
            if read_keyboard(chat) == "eof":
                chat.wait_idle()  # piped input: let the answer finish before leaving
        else:
            # Voice runs here; the keyboard keeps working next to it.
            def keyboard() -> None:
                if read_keyboard(chat) == "exit":
                    assistant.request_stop()

            threading.Thread(target=keyboard, daemon=True).start()

            try:
                assistant.run()
            except KeyboardInterrupt:
                assistant.request_stop()
    finally:
        chat.close()
        JOBS.close()  # background commands must not outlive the app
        wait_for_memory()  # a save that is still running should finish


if __name__ == "__main__":
    main()
