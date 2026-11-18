import threading
import time
import unittest

from tests import stubs

stubs.install()

from turns import TurnCancelled  # noqa: E402
from backend.session import ChatSession  # noqa: E402


def wait_for(session, kind, after=0, timeout=3.0):
    """Block until an event of this type exists; return all events after `after`."""
    seen = []
    cursor = after
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        new = session.events_after(cursor, timeout=0.1)
        seen += new
        if new:
            cursor = new[-1]["id"]
        if any(e["type"] == kind for e in seen):
            return seen
    raise AssertionError(f"no {kind!r} event; got {[e['type'] for e in seen]}")


def scripted(*chunks):
    def run(app, text, config, on_text, approve, cancel=None, **kwargs):
        for chunk in chunks:
            on_text(chunk)
        return False

    return run


class SendTests(unittest.TestCase):
    def test_streams_tokens_then_done(self):
        session = ChatSession(app=None, run=scripted("Hel", "lo"))
        self.assertTrue(session.send("hi"))
        events = wait_for(session, "done")
        self.assertEqual([e["type"] for e in events], ["user", "token", "token", "done"])
        self.assertEqual(events[0]["text"], "hi")
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "Hello")
        self.assertFalse(session.busy)

    def test_event_ids_increase(self):
        session = ChatSession(app=None, run=scripted("a"))
        session.send("x")
        events = wait_for(session, "done")
        ids = [e["id"] for e in events]
        self.assertEqual(ids, sorted(set(ids)))

    def test_events_after_filters(self):
        session = ChatSession(app=None, run=scripted("a"))
        session.send("x")
        events = wait_for(session, "done")
        # events are: user, token, done; asking after the token leaves only done
        later = session.events_after(events[1]["id"], timeout=0)
        self.assertEqual([e["type"] for e in later], ["done"])

    def test_blank_message_is_refused(self):
        session = ChatSession(app=None, run=scripted("a"))
        self.assertFalse(session.send("   "))
        self.assertEqual(session.events_after(0, timeout=0), [])

    def test_second_send_while_busy_is_refused(self):
        gate = threading.Event()

        def slow(app, text, config, on_text, approve, cancel=None, **kwargs):
            gate.wait(3)
            return False

        session = ChatSession(app=None, run=slow)
        self.assertTrue(session.send("one"))
        self.assertTrue(session.busy)
        self.assertFalse(session.send("two"))
        gate.set()
        wait_for(session, "done")
        self.assertTrue(session.send("three"))
        wait_for(session, "done", after=0)

    def test_error_becomes_event_and_session_stays_usable(self):
        def boom(app, text, config, on_text, approve, cancel=None, **kwargs):
            raise RuntimeError("model down")

        session = ChatSession(app=None, run=boom)
        session.send("x")
        events = wait_for(session, "error")
        error = next(e for e in events if e["type"] == "error")
        self.assertIn("model down", error["text"])
        self.assertFalse(session.busy)
        self.assertTrue(session.send("again"))


class ApprovalTests(unittest.TestCase):
    def make(self, results):
        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            results.append(approve({"calls": [{"name": "write_file", "args": {"path": "a.txt"}}]}))
            on_text("after")
            return False

        return ChatSession(app=None, run=run)

    def test_approval_event_then_allow(self):
        results = []
        session = self.make(results)
        session.send("go")
        events = wait_for(session, "approval")
        approval = next(e for e in events if e["type"] == "approval")
        self.assertEqual(approval["calls"][0]["name"], "write_file")
        pending = session.pending_approval
        assert pending is not None
        self.assertEqual(pending["calls"][0]["name"], "write_file")
        self.assertTrue(session.approve(True))
        wait_for(session, "done")
        self.assertEqual(results, [True])
        self.assertIsNone(session.pending_approval)

    def test_deny(self):
        results = []
        session = self.make(results)
        session.send("go")
        wait_for(session, "approval")
        session.approve(False)
        wait_for(session, "done")
        self.assertEqual(results, [False])

    def test_approve_without_pending_request_is_refused(self):
        session = ChatSession(app=None, run=scripted("a"))
        self.assertFalse(session.approve(True))

    def test_stop_while_waiting_for_approval_cancels(self):
        results = []
        session = self.make(results)
        session.send("go")
        wait_for(session, "approval")
        session.stop()
        events = wait_for(session, "cancelled")
        self.assertEqual(results, [])
        self.assertNotIn("done", [e["type"] for e in events])
        self.assertFalse(session.busy)


class StopAndNewChatTests(unittest.TestCase):
    def test_stop_cancels_a_running_turn(self):
        started = threading.Event()

        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            assert cancel is not None
            started.set()
            while not cancel.is_set():
                cancel.wait(0.05)
            raise TurnCancelled

        session = ChatSession(app=None, run=run)
        session.send("long task")
        self.assertTrue(started.wait(3))
        session.stop()
        wait_for(session, "cancelled")
        self.assertFalse(session.busy)
        self.assertTrue(session.send("next"))  # a stop must not cancel the next message

    def test_stop_when_idle_does_nothing(self):
        session = ChatSession(app=None, run=scripted("a"))
        session.stop()
        self.assertEqual(session.events_after(0, timeout=0), [])

    def test_new_chat_changes_thread_and_clears_history(self):
        threads = []

        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            threads.append(config["configurable"]["thread_id"])
            return False

        session = ChatSession(app=None, run=run)
        session.send("a")
        wait_for(session, "done")
        self.assertTrue(session.new_chat())
        events = session.events_after(0, timeout=0)
        self.assertEqual([e["type"] for e in events], ["reset"])
        session.send("b")
        wait_for(session, "done", after=events[-1]["id"])
        self.assertEqual(len(set(threads)), 2)

    def test_new_chat_refused_while_busy(self):
        gate = threading.Event()

        def slow(app, text, config, on_text, approve, cancel=None, **kwargs):
            gate.wait(3)
            return False

        session = ChatSession(app=None, run=slow)
        session.send("a")
        self.assertFalse(session.new_chat())
        gate.set()
        wait_for(session, "done")

    def test_run_receives_recursion_limit(self):
        seen = []

        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            seen.append(config["recursion_limit"])
            return False

        session = ChatSession(app=None, run=run)
        session.send("a")
        wait_for(session, "done")
        self.assertEqual(seen, [100])


def created(manager, **kwargs):
    """manager.create() for tests that expect room for a new session."""
    session = manager.create(**kwargs)
    assert session is not None, "session limit reached"
    return session


def blocking():
    """A run that waits for its gate; returns (run, started, gate)."""
    started = threading.Event()
    gate = threading.Event()

    def run(app, text, config, on_text, approve, cancel=None, **kwargs):
        started.set()
        while not gate.is_set():
            if cancel is not None and cancel.is_set():
                raise TurnCancelled
            gate.wait(0.02)
        on_text("ok")
        return False

    return run, started, gate


class IdentityTests(unittest.TestCase):
    def test_id_and_thread_are_distinct_per_session(self):
        a = ChatSession(app=None, run=scripted("a"))
        b = ChatSession(app=None, run=scripted("a"))
        self.assertNotEqual(a.id, b.id)
        self.assertRegex(a.id, r"^[0-9a-f]{12}$")
        self.assertNotEqual(a.thread_id, b.thread_id)

    def test_title_comes_from_first_message(self):
        session = ChatSession(app=None, run=scripted("a"))
        self.assertEqual(session.title, "New chat")
        session.send("  Explain   the graph\nplease ")
        wait_for(session, "done")
        self.assertEqual(session.title, "Explain the graph please")
        session.send("second message")
        wait_for(session, "done", after=3)
        self.assertEqual(session.title, "Explain the graph please")

    def test_long_first_message_title_is_shortened(self):
        session = ChatSession(app=None, run=scripted("a"))
        session.send("x" * 200)
        wait_for(session, "done")
        self.assertLessEqual(len(session.title), 60)
        self.assertTrue(session.title.endswith("…"))

    def test_rename(self):
        session = ChatSession(app=None, run=scripted("a"))
        self.assertTrue(session.rename("  My chat "))
        self.assertEqual(session.title, "My chat")
        self.assertFalse(session.rename("   "))
        session.rename("y" * 200)
        self.assertLessEqual(len(session.title), 60)
        session.send("first")
        wait_for(session, "done")
        self.assertEqual(session.title, "y" * 59 + "…")  # a chosen name is not replaced

    def test_info(self):
        session = ChatSession(app=None, run=scripted("a"))
        info = session.info()
        self.assertEqual(info["id"], session.id)
        self.assertEqual(info["title"], "New chat")
        self.assertEqual(info["state"], "idle")
        self.assertEqual(info["last_event"], 0)
        self.assertIsInstance(info["created"], float)

    def test_info_state_follows_the_turn(self):
        run, started, gate = blocking()
        session = ChatSession(app=None, run=run)
        session.send("go")
        self.assertTrue(started.wait(3))
        self.assertEqual(session.info()["state"], "running")
        gate.set()
        wait_for(session, "done")
        self.assertEqual(session.info()["state"], "idle")

    def test_info_state_approval(self):
        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            approve({"calls": []})
            return False

        session = ChatSession(app=None, run=run)
        session.send("go")
        wait_for(session, "approval")
        self.assertEqual(session.info()["state"], "approval")
        session.approve(True)
        wait_for(session, "done")


class TurnSlotTests(unittest.TestCase):
    def test_second_session_waits_for_the_slot(self):
        slot = threading.Semaphore(1)
        run_a, started_a, gate_a = blocking()
        a = ChatSession(app=None, run=run_a, turn_slot=slot)
        b = ChatSession(app=None, run=scripted("b"), turn_slot=slot)
        a.send("first")
        self.assertTrue(started_a.wait(3))
        b.send("second")
        events = wait_for(b, "waiting")
        self.assertNotIn("done", [e["type"] for e in events])
        self.assertEqual(b.info()["state"], "waiting")
        gate_a.set()
        wait_for(b, "done")
        wait_for(a, "done")

    def test_stop_while_waiting_cancels_without_running(self):
        slot = threading.Semaphore(1)
        run_a, started_a, gate_a = blocking()
        ran = []

        def run_b(app, text, config, on_text, approve, cancel=None, **kwargs):
            ran.append(text)
            return False

        a = ChatSession(app=None, run=run_a, turn_slot=slot)
        b = ChatSession(app=None, run=run_b, turn_slot=slot)
        a.send("first")
        self.assertTrue(started_a.wait(3))
        b.send("second")
        wait_for(b, "waiting")
        b.stop()
        wait_for(b, "cancelled")
        self.assertFalse(b.busy)
        gate_a.set()
        wait_for(a, "done")
        self.assertEqual(ran, [])

    def test_slot_is_released_after_error(self):
        slot = threading.Semaphore(1)

        def boom(app, text, config, on_text, approve, cancel=None, **kwargs):
            raise RuntimeError("x")

        a = ChatSession(app=None, run=boom, turn_slot=slot)
        a.send("x")
        wait_for(a, "error")
        self.assertTrue(slot.acquire(timeout=1))

    def test_free_slot_starts_without_waiting_event(self):
        session = ChatSession(app=None, run=scripted("a"), turn_slot=threading.Semaphore(1))
        session.send("go")
        events = wait_for(session, "done")
        self.assertNotIn("waiting", [e["type"] for e in events])


if __name__ == "__main__":
    unittest.main()
