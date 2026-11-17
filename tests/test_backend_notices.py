"""Background-job completion notices reach the web session that started the job."""

import threading
import time
import unittest
from types import SimpleNamespace

from tests import stubs

stubs.install()

from backend.session import ChatSession  # noqa: E402
from backend.sessions import SessionManager  # noqa: E402
from jobs import JOB_OWNER, NOTICE_PREFIX  # noqa: E402
from tests.test_backend_session import blocking, wait_for  # noqa: E402


def recording():
    """A run that records (text, notice flag, job owner) and answers 'ok'."""
    calls = []

    def run(app, text, config, on_text, approve, cancel=None, notice=False, **kwargs):
        calls.append((text, notice, JOB_OWNER.get()))
        on_text("ok")
        return False

    return run, calls


def finished_job(job_id=3, owner=None, output="all passed"):
    return SimpleNamespace(
        id=job_id, command="pytest", owner=owner, report=lambda: output, status="finished", returncode=0
    )


class OwnerTests(unittest.TestCase):
    def test_turn_runs_with_the_session_as_job_owner(self):
        run, calls = recording()
        session = ChatSession(app=None, run=run)
        session.send("start the tests")
        wait_for(session, "done")
        self.assertEqual(calls, [("start the tests", False, session.id)])
        self.assertIsNone(JOB_OWNER.get())


class NotifyTests(unittest.TestCase):
    def test_idle_session_runs_a_notice_turn(self):
        run, calls = recording()
        session = ChatSession(app=None, run=run)
        session.notify("NOTICE TEXT", "Job 3 finished: pytest")
        events = wait_for(session, "done")
        self.assertEqual([e["type"] for e in events], ["notice", "token", "done"])
        self.assertEqual(events[0]["text"], "Job 3 finished: pytest")
        self.assertEqual(calls, [("NOTICE TEXT", True, session.id)])

    def test_notice_does_not_name_the_session(self):
        run, _ = recording()
        session = ChatSession(app=None, run=run)
        session.notify("NOTICE TEXT", "Job 3 finished")
        wait_for(session, "done")
        self.assertEqual(session.title, "New chat")

    def test_busy_session_runs_the_notice_after_the_turn(self):
        run, started, gate = blocking()
        notices = []

        def both(app, text, config, on_text, approve, cancel=None, notice=False, **kwargs):
            if notice:
                notices.append(text)
                return False
            return run(app, text, config, on_text, approve, cancel)

        session = ChatSession(app=None, run=both)
        session.send("long task")
        self.assertTrue(started.wait(3))
        session.notify("NOTICE", "Job 1 finished")
        self.assertEqual(notices, [])
        gate.set()
        deadline = time.monotonic() + 3
        while len([e for e in session.events_after(0) if e["type"] == "done"]) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        types = [e["type"] for e in session.events_after(0)]
        self.assertEqual(types, ["user", "token", "done", "notice", "done"])
        self.assertEqual(notices, ["NOTICE"])
        self.assertFalse(session.busy)

    def test_session_stays_busy_between_turn_and_queued_notice(self):
        run, started, gate = blocking()
        release_notice = threading.Event()

        def both(app, text, config, on_text, approve, cancel=None, notice=False, **kwargs):
            if notice:
                release_notice.wait(3)
                return False
            return run(app, text, config, on_text, approve, cancel)

        session = ChatSession(app=None, run=both)
        session.send("long task")
        self.assertTrue(started.wait(3))
        session.notify("NOTICE", "Job 1 finished")
        gate.set()
        wait_for(session, "notice")
        self.assertFalse(session.send("user message"))  # the notice turn is running
        release_notice.set()
        wait_for(session, "done", after=3)

    def test_stop_cancels_the_turn_but_keeps_queued_notices(self):
        run, started, gate = blocking()
        notices = []

        def both(app, text, config, on_text, approve, cancel=None, notice=False, **kwargs):
            if notice:
                notices.append(text)
                return False
            return run(app, text, config, on_text, approve, cancel)

        session = ChatSession(app=None, run=both)
        session.send("long task")
        self.assertTrue(started.wait(3))
        session.notify("NOTICE", "Job 1 finished")
        session.stop()
        wait_for(session, "cancelled")
        wait_for(session, "notice")
        self.assertEqual(wait_for(session, "done")[-1]["type"], "done")
        self.assertEqual(notices, ["NOTICE"])

    def test_notices_run_in_order(self):
        run, calls = recording()
        gate = threading.Event()

        def slow(app, text, config, on_text, approve, cancel=None, notice=False, **kwargs):
            gate.wait(3)
            return run(app, text, config, on_text, approve, cancel, notice=notice)

        session = ChatSession(app=None, run=slow)
        session.notify("first", "Job 1")
        session.notify("second", "Job 2")
        gate.set()
        seen = []
        cursor = 0
        while len([e for e in seen if e["type"] == "done"]) < 2:
            new = wait_for(session, "done", after=cursor)
            seen += new
            cursor = new[-1]["id"]
        self.assertEqual([c[0] for c in calls], ["first", "second"])


class ManagerRoutingTests(unittest.TestCase):
    def test_finished_job_goes_to_its_session(self):
        run, calls = recording()
        manager = SessionManager(app=None, run=run)
        a, b = manager.create(), manager.create()
        manager.job_finished(finished_job(owner=b.id))
        events = wait_for(b, "done")
        self.assertEqual(events[0]["type"], "notice")
        self.assertIn("job 3", events[0]["text"].lower())
        self.assertIn("pytest", events[0]["text"])
        self.assertTrue(calls[0][0].startswith(NOTICE_PREFIX))
        self.assertIn("all passed", calls[0][0])
        self.assertTrue(calls[0][1])
        self.assertEqual(a.events_after(0), [])

    def test_job_of_a_deleted_or_unknown_session_is_dropped(self):
        run, calls = recording()
        manager = SessionManager(app=None, run=run)
        s = manager.create()
        manager.delete(s.id)
        manager.job_finished(finished_job(owner=s.id))
        manager.job_finished(finished_job(owner=None))
        self.assertEqual(calls, [])

    def test_job_finished_never_raises(self):
        manager = SessionManager(app=None, run=recording()[0])
        broken = SimpleNamespace(id=1, command="x", owner="nope")  # no report()
        manager.job_finished(broken)


if __name__ == "__main__":
    unittest.main()
