import threading
import unittest

from tests import stubs

stubs.install()

from backend.sessions import SessionManager  # noqa: E402
from tests.test_backend_session import blocking, scripted, wait_for  # noqa: E402


class ManagerTests(unittest.TestCase):
    def test_create_get_list(self):
        manager = SessionManager(app=None, run=scripted("a"))
        a = manager.create()
        b = manager.create()
        self.assertIs(manager.get(a.id), a)
        self.assertEqual([s["id"] for s in manager.list()], [a.id, b.id])

    def test_get_unknown_is_none(self):
        manager = SessionManager(app=None, run=scripted("a"))
        self.assertIsNone(manager.get("000000000000"))

    def test_create_with_title(self):
        manager = SessionManager(app=None, run=scripted("a"))
        self.assertEqual(manager.create(title="Plans").title, "Plans")

    def test_sessions_have_separate_conversations(self):
        threads = []

        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            threads.append(config["configurable"]["thread_id"])
            on_text(text.upper())
            return False

        manager = SessionManager(app=None, run=run)
        a, b = manager.create(), manager.create()
        a.send("one")
        wait_for(a, "done")
        b.send("two")
        wait_for(b, "done")
        self.assertEqual(len(set(threads)), 2)
        tokens_a = [e["text"] for e in a.events_after(0) if e["type"] == "token"]
        self.assertEqual(tokens_a, ["ONE"])

    def test_runs_with_the_shared_app(self):
        apps = []

        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            apps.append(app)
            return False

        marker = object()
        manager = SessionManager(app=marker, run=run)
        s = manager.create()
        s.send("x")
        wait_for(s, "done")
        self.assertEqual(apps, [marker])

    def test_delete(self):
        manager = SessionManager(app=None, run=scripted("a"))
        s = manager.create()
        self.assertEqual(manager.delete(s.id), "deleted")
        self.assertIsNone(manager.get(s.id))
        self.assertEqual(manager.list(), [])
        self.assertEqual(manager.delete(s.id), "missing")

    def test_delete_refused_while_busy(self):
        run, started, gate = blocking()
        manager = SessionManager(app=None, run=run)
        s = manager.create()
        s.send("go")
        self.assertTrue(started.wait(3))
        self.assertEqual(manager.delete(s.id), "busy")
        gate.set()
        wait_for(s, "done")
        self.assertEqual(manager.delete(s.id), "deleted")

    def test_session_limit(self):
        manager = SessionManager(app=None, run=scripted("a"), max_sessions=2)
        manager.create()
        manager.create()
        self.assertIsNone(manager.create())

    def test_only_max_turns_run_at_once(self):
        run, started, gate = blocking()
        manager = SessionManager(app=None, run=run, max_turns=1)
        a, b = manager.create(), manager.create()
        a.send("a")
        self.assertTrue(started.wait(3))
        b.send("b")
        wait_for(b, "waiting")
        gate.set()
        wait_for(a, "done")
        wait_for(b, "done")

    def test_two_turns_allowed_in_parallel(self):
        both = threading.Barrier(2, timeout=3)

        def run(app, text, config, on_text, approve, cancel=None, **kwargs):
            both.wait()  # breaks (raises) unless both turns run at the same time
            return False

        manager = SessionManager(app=None, run=run, max_turns=2)
        a, b = manager.create(), manager.create()
        a.send("a")
        b.send("b")
        wait_for(a, "done")
        wait_for(b, "done")

    def test_max_turns_must_be_positive(self):
        with self.assertRaises(ValueError):
            SessionManager(app=None, max_turns=0)

    def test_stop_all(self):
        run, started, gate = blocking()
        manager = SessionManager(app=None, run=run)
        s = manager.create()
        s.send("go")
        self.assertTrue(started.wait(3))
        manager.stop_all()
        wait_for(s, "cancelled")


if __name__ == "__main__":
    unittest.main()
