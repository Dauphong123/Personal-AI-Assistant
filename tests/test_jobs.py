import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import jobs
from jobs import JobManager


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        self.cwd = Path(tempfile.gettempdir())
        self.manager = JobManager()
        self.finished = []
        self.event = threading.Event()
        self.manager.on_finish = self._on_finish

    def _on_finish(self, job):
        self.finished.append(job)
        self.event.set()

    def start(self, code, **kwargs):
        with patch.object(jobs, "shell_args", return_value=[sys.executable, "-c", code]):
            job = self.manager.start("cmd", cwd=self.cwd, **kwargs)

        self.addCleanup(self.manager.cancel, job.id)
        return job

    def test_quick_job_finishes_without_notification(self):
        job = self.start("print('ok')")
        self.assertTrue(self.manager.wait_or_detach(job, 10))
        self.assertEqual(job.report().strip(), "ok")
        self.assertEqual(self.finished, [])

    def test_detached_job_notifies_when_it_ends(self):
        job = self.start("import time; time.sleep(0.5); print('late')")
        self.assertFalse(self.manager.wait_or_detach(job, 0.05))
        self.assertTrue(self.event.wait(10))
        self.assertEqual(self.finished, [job])
        self.assertEqual(job.report().strip(), "late")

    def test_cancelled_job_does_not_notify(self):
        job = self.start("import time; time.sleep(30)")
        self.assertFalse(self.manager.wait_or_detach(job, 0.05))
        self.assertIn("Cancelled", self.manager.cancel(job.id))
        self.assertTrue(job.done.wait(10))
        self.assertEqual(job.status, jobs.CANCELLED)
        self.assertFalse(self.event.wait(0.3))

    def test_timeout_kills_job(self):
        job = self.start("import time; time.sleep(30)", timeout=1)
        self.assertTrue(job.done.wait(15))
        self.assertEqual(job.status, jobs.TIMED_OUT)
        self.assertIn("timed out after 1s", job.report())

    def test_cancel_unknown_or_finished_job(self):
        self.assertIn("no job", self.manager.cancel(42))
        job = self.start("pass")
        job.done.wait(10)
        self.assertIn("not running", self.manager.cancel(job.id))

    def test_long_output_is_truncated_to_its_tail(self):
        job = self.start("print('x' * 10000 + 'END')")
        job.done.wait(10)
        report = job.report()
        self.assertLess(len(report), jobs.MAX_REPORT_CHARS + 100)
        self.assertTrue(report.strip().endswith("END"))


if __name__ == "__main__":
    unittest.main()
