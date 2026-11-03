"""Background shell jobs: a command that outlives the turn that started it."""

import itertools
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

JOB_TIMEOUT = 600  # seconds before a job is killed
MAX_REPORT_CHARS = 4000
MAX_CAPTURE_CHARS = 200_000  # only the end of a very chatty job's output is kept

RUNNING = "running"
FINISHED = "finished"
CANCELLED = "cancelled"
TIMED_OUT = "timed out"


def shell_args(command: str) -> list[str]:
    if os.name == "nt":
        # Prefer PowerShell 7, fall back to Windows PowerShell.
        shell = shutil.which("pwsh") or shutil.which("powershell")

        if shell is None:
            raise RuntimeError("PowerShell is not installed.")

        return [shell, "-NoProfile", "-Command", command]

    return ["/bin/sh", "-c", command]


def kill_tree(process: subprocess.Popen) -> None:
    """Kill the process and everything it started."""
    if process.poll() is not None:
        return

    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            import signal

            os.killpg(process.pid, signal.SIGKILL)
    except Exception:
        logger.warning("Could not kill process tree", exc_info=True)
        process.kill()


def _collect(stream, sink: list[str]) -> None:
    """Read a pipe to its end, keeping only the last MAX_CAPTURE_CHARS characters."""
    tail = ""

    try:
        while True:
            chunk = stream.read(65536)

            if not chunk:
                break

            tail = (tail + chunk)[-MAX_CAPTURE_CHARS:]
    except (ValueError, OSError):  # the pipe was closed under us
        pass

    sink.append(tail)


@dataclass
class Job:
    id: int
    command: str
    process: subprocess.Popen
    started: float = field(default_factory=time.monotonic)
    status: str = RUNNING
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timeout: int = JOB_TIMEOUT
    detached: bool = False  # the turn that started it has moved on
    done: threading.Event = field(default_factory=threading.Event)

    def report(self) -> str:
        """Describe the job the way the model should see it."""
        if self.status == RUNNING:
            elapsed = int(time.monotonic() - self.started)
            return f"Job {self.id} is still running ({elapsed}s): {self.command}"

        if self.status == CANCELLED:
            return f"Job {self.id} was cancelled: {self.command}"

        if self.status == TIMED_OUT:
            return f"Command timed out after {self.timeout}s."

        if self.returncode == 0:
            text = self.stdout
        else:
            text = (
                f"Command failed (exit code {self.returncode}):\n"
                f"{self.stderr.strip()}\n{self.stdout.strip()}"
            ).strip()

        if len(text) > MAX_REPORT_CHARS:
            text = "... (earlier output omitted)\n" + text[-MAX_REPORT_CHARS:]

        return text


class JobManager:
    def __init__(self, on_finish: Callable[[Job], None] | None = None):
        # Called from a watcher thread for detached jobs that end on their own.
        self.on_finish = on_finish
        self._jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def start(self, command: str, cwd: Path, timeout: int = JOB_TIMEOUT) -> Job:
        kwargs = {} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen(
            shell_args(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
        job = Job(id=next(self._ids), command=command, process=process, timeout=timeout)

        with self._lock:
            self._jobs[job.id] = job

        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        return job

    def _watch(self, job: Job) -> None:
        status = FINISHED
        captured: dict[str, list[str]] = {"out": [], "err": []}
        readers = [
            threading.Thread(target=_collect, args=(job.process.stdout, captured["out"]), daemon=True),
            threading.Thread(target=_collect, args=(job.process.stderr, captured["err"]), daemon=True),
        ]

        for reader in readers:
            reader.start()

        try:
            job.process.wait(timeout=job.timeout)
        except subprocess.TimeoutExpired:
            kill_tree(job.process)
            job.process.wait()
            status = TIMED_OUT
        except Exception:
            logger.exception("Job %s watcher failed", job.id)
            kill_tree(job.process)
            captured["err"] = ["internal error while running the job"]

        for reader in readers:
            reader.join(timeout=5)  # a grandchild may still hold the pipe open

        for pipe in (job.process.stdout, job.process.stderr):
            try:
                pipe.close()
            except OSError:
                pass

        stdout = "".join(captured["out"])
        stderr = "".join(captured["err"])

        with self._lock:
            if job.status == RUNNING:  # not already cancelled
                job.status = status

            job.stdout, job.stderr = stdout or "", stderr or ""
            job.returncode = job.process.returncode
            job.done.set()
            notify = job.detached and job.status != CANCELLED

        if notify and self.on_finish is not None:
            try:
                self.on_finish(job)
            except Exception:
                logger.exception("Job finish callback failed")

    def wait_or_detach(self, job: Job, seconds: float) -> bool:
        """Wait for a job. True if it finished; otherwise it is detached to finish later."""
        if job.done.wait(seconds):
            return True

        with self._lock:
            if job.done.is_set():
                return True

            job.detached = True

        return False

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def close(self) -> None:
        """Kill every job that is still running (they must not outlive the app)."""
        with self._lock:
            running = [job for job in self._jobs.values() if job.status == RUNNING]

            for job in running:
                job.status = CANCELLED

        for job in running:
            kill_tree(job.process)

    def cancel(self, job_id: int) -> str:
        job = self.get(job_id)

        if job is None:
            return f"ERROR: no job with id {job_id}"

        with self._lock:
            if job.status != RUNNING:
                return f"Job {job_id} is not running ({job.status})."

            job.status = CANCELLED

        kill_tree(job.process)
        return f"Cancelled job {job_id}."
