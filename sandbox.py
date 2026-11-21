"""The box `terminal` commands run in (F038).

- `child_env`: the command gets the app's environment minus anything that looks like a secret
  (tokens, keys, passwords, database URLs), so a command the model wrote cannot print them.
- `ProcessBox` (Windows): a job object holding the command and everything it starts, with a memory
  cap (`TERMINAL_MEMORY_MB`); `kill()` ends all of them, also a process whose parent already
  exited (which `taskkill /T` cannot find), and closing the app ends them too.

Not a security boundary against a hostile program: the command still runs as the user, in
PROJECT_ROOT, after the user approved it. A process the command starts in the moment before
it joins the box (a few ms after start) is not in it.
"""

import ctypes
import logging
import os
import re
from ctypes import wintypes

logger = logging.getLogger(__name__)

_SECRET_PARTS = frozenset({"TOKEN", "SECRET", "PASSWORD", "PASSWD", "KEY", "APIKEY", "AUTH", "CREDENTIAL", "CREDENTIALS"})
_SECRET_NAMES = frozenset({"DATABASE_URL", "DB_URL", "PGPASSWORD"})


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return upper in _SECRET_NAMES or any(part in _SECRET_PARTS for part in re.split(r"[^A-Z0-9]+", upper))


def child_env(env=None) -> dict[str, str]:
    env = os.environ if env is None else env
    return {name: value for name, value in env.items() if not is_secret_name(name)}


# --- Windows job objects -------------------------------------------------------------------

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("Read", "Write", "Other", "ReadBytes", "WriteBytes", "OtherBytes")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # pyright: ignore[reportAttributeAccessIssue]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel32


class ProcessBox:
    """A Windows job object for one command. On other systems every method does nothing."""

    def __init__(self, pid: int, memory_mb: int):
        self._handle = None

        if os.name != "nt":
            return

        try:
            self._handle = self._create(pid, memory_mb)
        except OSError:
            logger.warning("Could not put the command in a job object; it runs without limits", exc_info=True)

    @staticmethod
    def _create(pid: int, memory_mb: int):
        kernel32 = _kernel32()
        job = kernel32.CreateJobObjectW(None, None)

        if not job:
            raise ctypes.WinError(ctypes.get_last_error())  # pyright: ignore[reportAttributeAccessIssue]

        try:
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            if memory_mb > 0:
                limits.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_JOB_MEMORY
                limits.JobMemoryLimit = memory_mb * 1024 * 1024

            if not kernel32.SetInformationJobObject(
                job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise ctypes.WinError(ctypes.get_last_error())  # pyright: ignore[reportAttributeAccessIssue]

            process = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)

            if not process:
                raise ctypes.WinError(ctypes.get_last_error())  # pyright: ignore[reportAttributeAccessIssue]

            try:
                if not kernel32.AssignProcessToJobObject(job, process):
                    raise ctypes.WinError(ctypes.get_last_error())  # pyright: ignore[reportAttributeAccessIssue]
            finally:
                kernel32.CloseHandle(process)
        except BaseException:
            kernel32.CloseHandle(job)
            raise

        return job

    @property
    def active(self) -> bool:
        return self._handle is not None

    def kill(self) -> None:
        """End every process in the box. The box stays usable (empty) until closed."""
        if self._handle is not None:
            _kernel32().TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        """Close the box; anything still in it ends (kill on close)."""
        handle, self._handle = self._handle, None

        if handle is not None:
            _kernel32().CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """Whether a process with this id is still running (Windows; used by tests and diagnostics)."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    kernel32 = _kernel32()
    process = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)

    if not process:
        return False

    try:
        code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(process, ctypes.byref(code))) and code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(process)
