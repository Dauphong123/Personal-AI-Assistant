"""Pyright (installed in .venv, configured by pyrightconfig.json) reports no errors.

PENDING lists files that still carry uncommitted work from before 2026-11-17 (see
docs/PROJECT.md, Open questions). Their type errors are left for whoever commits that work;
take a file off the list once it is committed and clean. Everything else is checked."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYRIGHT = ROOT / ".venv" / "Scripts" / "pyright.exe"
SKIP_DIRS = {".venv", "node_modules", "chroma_db", "__pycache__", ".git", "dist"}

PENDING = {
    "memories/store.py",
    "tests/test_chat_loop.py",
    "tests/test_graph.py",
    "tests/test_hardening.py",
    "tests/test_memory.py",
    "tests/test_speaker.py",
    "tests/test_streaming_speech.py",
    "tests/test_vietnamese.py",
    "tests/test_voice.py",
    "tests/test_voice_engines.py",
    "tests/test_voice_hardening.py",
    "tests/test_voice_parts.py",
}


def python_files() -> list[str]:
    files = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if SKIP_DIRS.intersection(relative.parts):
            continue
        name = relative.as_posix()
        if name not in PENDING:
            files.append(name)
    return sorted(files)


def pyright_executable() -> str | None:
    if PYRIGHT.is_file():
        return str(PYRIGHT)
    return shutil.which("pyright")


@unittest.skipUnless(pyright_executable(), "pyright not installed (pip install pyright)")
class TypeCheckTests(unittest.TestCase):
    def test_config_points_pyright_at_the_venv(self):
        config = json.loads((ROOT / "pyrightconfig.json").read_text(encoding="utf-8"))
        self.assertEqual((config["venvPath"], config["venv"]), (".", ".venv"))

    def test_pending_files_still_exist(self):
        for name in PENDING:
            self.assertTrue((ROOT / name).is_file(), f"{name} is gone: drop it from PENDING")

    def test_no_type_errors(self):
        exe = pyright_executable()
        assert exe is not None
        files = python_files()
        self.assertIn("backend/server.py", files)
        result = subprocess.run(
            [exe, "--outputjson", *files],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
        )
        report = json.loads(result.stdout)
        errors = [
            f"{Path(d['file']).relative_to(ROOT).as_posix()}:{d['range']['start']['line'] + 1}: {d['message'].splitlines()[0]}"
            for d in report["generalDiagnostics"]
            if d["severity"] == "error"
        ]
        self.assertEqual(errors, [], f"{len(errors)} pyright errors:\n" + "\n".join(errors[:40]))


if __name__ == "__main__":
    unittest.main()
