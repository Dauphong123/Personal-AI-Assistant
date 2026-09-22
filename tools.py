import shutil
import os
from langchain_core.tools import tool
from pathlib import Path
import subprocess
import platform


def run_shell(command: str) -> subprocess.CompletedProcess[str]:
    """Run a command using the shell appropriate for the current OS."""

    if os.name == "nt":
        # Prefer PowerShell 7, fall back to Windows PowerShell.
        shell = shutil.which("pwsh") or shutil.which("powershell")

        if shell is None:
            raise RuntimeError("PowerShell is not installed.")

        args = [shell, "-NoProfile", "-Command", command]

    else:
        # Linux / macOS
        args = ["/bin/sh", "-c", command]

    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


@tool
def list_files(path: str = ".") -> list[str]:
    """Return a list of all files in a directory."""

    root = Path(path)

    if not root.exists():
        return [f"Error: directory does not exist: {root}"]

    if not root.is_dir():
        return [f"Error: path is not a directory: {root}"]

    return [str(file.relative_to(root)) for file in root.rglob("*") if file.is_file()]


@tool
def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read a file and optionally restrict the returned line range."""

    file_path = Path(path)

    if not file_path.exists():
        return f"Error: file does not exist: {file_path}"

    if not file_path.is_file():
        return f"Error: path is not a file: {file_path}"

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return f"Error reading file: {e}"

    start = start_line - 1 if start_line else 0
    end = end_line if end_line else len(lines)

    return "\n".join(
        f"{i + 1}: {line}" for i, line in enumerate(lines[start:end], start=start)
    )


@tool
def search_code(query: str, directory: str = ".") -> str:
    """Search the repository for a text pattern using ripgrep."""

    root = Path(directory)

    if not root.exists():
        return f"Error: directory does not exist: {root}"

    if not root.is_dir():
        return f"Error: path is not a directory: {root}"

    # No shell needed here.
    # This works on Windows, Linux, and macOS as long as rg is installed.
    result = subprocess.run(
        ["rg", "-n", query, str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode == 0:
        return result.stdout

    if result.returncode == 1:
        return f"No matches found for: {query}"

    return f"Error running ripgrep:\n{result.stderr.strip()}"


@tool
def git_diff(repo_path: str = ".") -> str:
    """Return the current git diff of the repository."""

    root = Path(repo_path)

    if not root.exists():
        return f"Error: repository path does not exist: {root}"

    if not root.is_dir():
        return f"Error: repository path is not a directory: {root}"

    result = subprocess.run(
        ["git", "diff"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        return f"Error running git diff:\n{result.stderr.strip()}"

    return result.stdout


@tool
def terminal(command: str) -> str:
    """Execute a shell command in the current operating system and return its output."""

    try:
        result = run_shell(command)
    except Exception as e:
        return f"Error starting shell: {e}"

    if result.returncode != 0:
        return (
            f"Command failed (exit code {result.returncode}):\n{result.stderr.strip()}"
        )

    return result.stdout
