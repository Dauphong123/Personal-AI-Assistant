import os
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from rag.rag import RAG

IGNORED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
}


def run_shell(command: str) -> subprocess.CompletedProcess[str]:
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

    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if any(part in IGNORED_DIRS for part in p.parts):
            continue

        files.append(str(p.relative_to(root)))

    return files[:500]


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


@tool
def write_file(path: str, content: str) -> str:
    """Write an new file to path"""
    file_path = Path(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    return f"Successful wrote {file_path}"


@tool
def edit_file(
    path: str,
    old: str,
    new: str,
    occurrence: int | None = None,
) -> str:
    """Replace text in a file. If occurrence is omitted, old text must be unique."""

    file_path = Path(path)

    if not file_path.exists():
        return f"ERROR: File does not exist: {path}"

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: Could not decode file as UTF-8: {path}"

    count = content.count(old)

    if count == 0:
        return "ERROR: The specified text was not found."

    # Default: require unique match
    if occurrence is None:
        if count > 1:
            return (
                f"ERROR: The specified text occurs {count} times. "
                "Specify an occurrence number or provide more context."
            )

        occurrence = 1

    # Validate occurrence
    if occurrence < 1 or occurrence > count:
        return f"ERROR: occurrence must be between 1 and {count}, got {occurrence}."

    # Find the requested occurrence
    start = 0

    index = 0
    for _ in range(occurrence):
        index = content.find(old, start)
        start = index + len(old)

    # Replace only that occurrence
    new_content = content[:index] + new + content[index + len(old) :]

    try:
        file_path.write_text(new_content, encoding="utf-8")
    except OSError as e:
        return f"ERROR: Failed to write file: {e}"

    return f"SUCCESS: Replaced occurrence {occurrence} of {count} in {path}."


@tool
def delete_file(path: str) -> str:
    """Delete a file from the project."""

    file_path = Path(path)

    if not file_path.exists():
        return f"ERROR: File does not exist: {path}"

    if not file_path.is_file():
        return f"ERROR: Path is not a file: {path}"

    try:
        file_path.unlink()
    except OSError as e:
        return f"ERROR: Failed to delete {path}: {e}"

    return f"SUCCESS: Deleted {path}"


@tool
def move_file(source: str, destination: str) -> str:
    """Move or rename a file."""

    source_path = Path(source)
    destination_path = Path(destination)

    if not source_path.exists():
        return f"ERROR: Source does not exist: {source}"

    if not source_path.is_file():
        return f"ERROR: Source is not a file: {source}"

    if destination_path.exists():
        return f"ERROR: Destination already exists: {destination}"

    try:
        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.move(
            str(source_path),
            str(destination_path),
        )

    except OSError as e:
        return f"ERROR: Failed to move file: {e}"

    return f"SUCCESS: Moved {source} → {destination}"


@tool
def git_status(dir: str) -> str:
    """show git status of the directory"""
    result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True,
        text=True,
        cwd=dir,
        check=False,
    )

    if result.returncode != 0:
        return f"ERROR: {result.stderr}"

    return result.stdout or "Working tree clean."


@tool
def git_log(dir: str, limit: int = 10) -> str:
    """use git log to show log of git in dir"""
    result = subprocess.run(
        [
            "git",
            "log",
            f"-{limit}",
            "--oneline",
            "--decorate",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=dir,
    )

    if result.returncode != 0:
        return f"ERROR: {result.stderr}"

    return result.stdout


@tool
def git_show(dir: str, commit: str) -> str:
    """show the changes in one commit"""
    result = subprocess.run(
        ["git", "show", "--stat", commit],
        capture_output=True,
        text=True,
        cwd=dir,
        check=False,
    )

    if result.returncode != 0:
        return f"ERROR: {result.stderr}"

    return result.stdout


rag = RAG()
rag.add_document("./documents/")
search_knowledge = rag.get_search_document_tools()
