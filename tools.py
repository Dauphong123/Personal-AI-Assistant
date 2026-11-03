import fnmatch
import os
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from dependency import dependency
import web
from jobs import JobManager

# Every file/directory argument must resolve inside this directory.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent)).resolve()

COMMAND_TIMEOUT = 60  # seconds (git, ripgrep)
FOREGROUND_WAIT = 5  # seconds a terminal command may block the turn before it goes to the background

JOBS = JobManager()
MAX_LISTED_FILES = 500
MAX_READ_LINES = 400

IGNORED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "chroma_db",
}

# Files that usually hold credentials. The agent has no need to read them, and a web page or
# document it reads could try to make it send them out, so read_file and search_code skip them
# (the user can still read one on purpose through the approved terminal tool).
SECRET_FILE_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "id_rsa*", "id_dsa*", "id_ecdsa*",
    "id_ed25519*", ".netrc", "_netrc", ".pgpass", ".npmrc", ".pypirc",
)  # fmt: skip
SAFE_TEMPLATE_PATTERNS = (".env.example", ".env.sample", ".env.template", "*.pub")

# Tools that change files or run commands; the graph asks the user before running them.
NEEDS_APPROVAL = {"terminal", "write_file", "edit_file", "delete_file", "move_file"}


class PathOutsideProjectError(ValueError):
    pass


def is_secret_file(path: Path) -> bool:
    name = path.name.lower()

    if any(fnmatch.fnmatch(name, pattern) for pattern in SAFE_TEMPLATE_PATTERNS):
        return False

    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_FILE_PATTERNS)


def resolve_path(path: str, writable: bool = False) -> Path:
    """Resolve `path` against PROJECT_ROOT and refuse anything outside it.

    With writable=True, the repository's own .git directory is refused as well.
    """
    candidate = Path(path)

    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    candidate = candidate.resolve()

    if not candidate.is_relative_to(PROJECT_ROOT):
        raise PathOutsideProjectError(f"path is outside the project: {path}")

    if writable and ".git" in candidate.relative_to(PROJECT_ROOT).parts:
        raise PathOutsideProjectError(f"refusing to modify .git: {path}")

    return candidate


def _git(args: list[str], repo: str) -> str:
    cwd = resolve_path(repo)

    if not cwd.is_dir():
        return f"ERROR: not a directory: {repo}"

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=COMMAND_TIMEOUT,
        )
    except FileNotFoundError:
        return "ERROR: git is not installed."
    except subprocess.TimeoutExpired:
        return f"ERROR: git timed out after {COMMAND_TIMEOUT}s."

    if result.returncode != 0:
        return f"ERROR: {result.stderr.strip()}"

    return result.stdout


@tool
def list_files(path: str = ".") -> list[str]:
    """Return a list of all files in a directory."""

    root = resolve_path(path)

    if not root.exists():
        return [f"Error: directory does not exist: {path}"]

    if not root.is_dir():
        return [f"Error: path is not a directory: {path}"]

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place so ignored trees (.venv, .git, ...) are never walked.
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for name in filenames:
            files.append(str((Path(dirpath) / name).relative_to(root)))

            if len(files) >= MAX_LISTED_FILES:
                return files

    return files


@tool
def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read a file with line numbers. Optionally restrict it with 1-based start_line/end_line.

    At most 400 lines are returned per call."""

    file_path = resolve_path(path)

    if is_secret_file(file_path):
        return (
            f"ERROR: {path} may contain secrets (passwords, keys, tokens), so it is not read. "
            "Ask the user to share the specific value you need."
        )

    if not file_path.exists():
        return f"Error: file does not exist: {path}"

    if not file_path.is_file():
        return f"Error: path is not a file: {path}"

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return f"Error reading file: {e}"

    if not lines:
        return "(file is empty)"

    start = max((start_line or 1) - 1, 0)
    end = end_line or len(lines)
    selected = lines[start:end]
    truncated = len(selected) > MAX_READ_LINES

    output = "\n".join(
        f"{i}: {line}"
        for i, line in enumerate(selected[:MAX_READ_LINES], start=start + 1)
    )

    if truncated:
        output += (
            f"\n... truncated at {MAX_READ_LINES} lines; the file has {len(lines)} "
            "lines. Use start_line/end_line to read more."
        )

    return output


@tool
def search_code(query: str, directory: str = ".") -> str:
    """Search the repository for a text pattern using ripgrep."""

    root = resolve_path(directory)

    if not root.exists():
        return f"Error: directory does not exist: {directory}"

    if not root.is_dir():
        return f"Error: path is not a directory: {directory}"

    ignore = []
    for name in IGNORED_DIRS:
        ignore += ["--glob", f"!{name}/**"]

    try:
        # -e and -- keep a query or path that starts with "-" from being read as a flag.
        result = subprocess.run(
            ["rg", "-n", "--null", *ignore, "-e", query, "--", str(root)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=COMMAND_TIMEOUT,
        )
    except FileNotFoundError:
        return "Error: ripgrep (rg) is not installed."
    except subprocess.TimeoutExpired:
        return f"Error: search timed out after {COMMAND_TIMEOUT}s."

    if result.returncode == 0:
        # --null puts a NUL after each file name, so the name can be told from the matched text.
        # Matches in credential files are dropped here (rg globs cannot keep .env.example
        # while hiding .env.*).
        kept = []

        for line in result.stdout.splitlines():
            name, separator, rest = line.partition("\0")

            if separator and is_secret_file(Path(name)):
                continue

            kept.append(f"{name}{rest}" if not separator else f"{name}:{rest}")

        if kept:
            return "\n".join(kept) + "\n"

        return f"No matches found for: {query}"

    if result.returncode == 1:
        return f"No matches found for: {query}"

    return f"Error running ripgrep:\n{result.stderr.strip()}"


@tool
def git_diff(repo_path: str = ".") -> str:
    """Return the current git diff of the repository."""

    return _git(["diff"], repo_path) or "No uncommitted changes."


@tool
def terminal(command: str) -> str:
    """Execute a shell command in the project directory and return its output.

    A command that takes longer than a few seconds keeps running in the background
    and a job id is returned; the user is told when it finishes."""

    try:
        job = JOBS.start(command, cwd=PROJECT_ROOT)
    except Exception as e:
        return f"Error starting shell: {e}"

    if JOBS.wait_or_detach(job, FOREGROUND_WAIT):
        return job.report()

    return (
        f"Job {job.id} is still running in the background. Do not wait for it: tell the "
        "user it was started. Its result will arrive in a later message; use "
        "job_status to check it or cancel_job to stop it."
    )


@tool
def job_status(job_id: int | None = None) -> str:
    """Show background jobs started by terminal. Without job_id, list all jobs."""

    if job_id is None:
        jobs = JOBS.all()
        return "\n".join(j.report().splitlines()[0] for j in jobs) or "No jobs."

    job = JOBS.get(job_id)

    if job is None:
        return f"ERROR: no job with id {job_id}"

    return job.report()


@tool
def cancel_job(job_id: int) -> str:
    """Cancel a running background job."""

    return JOBS.cancel(job_id)


@tool
def write_file(path: str, content: str) -> str:
    """Create a file, or overwrite it, with the given content."""

    file_path = resolve_path(path, writable=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")

    return f"Successfully wrote {path}"


@tool
def edit_file(
    path: str,
    old: str,
    new: str,
    occurrence: int | None = None,
) -> str:
    """Replace text in a file. If occurrence is omitted, old text must be unique."""

    file_path = resolve_path(path, writable=True)

    if not file_path.exists():
        return f"ERROR: File does not exist: {path}"

    if not old:
        return "ERROR: The text to replace must not be empty."

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

    file_path = resolve_path(path, writable=True)

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

    source_path = resolve_path(source, writable=True)
    destination_path = resolve_path(destination, writable=True)

    if not source_path.exists():
        return f"ERROR: Source does not exist: {source}"

    if not source_path.is_file():
        return f"ERROR: Source is not a file: {source}"

    if destination_path.exists():
        return f"ERROR: Destination already exists: {destination}"

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))
    except OSError as e:
        return f"ERROR: Failed to move file: {e}"

    return f"SUCCESS: Moved {source} -> {destination}"


@tool
def git_status(dir: str = ".") -> str:
    """Show git status of the directory."""

    return _git(["status", "--short"], dir) or "Working tree clean."


@tool
def git_log(dir: str = ".", limit: int = 10) -> str:
    """Show the most recent commits (one line each) of the repository in dir."""

    limit = max(1, min(int(limit), 100))

    return _git(["log", f"-{limit}", "--oneline", "--decorate"], dir)


@tool
def git_show(dir: str, commit: str) -> str:
    """Show the changed files of one commit."""

    # A leading "-" would be read by git as an option instead of a commit.
    if commit.startswith("-"):
        return f"ERROR: invalid commit: {commit}"

    return _git(["show", "--stat", commit, "--"], dir)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo) for current or external information that the repository
    and the knowledge documents do not have. Returns titles, links and short snippets."""

    return web.search(query, max_results)


@tool
def read_webpage(url: str, focus: str = "", start: int = 0) -> str:
    """Read a web page as plain text. Pass `focus` (what you are looking for) to get only the
    matching passages of a long page; otherwise the page is returned in parts, use `start`
    to continue. Only public web pages can be opened."""

    return web.read(url, focus, start)


rag = dependency["rag"]
search_knowledge = rag.get_search_document_tools()

# Single source of truth: the model is bound to TOOLS and the graph dispatches
# through TOOL_BY_NAME, so a tool can't be bound but missing from dispatch.
TOOLS = [
    list_files,
    read_file,
    search_code,
    terminal,
    job_status,
    cancel_job,
    write_file,
    edit_file,
    delete_file,
    move_file,
    git_status,
    git_diff,
    git_log,
    git_show,
    search_knowledge,
    web_search,
    read_webpage,
]

TOOL_BY_NAME = {t.name: t for t in TOOLS}
