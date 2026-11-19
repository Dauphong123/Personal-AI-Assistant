# Project guide: read this project from zero

Who this is for: you know basic Python (variables, `if`, `for`, functions, lists, dicts) and want to understand this project's real code. Every code snippet below is copied from the repository, so you can open the file and see it in context. The second half (Reference A and B) explains the same system as concepts and in more detail.

How to use it: read Part 1 once, then Parts 2–9 in order, run the commands in Part 10, then use Reference A and B (after Part 11) as the concept reference.

---

## Part 1. Python you will meet in this code

### 1.1 Imports
```python
import os
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
```
- `import os` loads a whole module; you write `os.environ`.
- `from X import Y` loads just `Y`; you write `Y` directly.
- Modules that start with a name in this repo (`from dependency import dependency`, `from tools import TOOLS`) are our own files. Others (`langchain_core`, `pathlib`) are libraries or the standard library.
- `from .node import ...` (with a dot) means "from the file `node.py` next to me" (used in `graph/graph.py`).

### 1.2 Functions, defaults and type hints
```python
def list_files(path: str = ".") -> list[str]:
    """Return a list of all files in a directory."""
```
- `path: str` says "path should be text". `= "."` is the default if you pass nothing.
- `-> list[str]` says "returns a list of texts".
- `int | None` means "a number or nothing" (`None` is Python's "no value"). Example: `start_line: int | None = None`.
- Python does **not** enforce hints when running. This project uses a tool called **Pyright** to check them, and the LLM tool system also reads them (see 1.8).
- The text in triple quotes right under `def` is a **docstring**. Here it matters a lot: the model reads it as the tool's description.

### 1.3 The four containers
```python
NEEDS_APPROVAL = {"terminal", "write_file", "edit_file", "delete_file", "move_file"}   # set
TOOLS = [list_files, read_file, search_code]                                           # list
TOOL_BY_NAME = {t.name: t for t in TOOLS}                                              # dict
SECRET_FILE_PATTERNS = (".env", "*.pem", "*.key")                                      # tuple
```
- **list** `[ ]`: ordered, changeable. **tuple** `( )`: ordered, fixed. **set** `{ }`: no duplicates, fast "is it in here?" (`name in NEEDS_APPROVAL`). **dict** `{k: v}`: look up a value by key (`TOOL_BY_NAME["read_file"]`).
- `{t.name: t for t in TOOLS}` is a **dict comprehension**: "for every tool `t`, make an entry name → tool".

### 1.4 Comprehensions and generator expressions
```python
dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
```
Read it as: "build a new list of every `d` in `dirnames` that is not ignored". The `[:]` on the left replaces the contents **in place** so `os.walk` stops entering `.venv`, `.git` and so on.
```python
any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_FILE_PATTERNS)
```
No square brackets: a **generator expression**. `any()` stops at the first `True`.

### 1.5 f-strings
```python
return f"ERROR: not a directory: {repo}"
```
`f"..."` inserts variable values where you write `{ }`. Used everywhere for messages.

### 1.6 Classes and objects
```python
class PathOutsideProjectError(ValueError):
    pass
```
A **class** is a blueprint. This one is a custom error type: it inherits (`(ValueError)`) everything a `ValueError` can do and only adds a new name, so code can say "catch this kind of error". Bigger classes in the repo (`ChatLoop`, `JobManager`, `MemoryStore`) hold data in `self.something` and have **methods** (functions inside a class).

### 1.7 Exceptions: try / except
```python
try:
    output = str(tool.invoke(args))
except Exception as e:
    return f"ERROR: {type(e).__name__}: {e}"
```
If the `try` part fails, Python jumps to `except` instead of crashing. This project uses it on purpose: a failing tool must never crash the chat. The error text goes back to the model, which can try again. `raise SomeError("message")` is how you cause one.

### 1.8 Decorators
```python
@tool
def read_file(path: str, ...) -> str:
```
A **decorator** (the `@name` line) wraps the function below it. Two matter here:
- `@tool` (from LangChain) turns a plain function into a tool object the model can call. It reads the name, parameter names/types and docstring to build the description the model sees.
- `@timed` (in `graph/node.py`) wraps a node to log how long it took:
```python
def timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timing_logger.info("%s %.2fs", fn.__name__, time.perf_counter() - start)
    return wrapper
```
`*args, **kwargs` means "accept any arguments and pass them on". `finally` runs whether the function succeeded or failed.

### 1.9 `with`, files and paths
```python
lines = file_path.read_text(encoding="utf-8").splitlines()
```
`pathlib.Path` is the modern way to handle file paths: `Path("a") / "b"` joins paths, `.exists()`, `.is_dir()`, `.read_text()`, `.resolve()` (make absolute, remove `..`). `with something:` is used when a resource must be closed/finished properly (database sessions, locks).

### 1.10 Slicing and `enumerate`
```python
selected = lines[start:end]
for i, line in enumerate(selected, start=start + 1):
```
`lines[2:5]` is items 2, 3, 4 (end not included). `enumerate` gives (counter, item) pairs. This is how `read_file` prints `12: some code`.

### 1.11 Regular expressions (`re`)
```python
_FILE_NAME = re.compile(r"[\w\-/\\]\w*\.(py|ts|json|md)\b", re.IGNORECASE)
```
A pattern that matches text shapes. This one finds things that look like `main.py` or `src/app.ts` (the real one lists more extensions). `pattern.search(text)` returns a match or `None`. The router uses these to decide quickly (Part 5).

### 1.12 Environment variables and `.env`
```python
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent)).resolve()
```
`os.environ.get("X", default)` reads a setting from the environment, with a fallback. `load_dotenv()` fills the environment from the `.env` file. `__file__` is the path of the current Python file.

### 1.13 `if __name__ == "__main__":`
At the bottom of `main.py`. It means "run `main()` only when this file is started directly (`python main.py`), not when another file imports it".

### 1.14 Threads and queues
```python
threading.Thread(target=keyboard, daemon=True).start()
self._tasks: queue.Queue[tuple[str, bool] | None] = queue.Queue()
```
A **thread** runs a function at the same time as the rest of the program. A **queue** is a safe waiting line between threads: one puts items in, another takes them out in order. `daemon=True` means the thread will not keep the program alive at exit. This is how you can type while the model is still answering.

### 1.15 TypedDict and Annotated
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    memory_context: str
    route: str
```
A `TypedDict` is a plain dict with named, typed keys. `Annotated[list, add_messages]` attaches an extra instruction to the type: "when two updates arrive for `messages`, combine them with `add_messages`" (append) instead of replacing.

### 1.16 The `|` pipe in LangChain
```python
code_chain = code_prompt | code_agent
```
Not "bitwise or" here. LangChain overloads `|` to mean "send the output of the left into the right": the prompt template fills in the messages, then the model reads them. Calling `.invoke({...})` runs the whole pipe.

### 1.17 Tests
`tests/` uses `unittest`: each `class ...Tests(unittest.TestCase)` has methods named `test_...` that call `self.assertEqual(...)`. Run them all with `python -m unittest discover -s tests -t .`.

---

## Part 2. The map of the project

Top-level files (from `ls`):

| File / folder | What it is |
|---|---|
| `main.py` | Start of the terminal app. Reads what you type. |
| `chat_loop.py` | `ChatLoop`: queue of messages, worker thread, approval questions, stop handling. |
| `turns.py` | `run_turn`: runs one turn of the graph and streams the output; `SpeechStream` for voice. |
| `dependency.py` | Builds the model, RAG, tokenizer and memory once, at import. |
| `graph/graph.py` | The graph wiring (which node follows which). |
| `graph/node.py` | The node functions (the real logic). |
| `data/state.py` | The `State` type passed through the graph. `data/route_query.py` has `Router`. |
| `chains/*.py` | Prompt + model pipelines: `code_chain`, `direct_chain`, `router_chain`, `summerize_chain`. |
| `tools.py` | The tools the model can call (files, git, terminal, web, documents). |
| `tool_args.py` | Repairs/refuses wrong tool arguments before they run. |
| `jobs.py` | Background commands (`JobManager`). |
| `rag/rag.py` | Document search (Chroma + keyword search). |
| `retrieval.py` | Keyword (BM25) scoring and rank merging shared by RAG and memory. |
| `web.py` | `web_search` and `read_webpage` with safety checks. |
| `memories/` | Long-term memory in Postgres (see Part 8). |
| `voice/` | Optional voice mode. |
| `backend/` and `webui/` | Browser version: Python API and TypeScript page. |
| `settings.py`, `llm_options.py`, `log_setup.py`, `language.py` | Config checks, model options, logging, Vietnamese/English helpers. |
| `documents/` | The files the RAG tool searches. |
| `tests/` | All automated tests. |
| `docs/`, `CLAUDE.md` | Documentation and project rules. |
| `compose.yaml` | Starts the Postgres database in Docker. |
| `requirements*.txt`, `constraints.txt` | Libraries to install, and their locked versions. |

---

## Part 3. What happens when you start `python main.py`

Open [main.py](../main.py). In order:

1. **Imports.** Importing `chat_loop`, `graph` and `tools` triggers `dependency.py` (Part 4) which loads the model client, indexes documents, loads the tokenizer and connects to memory. This is why starting takes a while.
2. **`argparse`.** Reads `--voice` from the command line.
3. **Encoding fix.** `sys.stdout.reconfigure(encoding="utf-8", ...)`: Windows consoles cannot print emoji or Vietnamese by default.
4. **`configure_logging()`** sets up log output; a bad `LOG_LEVEL` exits with a message.
5. **Voice config** is only loaded if requested.
6. **Compile the graph with a checkpointer:**
   ```python
   app = ModelGraph.compile(checkpointer=InMemorySaver())
   config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
   ```
   *Compile* turns the graph description into something runnable. The checkpointer stores the conversation (in memory only). `thread_id` names this conversation. `recursion_limit` stops a runaway loop after 100 steps.
7. **`chat = ChatLoop(app, config, JOBS)`** and `chat.start()` start the worker thread.
8. **`read_keyboard(chat)`** loops forever:
   ```python
   line = input("User: ")
   if not chat.submit(line):
       return "exit"
   ```
   It only reads lines and hands them to `chat.submit`. `submit` returns `False` for `exit`/`quit`. `EOFError` (input ended, e.g. a pipe) and Ctrl+C (`KeyboardInterrupt`) are handled.
9. **`finally:` cleanup** always runs: `chat.close()`, `JOBS.close()` (kills background commands), `wait_for_memory()` (lets a memory save finish).

Why a worker thread? `input()` blocks. If the model ran on the same thread you could not type "stop" while it answers. So `main.py` only reads the keyboard; `ChatLoop` runs the turns.

---

## Part 4. `dependency.py`: the shared toolbox

```python
load_dotenv()
_problems = check_settings()
if _problems:
    raise SystemExit(settings_error(_problems))
...
def dependency_init():
    model = model_init()
    rag = rag_init()
    tokenizer = tokenizer_init()
    memory = memory_init(model)
    return {"model": model, "rag": rag, "tokenizer": tokenizer, "memory": memory}

dependency = dependency_init()
```
- It is a plain **dict** with four things: `model`, `rag`, `tokenizer`, `memory`.
- The last line runs **when the file is imported**, so any file doing `from dependency import dependency` gets the ready-made objects. One shared copy for the whole program: this pattern is called a **singleton**.
- `model_init` creates `ChatOllama(model=MODEL_NAME, **ollama_options())`. `**dict` unpacks a dict into keyword arguments.
- `rag_init` makes the RAG object and calls `add_document("./documents/")`, which indexes your files.
- `tokenizer_init` loads a tokenizer only to **count tokens** (so the code can tell when history is too long).
- `memory_init` wraps everything in `try/except`. If Postgres is not running it logs a warning and returns `None`. Code elsewhere checks `if manager is not None`, so the app works without memory.
- Rule: `dependency.py` must never import `graph`, or Python would find a circular import (A needs B, B needs A).

---

## Part 5. The graph: `state.py`, `graph.py`, `node.py`

### 5.1 State
`data/state.py` (15 lines): the dict that moves through the graph.
- `messages`: the whole conversation (human, AI and tool messages). Uses `add_messages` so returning `{"messages": [new]}` **adds** to the list.
- `memory_context`: text recalled from long-term memory for this turn.
- `route`: `"code_tools"` or `"direct"`.

### 5.2 Wiring (`graph/graph.py`)
```python
graph.add_edge(START, "recall_memory")
graph.add_edge(START, "route")
graph.add_edge(["recall_memory", "route"], "routed")
graph.add_conditional_edges("routed", chosen_route, {"code_tools": "code_tools", "direct": "direct"})
graph.add_conditional_edges("code_tools", has_tools, {"tool_execute": "tool_execute", "continue": "save_memory"})
graph.add_edge("tool_execute", "code_tools")
graph.add_edge("direct", "save_memory")
graph.add_conditional_edges("save_memory", summarize_need, {"summarize": "summarize", "continue": END})
graph.add_edge("summarize", END)
```
As a picture:
```
START ─┬─> recall_memory ─┐
       └─> route ─────────┴─> routed ─┬─(direct)─────> direct ──────────────┐
                                      └─(code_tools)─> code_tools ─┐        │
                                                          ▲        │ has tool calls?
                                                          └─ tool_execute <┘ yes
                                                                   │ no
                                                                   ▼
                                                              save_memory ─┬─(too long)─> summarize ─> END
                                                                           └─(ok)───────> END
```
(`direct` also flows into `save_memory`.)

- A **node** (`add_node`) is a function. An **edge** says what runs next.
- A **conditional edge** calls a small function (`chosen_route`, `has_tools`, `summarize_need`) that returns a **label**; the dict maps label → next node.
- Two edges from `START` mean `recall_memory` and `route` run **at the same time**. `routed` is a do-nothing join point that waits for both (`def routed(state): return {}`).
- The loop `code_tools → tool_execute → code_tools` repeats until the model answers without asking for a tool.

### 5.3 How a node works
A node takes `state` and returns a **dict of changes**:
```python
def code_tools(state: State):
    response = code_chain.invoke({"messages": _model_messages(state)})
    _log_reply("code_tools", response)
    return {"messages": [response]}
```
It does not modify `state` directly. It returns `{"messages": [response]}` and LangGraph merges it using `add_messages`. Returning `{}` means "no change".

### 5.4 Routing (`route`, `quick_route`, `router`)
```python
def quick_route(text):
    text = normalize(text).strip()
    bare = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if bare in _SMALL_TALK:
        return "direct"
    for pattern in (_FILE_NAME, _COMMANDS, _FILE_ACTIONS, _VIETNAMESE_ACTIONS):
        if pattern.search(text):
            return "code_tools"
    return None
```
- Remove punctuation and lowercase; if it is "hi", "thanks", "xin chào"… → `direct`.
- If it contains a file name (`main.py`), a command (`git status`, `pytest`), a file action plus file word ("read the file"), or a Vietnamese action phrase → `code_tools`.
- `None` means "I cannot tell". Then `router()` asks the model: `router_chain.invoke(...)` returns a `Router` object whose `destination` is `"code"` or something else. The router prompt also gets the last 4 messages so "yes, do it" is understood.
- Why rules first? A model call costs seconds on this machine; a regex costs microseconds.

### 5.5 Recalling memory
```python
def recall_memory(state):
    manager = dependency.get("memory")
    context = ""
    last = state["messages"][-1]
    if manager is not None and not last.additional_kwargs.get("notice"):
        try:
            context = manager.recall(*memory_query(state["messages"]))
        except Exception:
            logger.warning("Memory recall failed", exc_info=True)
    return {"memory_context": context}
```
- `dependency.get("memory")` is `None` if memory is off.
- Notices (job-finished messages) are skipped.
- A failure is logged and the turn goes on with an empty `context`: memory is a bonus, never a requirement.
- `*memory_query(...)` unpacks the two strings it returns into two arguments.

### 5.6 Building what the model sees (`_model_messages`)
```python
def _model_messages(state):
    memory = state.get("memory_context", "")
    budget = MESSAGE_BUDGET - (count_message_tokens([HumanMessage(content=memory)]) if memory else 0)
    return with_memory(fit_to_context(state["messages"], budget), memory)
```
1. `MESSAGE_BUDGET = 8192 − 1400 − 1500 = 5292` tokens for messages (context window minus the tool-description prompt minus room for the answer).
2. `fit_to_context` counts tokens; if too many, it shortens **old tool results** (keeps the first 300 characters + a note), newest last. The stored state is **not** changed, only what is sent.
3. `with_memory` inserts the memory text as an extra **user** message just before the newest user message. Reason (from the docstring): Ollama re-reads a prompt from the first changed token; putting changing text at the top (system prompt) made it re-read everything.

### 5.7 The code node and its prompt (`chains/code_chain.py`)
```python
code_prompt = ChatPromptTemplate.from_messages([("system", """You are a coding assistant ..."""), MessagesPlaceholder("messages")])
code_agent = model.bind_tools(TOOLS)
code_chain = code_prompt | code_agent
```
- `("system", "...")`: fixed rules (use `list_files` to explore, read before editing, never claim success without verifying…).
- `MessagesPlaceholder("messages")`: "insert the conversation here".
- `model.bind_tools(TOOLS)`: tells the model which tools exist (names, parameters, docstrings).
- `|` joins prompt → model.

### 5.8 Do we need a tool? (`has_tools`)
```python
def has_tools(state):
    if state["messages"][-1].tool_calls:
        return "tool_execute"
    return "continue"
```
The last message is an `AIMessage`; if it has `tool_calls`, the model wants tools run.

### 5.9 Running tools (`tool_execute`), step by step
1. `calls = state["messages"][-1].tool_calls`. Each call is a dict like `{"name": "read_file", "args": {...}, "id": "..."}`.
2. Count how many tool results already exist in this turn (`used`); the limit is 30.
3. `_prepare(call)` for each call: looks the tool up in `TOOL_BY_NAME`; unknown name → an `ERROR` text; otherwise `tool_args.prepare` repairs the arguments (6.3) or returns an error.
4. `risky` = calls whose name is in `NEEDS_APPROVAL` and that are not already refused.
5. **If any risky:** `approved = bool(interrupt({"calls": [...]}))`. The graph **stops here** and hands the list to whoever is running it (the CLI asks y/N). Their answer is sent back as `Command(resume=True/False)` and `interrupt(...)` returns it. Important comment in the code: on resume this node runs again from its top, so nothing with side effects may happen before the `interrupt`.
6. For each call decide the output text: over the limit → error; refused → its error; needs approval but declined → `"ERROR: the user declined this action."`; else `_run_tool(call)`.
7. `_run_tool` does `tool.invoke(args)` inside `try/except`, and cuts output longer than 20,000 characters.
8. Every result becomes `ToolMessage(content=output, tool_call_id=call["id"])`. The id ties a result to its call; the model needs that pairing.
9. Return `{"messages": messages}`; the graph goes back to `code_tools`.

### 5.10 The direct node
Same as `code_tools` but with `direct_chain` (no tools), so one model call produces the answer.

### 5.11 Ending the turn: `save_memory`, `summarize_need`, `summarize`
- `save_memory` does **not** save immediately: it puts the conversation on `MEMORY_SCHEDULER`'s queue; saving happens when you have been idle a few seconds.
- `summarize_need` counts tokens of the whole history; over `MAX_HISTORY_TOKENS` (5000) → `"summarize"`.
- `summarize` finds the index of the last human message (`_last_human_index`), summarizes everything before it with `summerize_chain`, and returns:
  ```python
  [RemoveMessage(id=m.id) for m in messages]
  + [AIMessage(content=f"Summary of the earlier conversation:\n{summary}")]
  + [m.model_copy(update={"id": str(uuid4())}) for m in recent]
  ```
  i.e. delete all messages, add the summary, re-add the recent ones with new ids (new messages are appended after removals, so this keeps the order). The latest turn stays whole so no tool call is separated from its result.

---

## Part 6. Tools in detail

### 6.1 Anatomy of a tool: `list_files`
```python
@tool
def list_files(path: str = ".") -> list[str]:
    """Return a list of all files in a directory."""
    root = resolve_path(path)
    if not root.exists():
        return [f"Error: directory does not exist: {path}"]
    ...
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in filenames:
            files.append(str((Path(dirpath) / name).relative_to(root)))
            if len(files) >= MAX_LISTED_FILES:
                return files
    return files
```
- First line is always `resolve_path` (the safety gate, 6.2).
- Problems are returned as normal text starting with `Error:`, not raised. The model reads them like any result.
- `os.walk` visits folders; ignored ones are pruned; at most 500 files are returned so one call cannot flood the context.

### 6.2 The safety gate: `resolve_path`
```python
candidate = Path(path)
...
if not candidate.is_absolute():
    candidate = PROJECT_ROOT / candidate
candidate = candidate.resolve()
if not candidate.is_relative_to(PROJECT_ROOT):
    raise PathOutsideProjectError(f"path is outside the project: {path}")
if writable and ".git" in candidate.relative_to(PROJECT_ROOT).parts:
    raise PathOutsideProjectError(f"refusing to modify .git: {path}")
return candidate
```
- Relative paths are placed under `PROJECT_ROOT`.
- `.resolve()` removes `..` tricks (`../../secret`), then `is_relative_to` checks the final real path is still inside the project.
- A path like `/voice` (slash, no drive) is treated as `PROJECT_ROOT/voice`, because the model often writes it this way and on Windows it would otherwise mean the drive root.
- Writing is also refused inside `.git`.
- Secret files (`.env`, `*.pem`, `id_rsa*`…) are refused by `is_secret_file` in `read_file`/`search_code`; `.env.example` is allowed (`SAFE_TEMPLATE_PATTERNS`).

### 6.3 Argument repair: `tool_args.py`
The small model makes sloppy calls: `file_path` instead of `path`, or JSON where text is expected. `prepare(tool, args)` fixes what is safe to fix (rename a synonym when exactly one parameter fits, turn a dict back into JSON text) and **refuses** the rest with the tool's real signature so the model can correct itself. It runs before the approval question, so you are never asked to approve a call that would fail anyway.

### 6.4 The full tool list (`TOOLS` in `tools.py`)
`list_files`, `read_file`, `search_code`, `terminal`, `job_status`, `cancel_job`, `write_file`, `edit_file`, `delete_file`, `move_file`, `git_status`, `git_diff`, `git_log`, `git_show`, `search_document` (the variable is called `search_knowledge`), `web_search`, `read_webpage`.

Needs your approval: `terminal`, `write_file`, `edit_file`, `delete_file`, `move_file` (`NEEDS_APPROVAL`). Read-only tools run directly.

`TOOLS` is the **single source of truth**: `bind_tools(TOOLS)` tells the model about them and `TOOL_BY_NAME = {t.name: t for t in TOOLS}` is used to run them, so a tool cannot be known to the model but missing from the runner.

### 6.5 The terminal tool and background jobs
`terminal` calls `JOBS.start(command, cwd=PROJECT_ROOT)`, then waits up to 5 seconds. Finished → returns the report. Still running → returns "Job N is still running in the background…" and the chat continues. When the job ends, a `[system notice]` message is queued as a new turn so the model can tell you. `job_status` and `cancel_job` let the model look at or stop a job. Jobs are killed after 10 minutes or when the app exits.

### 6.6 Adding your own tool (practice recipe)
1. In `tools.py` write `@tool def my_tool(arg: str) -> str:` with a clear docstring.
2. If it takes a file path, call `resolve_path` first.
3. Add it to `TOOLS` (bottom of the file).
4. Add a bullet in the prompt in `chains/code_chain.py` (a test checks every tool named there exists).
5. If it changes anything, add its name to `NEEDS_APPROVAL`.
6. Add a test in `tests/`.

---

## Part 7. One turn through the program (code level)

Typing `read main.py` in the terminal:

1. `read_keyboard` → `chat.submit("read main.py")` puts it on the queue.
2. The worker thread takes it and calls `run_turn` (`turns.py`), which streams the graph.
3. `recall_memory` and `route` start together. `quick_route("read main.py")` matches `_FILE_NAME` → `"code_tools"`.
4. `routed` joins; `chosen_route` returns `"code_tools"`.
5. `code_tools` builds messages, calls the model through `code_chain`. The model answers with a tool call `read_file(path="main.py")`.
6. `has_tools` → `"tool_execute"`. `read_file` is not in `NEEDS_APPROVAL`, so no question. The file text (numbered lines, max 400) becomes a `ToolMessage`.
7. Back to `code_tools`; the model now has the file, writes an explanation, no tool call → `has_tools` returns `"continue"`.
8. `save_memory` queues the turn; `summarize_need` checks size; the turn ends.
9. Tokens printed to the screen as they stream.

Typing `delete old.txt` (a risky tool): same until step 6, then `interrupt(...)` pauses; `ChatLoop` prints the question; you type `y` or `n`; it resumes with `Command(resume=True/False)`.

---

## Part 8. Memory and documents in plain steps

### 8.1 Documents (RAG)
1. At startup `rag.add_document("./documents/")` reads files, cuts them into passages and stores each as an **embedding** (a list of numbers describing meaning) in Chroma (`chroma_db/`). It is safe to run every start: passage ids are stable, and unchanged ones are skipped; deleted files' passages are removed.
2. The model calls `search_document(query)` when it needs your notes.
3. Search = vector similarity **plus** keyword scoring (BM25), merged with `reciprocal_rank_fusion` in `retrieval.py`; results too far in meaning (distance above 0.9) are dropped; at most 2 passages per file.

### 8.2 Long-term memory
Three tables in Postgres (`memories/table.py`): *episodes* (what happened), *facts* (things about you), *procedures* (how-tos).

**Saving** (after a turn, when idle):
1. `MemoryExtractor` asks the model, at temperature 0, to list facts in the turn using fixed categories.
2. Turns that are only "hi/thanks" are skipped. Anything that looks like a secret/key is dropped (`looks_secret`).
3. For each new fact, the store looks for a very similar old fact and **overwrites** it (so "I use Vim" replaces "I use VS Code").
4. Everything is written in one database transaction (`save_turn`): all or nothing.

**Recalling** (before the turn): profile facts + facts closest in meaning/keywords to your message + a few hits using your previous question.

**If the DB is down:** memory is `None` and everything else works.

---

## Part 9. Reading order for the rest

When you are comfortable with Parts 1–8, continue with:
1. `chat_loop.py` (queue, worker, approvals, stop).
2. `turns.py` (`run_turn` and streaming).
3. `jobs.py`.
4. `rag/rag.py` and `retrieval.py`.
5. `memories/manager.py`, then `store.py`.
6. `backend/` and `webui/` (web version), `voice/` (speech).
Each folder has a `CLAUDE.md` with the design rules.

---

## Part 10. Try it yourself

Run from the project folder in PowerShell (activate the venv first).

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -t .
```
Runs every test. Slow, but shows what the project promises.

```powershell
python -m unittest tests.test_tools -v
```
Runs just the tool tests; open `tests/test_tools.py` beside it and read how a tool is tested.

```powershell
$env:LOG_TIMING = "true"; python main.py
```
Then type `list the files in the project`. Watch the timing lines: which nodes ran and how long the model took.

Small exercises:
1. Change `MAX_LISTED_FILES` in `tools.py` to 10 and ask for the file list. Why does the list stop? Put it back.
2. Add `"yo"` to `_SMALL_TALK` in `graph/node.py`. Type `yo`; check the timing log says `router: rule -> direct`.
3. Add a tiny read-only tool `count_lines(path)`: follow the recipe in 6.6.
4. Find where `"ERROR: the user declined this action."` is created, then decline a `write_file` request and watch what the model says next.

---

## Part 11. Common beginner mistakes in this codebase
- **Importing a module loads models.** Importing `tools`, `graph` or `chains` in a new script is slow and needs Ollama. Tests avoid this with stubs (`tests/stubs.py`).
- **Editing the wrong "state".** A node must return changes, not mutate `state`.
- **Raising inside a tool.** Prefer returning an `Error:` string. `tool_execute` catches exceptions anyway, but a clear message helps the model.
- **Putting side effects before `interrupt()`.** The node re-runs from the top after resume, so the action would happen twice.
- **Forgetting `resolve_path`.** Every file argument must go through it.
- **Big prompts.** The window is 8192 tokens and the tool descriptions already use about 1,400. Long additions to the system prompt or tool docstrings cost every single call.
- **Assuming memory exists.** Always handle `dependency.get("memory") is None`.
- **Windows paths.** `/x` without a drive, backslashes and drive letters all show up; use `pathlib`, not string slicing.

---

## Reference A. Basic building blocks

Each block says what it is in plain words, then where this project uses it.

### A1. The language model (LLM)
- **What:** a program that reads text and predicts the next piece of text. It has no memory of its own: it only sees what you send in that one call.
- **Token:** the unit it reads and writes, roughly ¾ of an English word. Everything is counted in tokens: input, output, history.
- **Context window:** the maximum tokens it can see per call. Here it is 8192. Prompt + history + tool descriptions + the answer must all fit. If not, Ollama silently drops the oldest messages.
- **Message roles:** `system` (rules), `user` (the person), `assistant` (the model), `tool` (a tool's result). A conversation is a list of these.
- **Temperature:** randomness. 0 = repeat the same answer; higher = more varied. Memory extraction uses 0 so facts do not come and go.
- **Streaming:** receiving the answer token by token instead of waiting for the end. The CLI, web and speech all use it.
- **Thinking mode:** qwen3 can "think" before answering. It is turned off here (`reasoning=False`) because routing took 113 s with it and 3 s without.
- **Speed:** on this GPU `qwen3:8b` writes about 4–6 tokens/s, so a long answer is slow. This is why prompts are short and the code avoids extra model calls.

### A2. Ollama
- **What:** a local server that runs the model on your machine. The app calls it over HTTP on `localhost`.
- **Here:** `llm_options.py` builds every `ChatOllama` client: `keep_alive=30m` (keep the model loaded), optional `LLM_NUM_CTX`.
- Only one model fits in the 4 GB GPU memory at a time. Loading another unloads the first (about 8 s to reload), so every model call uses the same model.
- **Prompt reuse:** Ollama reuses work if the start of the prompt is unchanged. So the code keeps the prompt start stable (this decides where memory is placed, see C4).

### A3. LangChain
- **What:** a library of building blocks for model apps: message classes, model clients, the `@tool` decorator, structured output.
- **Message classes:** `HumanMessage`, `AIMessage`, `ToolMessage`, `SystemMessage` mirror the roles in A1. An `AIMessage` can carry `tool_calls`.
- **`@tool`:** turns a Python function into something the model can call. The function name, parameters and docstring become the description the model reads. A better docstring means better tool use.
- **Structured output:** forces the model to answer in a fixed shape (a pydantic class). The memory extractor uses it to get facts with fixed categories.
- **Chains** (`chains/*`): a prompt plus a model, wired together. There is a code chain, a chat chain, a router chain and a summarize chain (file name `summerize_chain`, a known typo).

### A4. Tool calling and agents
- **Idea:** the model cannot touch your disk. It can only output text, so it outputs a request such as "call `read_file` with path `x.py`". Your code runs it and sends the result back.
- **The loop:** model → `tool_calls` → code runs tools → `ToolMessage` results → model again → … until the model answers with no tool call.
- **Agent:** a model running this loop. Here that is the `code_tools ⇄ tool_execute` cycle.
- **Weakness of a small model:** `qwen3:8b` sometimes sends wrong argument names or JSON where text is expected. `tool_args.py` repairs or refuses these before anything runs (see section 4).

### A5. LangGraph
- **What:** a way to write the app as a graph instead of a tangle of `if` statements.
- **State:** one shared object passed through the graph. Here: `messages`, `memory_context`, `route`.
- **Node:** a Python function that reads state and returns changes (`route`, `direct`, `code_tools`, `tool_execute`, `summarize`, `recall_memory`, `save_memory`).
- **Edge:** "after A, run B". A **conditional edge** picks the next node from state (`routed → direct | code_tools`). Two edges from `START` run in parallel (`recall_memory ∥ route`).
- **Reducer:** says how to merge updates. `messages` uses `add_messages`, which appends rather than replaces.
- **Checkpointer + thread id:** saves state per conversation. `main.py` uses an in-memory one with thread `"1"`; the web backend gives every session its own thread. Restart = history gone.
- **`interrupt()`:** pauses the graph mid-run to ask a human. The caller later resumes with `Command(resume=True/False)`. This is how tool approval works.

### A6. Embeddings and vector search
- **Embedding:** a list of numbers (here 384) that represents the meaning of a text. Texts with similar meaning get nearby lists.
- **Distance:** how far apart two embeddings are (cosine distance). Small = related. The cutoffs in the code (`MAX_VECTOR_DISTANCE` 0.9, memory recall 0.20, …) were measured, not guessed.
- **Vector store:** a database that finds the nearest embeddings fast. RAG uses **Chroma**; memory uses **Postgres + pgvector** with an **HNSW** index (an approximate nearest-neighbour structure).
- **Two models, on purpose:** `bge-small-en-v1.5` for documents (English only), `multilingual-e5-small` for memory (handles Vietnamese). They are not interchangeable: changing one means re-embedding everything.
- **Chunking:** documents are cut into passages before embedding, so a search returns a relevant paragraph, not a whole file.

### A7. Keyword search, hybrid search and RAG
- **BM25:** classic keyword scoring. It is good at exact words, names and numbers where embeddings are weak.
- **Hybrid:** run vector search and BM25, then merge the two ranked lists with **reciprocal rank fusion** (an item ranked high in both wins). `retrieval.py` holds this.
- **RAG (retrieval-augmented generation):** before answering, look up relevant passages and give them to the model. Here RAG is not automatic: it is the `search_document` **tool**, so the model chooses when to search.

### A8. Long-term memory, in simple terms
- **Why:** history vanishes on restart and is trimmed when long. Memory keeps facts across sessions.
- **Three kinds:** *episodes* (what happened in a turn), *facts* (e.g. "the user's editor is VS Code"), *procedures* (how to do something).
- **Write path:** after a turn, a model call extracts facts; secrets are filtered out; contradicted old facts are replaced.
- **Read path:** before a turn, relevant facts are found and shown to the model.
- **Not blocking:** saving happens when you are idle, so it never slows an answer.

### A9. Threads, queues and shared state
- **Thread:** code running in parallel. The CLI reads the keyboard on one thread and runs the agent on another so you can type while it works.
- **Queue:** a waiting line. Messages typed mid-turn wait their turn; memory saves wait for idle time.
- **Semaphore:** limits how many things run at once (web: one turn at a time by default).
- **Contextvar:** a value that follows the current task. A background job remembers which web session started it, so its "finished" notice goes to the right chat.
- **Cancellation:** stopping a turn closes the HTTP request to Ollama, which stops generation within about a second.

### A10. Web basics used here
- **HTTP API:** the browser calls routes like `/api/handshake` and `/api/sessions`.
- **SSE (server-sent events):** a long-open response where the server pushes events. The page receives tokens and approvals this way.
- **Event log:** each session numbers its events (1, 2, 3 …). A reconnecting page asks "everything after N", so nothing is lost or doubled.
- **Handshake and protocol number:** backend and page must agree on a version; if not, the page refuses to run.
- **Token and loopback:** the server listens only on `127.0.0.1` and needs a secret token, so other programs and websites cannot drive it.
- **CSP:** a browser rule that forbids inline scripts. Together with "text nodes only, never `innerHTML`" it stops model or web text from running code in the page.
- **SSRF guard:** `read_webpage` refuses private addresses so the model cannot be tricked into fetching internal services.

### A11. Voice basics
- **STT (speech to text):** faster-whisper turns audio into text.
- **TTS (text to speech):** Piper reads answers aloud, one sentence at a time while later ones are still being written.
- **VAD (voice activity detection):** Silero finds where speech starts and ends, so recording stops at a pause.
- **Wake word:** an always-listening small model that waits for "hey jarvis".
- **Push-to-talk:** hold Right Ctrl to record; **hands-free:** double tap to keep listening and send each phrase.
- **Dictation cleanup:** simple rules remove "um", repeats and "scratch that", because asking the model to do it would cost seconds.

### A12. Setup basics
- **`.env`:** your settings (model, database URL, flags). Copy `.env.example`. It is gitignored. Bad values stop the app at startup with a list of problems.
- **`.venv`:** the isolated Python environment holding the dependencies.
- **Docker Compose:** starts the Postgres + pgvector database on host port 15432.
- **Tests:** `unittest` for Python, `node --test` for the web page; Pyright checks types. CI runs all of it on every push.

### A13. One request, step by step
You type: *"list the files in src and show main.py"*.

1. `ChatLoop` puts the message on a queue and starts a turn on the worker thread.
2. The graph starts two nodes at once. `recall_memory` fetches profile facts and facts close in meaning to your message. `route` runs `quick_route` (`graph/node.py`): pure small talk ("hi") → `direct`; a file name, a command or a file-action word (or its Vietnamese equivalent) → `code_tools`; both without calling a model. Only an unclear message goes to the router model.
3. (Nothing happens here for history: summarizing runs at the end of the turn, see step 9.)
4. `code_tools` builds the prompt: rules + tool descriptions + history + a user-role memory message + your message. It calls the model.
5. The model replies with a tool call: `list_files(path="src")`. `tool_args` checks the arguments; `resolve_path` confirms `src` is inside the project; the tool is read-only, so no approval is needed.
6. `tool_execute` runs it and appends a `ToolMessage`. Control goes back to `code_tools`; the model asks for `read_file(path="main.py")` and gets the lines.
7. The model writes the final answer with no tool call. Tokens stream to the screen (and, in voice mode, sentence by sentence to the speaker).
8. `save_memory` queues the turn. After a few idle seconds the extractor stores any new facts. If you send another message first, the save is cancelled and retried later.

9. `summarize_need` counts the tokens of the whole history. If it is over `MAX_HISTORY_TOKENS`, `summarize` replaces everything before this turn with a short summary; otherwise the turn ends.

If the request had been *"edit main.py"*, step 5 would pause at `interrupt()`: you answer y/N (or click Allow/Deny, or say yes), and only then does the file change.

---

## Reference B. The system in detail

## 1. What the project is

- A **local LLM agent** built with **LangChain + LangGraph** on **Ollama** (default model `qwen3:8b`).
- You chat with it in a terminal (`python main.py`) or in a browser (`python -m backend`).
- Every message is **routed** to one of two paths: a plain **chat answer**, or a **coding agent** that can use filesystem, shell and git tools.
- Extras: **RAG** over `./documents/` (Chroma), **long-term memory** (Postgres + pgvector), optional **voice** (wake word, push-to-talk, hands-free).
- Everything runs on the user's machine (Windows, Python 3.12, `.venv`). No cloud model.

## 2. Request flow (the graph)

Defined in `graph/graph.py` and `graph/node.py`; state is `data/state.py` = `{messages, memory_context, route}`.

1. `START` fans out to two nodes in parallel: `recall_memory` and `route`.
2. `route` decides `direct` or `code_tools`. It uses `quick_route` rules for obvious messages and falls back to the router model (which sees the last few messages).
3. `direct`: plain answer from the chat model.
4. `code_tools` ⇄ `tool_execute`: the model asks for tools, `tool_execute` runs them, repeat while the AI message has `tool_calls`.
5. Both paths go on to `save_memory`, then `summarize_need` decides whether `summarize` runs before the end.

Supporting rules:

- **History limit.** Over `MAX_HISTORY_TOKENS` (default 5000) the `summarize` node condenses everything before the latest turn. The latest turn stays whole, so a tool call is never split from its result.
- **`fit_to_context`** trims older tool results before each model call (budget = context − 1,400 − 1,500 tokens).
- **Checkpointer.** `main.py` uses `InMemorySaver` (thread id `"1"`). History is lost on exit.
- **Streaming.** `turns.py` streams chunks from `direct`, `code_tools` and `tool_execute`.

## 3. Import-time singleton (`dependency.py`)

- On import it builds the model, RAG, tokenizer and memory manager: `dependency = dependency_init()`.
- `chains/*`, `tools.py` and `graph/node.py` use `from dependency import dependency`.
- Consequence: importing almost any module loads models and syncs `./documents/` into Chroma (idempotent: stable chunk ids, stale chunks deleted).
- `dependency.py` must **not** import `graph` (circular). `main.py` compiles the graph.
- Right after `load_dotenv` it runs `settings.check_settings` and exits listing every bad value.

## 4. Tools and safety

Tools live in `tools.py`; `TOOLS` at the bottom drives binding and dispatch.

- **Project sandbox.** Every file argument goes through `resolve_path()`: it rejects paths outside `PROJECT_ROOT`, refuses to write into `.git`, and treats `/x` (no drive) as `PROJECT_ROOT/x`.
- **Secret files.** `read_file` and `search_code` refuse credential files (`SECRET_FILE_PATTERNS`).
- **Approval.** Tools in `NEEDS_APPROVAL` (terminal, write, edit, delete, move) pause with LangGraph `interrupt()`. The CLI asks y/N (voice asks aloud, the web shows Allow/Deny) and resumes with `Command(resume=bool)`.
- **`tool_execute` never raises.** Unknown tools and errors become `ERROR:` tool messages; output is capped; at most 30 tool calls per turn.
- **Argument repair (`tool_args.prepare`).** Before approval or execution: a dict/list passed for a `str` parameter becomes JSON text again; a synonym (`file_path`, `dir`, `old_string`, …) is renamed when exactly one parameter matches; unknown/invalid arguments are refused with the tool's signature. Reason: langchain_ollama parses JSON-looking strings, and qwen3:8b often invents argument names.
- **Adding a tool:** `@tool` + add to `TOOLS`; one line in the code prompt (`chains/code_chain.py`, a test checks the names exist); `NEEDS_APPROVAL` if it changes state; reuse parameter names or extend `tool_args.SYNONYMS`.

## 5. Jobs and live input

- `main.py` only reads the keyboard. `chat_loop.py` (`ChatLoop`) runs turns one at a time on a worker thread.
- Messages typed mid-turn are **queued**. A short stop message (`is_stop_request`: "stop", `/stop`, Vietnamese phrases) cancels the turn and drops the queue.
- `/jobs` and `/cancel <id>` work at any time.
- **`terminal` runs through `jobs.py` (`JobManager`).** Past `FOREGROUND_WAIT` (5 s) the command goes to the background and the tool returns a job id. On completion a `[system notice]` turn is queued. Jobs are killed after 10 minutes; output is a bounded tail.
- Tools `job_status` and `cancel_job` let the model check or stop a job.
- Jobs carry an `owner` (`jobs.JOB_OWNER`, a contextvar) so the web backend routes the notice to the right session.
- A cancelled turn's unanswered tool calls receive `ERROR: cancelled` before the next turn.

## 6. Streaming speech (`turns.SpeechStream`)

- Turns streamed answers into sentences for text-to-speech (nodes `direct` and `code_tools` only).
- Tool-calling messages are muted; `code_tools` holds its newest sentence until the next starts (the tool call arrives after the narration).
- Code blocks become "(code omitted)"; speech stops at `speech_limit`.
- If something streamed, `ChatLoop` skips `on_reply` so nothing is spoken twice.

## 7. Search and RAG

- **`retrieval.py`**: shared BM25 (stop words, common-word filter, accent folding), weighted reciprocal rank fusion, passage splitting.
- **`RAG.search_hybrid`** fuses vector and BM25 hits, drops vector hits beyond `MAX_VECTOR_DISTANCE` (0.9), takes at most 2 passages per file, and returns nothing for unrelated queries.
- RAG embeddings: `BAAI/bge-small-en-v1.5` (English only), so the code prompt tells the model to search documents in English.
- **Web (`web.py`)**: `web_search` (ddgs, cached) and `read_webpage`. The reader is SSRF-guarded: public addresses only, redirects re-checked, DNS pinned to checked addresses, size/time/type limits. Output is labelled untrusted. `WEB_SEARCH_ENABLED=false` disables both.

## 8. Long-term memory (`memories/`)

Optional; if Postgres is unreachable the app runs without it (a warning is logged).

- **Three tables** (`table.py`): episodic, semantic (facts), procedural; 384-dim vectors, HNSW cosine indexes.
- **Embeddings**: `intfloat/multilingual-e5-small` (CPU, `query: ` prefix). A model change re-embeds all rows at startup.
- **Saving (`manager.remember`)**: an extractor LLM (temperature 0, fixed fact categories) pulls facts from a turn. Greetings are skipped, secrets are dropped (`looks_secret`), superseded facts are found and replaced. Everything is written in one transaction via `store.save_turn` (all or nothing).
- **Recalling (`manager.recall`)**: always-on profile facts (cap 12) + hybrid fact search (vector + BM25) + a few follow-up hits using the previous question and last answer.
- **Scheduling (`scheduler.py`)**: saves run only after `MEMORY_IDLE_SECONDS` without a turn. A new turn cancels a running save (the Ollama request is closed) and it retries. `wait_for_memory()` runs the queue at exit.
- **Placement**: recalled memory goes in a **user-role message before the latest one** (`with_memory`), never a system message, to keep Ollama's prompt cache valid.
- **Job notices** are never recalled for or learned from.
- **Quality gate**: `tests/test_memory_eval.py` (40/40 with real embeddings and DB).
- **Connection issues**: `python -m memories.check` diagnoses them. Docker publishes Postgres on host port **15432** because a native Windows Postgres owns 5432. `pool_pre_ping` survives DB restarts.

## 9. Speed on this machine

RTX 3050 4 GB: `qwen3:8b` runs about 4–6 tokens/s, so output length dominates cost.

- `reasoning=False` on every `ChatOllama` (thinking made routing take 113 s instead of 3 s).
- `keep_alive=30m`; optional `LLM_NUM_CTX`.
- `MAX_HISTORY_TOKENS` must stay below the 8192 context minus the ~1,400-token tool prompt and the answer, or Ollama silently drops the oldest messages.
- Only one model fits in memory at a time, so summaries use the main model.
- Ollama reuses work only for an unchanged prompt start; its qwen3 template hoists system messages to the top. This is why memory is not a system message.
- `LOG_TIMING=true` logs node durations and token counts.

## 10. Vietnamese support

- `language.py` (`detect_language`, `say`, NFC `normalize`) is shared by `chat_loop.py`, `turns.py` and `voice/`.
- Prompts say to answer in the user's language.
- RAG embeddings are English-only; memory embeddings are multilingual.
- Voice: `SPEECH_LANGUAGE` can be `vi,en`. Whisper's detector leans English on short Vietnamese, so `voice/stt.py` uses the main language unless the other one scores ≥ 0.8, and compares both decodings in the 0.3–0.8 range.

## 11. Voice mode (`voice/`)

Optional: `python main.py --voice` or `VOICE_ENABLED=true`.

- Voice is just another input/output for `ChatLoop`: transcripts go to `chat.submit()`, replies return through `chat.on_reply`.
- **State machine** in `assistant.py` (`VoiceState`, explicit `TRANSITIONS`); components are injected, so tests use fakes.
- **Parts**: `wakeword.py` (openWakeWord, `hey_jarvis`), `vad.py` (Silero VAD), `stt.py` (faster-whisper), `tts.py` (Piper), `audio.py` (sounddevice), `hotkey.py`, `dictation.py`, `speaker.py`.
- **Triggers** (`VOICE_TRIGGER`): wake word, push-to-talk (default key Right Ctrl, read with `GetAsyncKeyState`), or both.
- **Push-to-talk**: hold to record, release to transcribe; a tap does nothing; a double tap enters **hands-free**.
- **Hands-free**: listens in a loop, each pause ends a prompt, queued behind the running turn. Ends on a tap, "stop listening", or 120 s of silence.
- **Dictation cleanup** is rule-based (fillers, stutters, "scratch that"), because an LLM pass would cost seconds.
- **Speaker verification** (optional): a voiceprint from `voice.enroll`; a convenience lock, not authentication.
- Safety behaviors: the wake word is checked while the agent works ("Jarvis, stop") but never while speaking; the mic queue is flushed after each playback; tool approvals are spoken and answered by voice (silence declines).

## 12. Web chat backend (`backend/`)

`python -m backend`: a stdlib `ThreadingHTTPServer` over the same graph, plus the built frontend.

- **Handshake** (`POST /api/handshake`): the first call. `PROTOCOL` must match `webui/src/types.ts` (currently 2); a mismatch returns 409 ("rebuild"). It reports features (sessions, notices, voice) and limits.
- **Sessions** (`session.py`, `sessions.py`): each `ChatSession` has its own LangGraph thread and an id-numbered **event log** (`user`, `token`, `approval`, `done`, `error`, …). Clients resume with `events_after(id)`. Max 50 sessions, in memory only. One shared semaphore limits concurrent turns (default 1).
- **Approvals** carry a request id; `/approve` must name it, so a stale click or a second tab cannot approve a later tool call.
- **Job notices** are routed to their owning session via the contextvar.
- **Web voice**: `POST /api/voice/transcribe` (audio → PyAV → faster-whisper) and `/api/voice/speak` (text → Piper → WAV). On by default when voice packages are installed; `--no-voice` turns it off. No wake word on the web.
- **Security**: loopback only unless `--allow-remote`; token compared with `hmac.compare_digest`; Host/Origin checks; JSON-only POST (64 KB); static files only from the build dir; CSP forbids inline script/style; no access log (URLs carry the token); `/api/health` needs no token.

## 13. Web frontend (`webui/`)

TypeScript, no framework, no runtime dependencies; built with `tsc` only.

- **`state.ts`**: a pure reducer `apply(conversation, event)` that ignores already-seen ids, so replays are harmless.
- **`markdown.ts` + `render.ts`**: own markdown parser; DOM built from text nodes only. Never `innerHTML` or `eval` (a test scans for them), because model and web text is untrusted.
- **`main.ts`**: DOM wiring, sidebar polling every 2 s, one `EventSource` per active session.
- **Voice in the browser**: `Recorder`, `Player`, `HandsFreeListener`, `Endpointer` and `PressTracker` mirror the Python side (hold Talk or Right Ctrl, double tap for hands-free).
- Theme: system/light/dark via `data-theme`, kept in `localStorage`. No web fonts.

## 14. Reliability and hardening

- Job notices reach the model as a flagged, fenced message and are never learned from.
- `ChatLoop.close()` cancels work; `JobManager.close()` kills running jobs.
- Backend: 500 on a crash, 30 s socket timeout, SIGTERM/Ctrl+Break stop cleanly.
- Logging: `LOG_LEVEL` for the console, `LOG_FILE` = rotating file at INFO.
- Pyright checks every file, tests included.

## 15. Testing and CI

- `python -m unittest discover -s tests -t .` (Postgres tests skip if the DB is unreachable); `npm --prefix webui test` for the frontend.
- Opt-in evals: `RUN_MEMORY_EVAL=1`, `RUN_MODEL_EVAL=1`, `RUN_VOICE_ENGINE_TESTS=1`.
- CI (`.github/workflows/ci.yml`): a `python` job (Windows, locked by `constraints.txt`, Pyright, ripgrep, built `webui/`) and a `webui` job (check/test/build). `main` is protected: both must pass, linear history, no force push.

## 16. How we work on it

- Improve what exists; do not rebuild. Inspect real source before proposing anything.
- One feature per branch `feature/F0xx`, recorded in `feature-lists.json` as `open` before planning.
- Roles are skills in `.claude/skills/` (`architect`, `implement`, `test`, `debug`, `docs-sync`); see `docs/ROSTER.md`.
- `docs/PROJECT.md` holds decisions, verified findings and open questions, tagged **verified / assumed / unknown**.

## 17. Quick glossary

| Term | Meaning |
|---|---|
| Route | The decision `direct` vs `code_tools` for a message |
| Turn | One user message and everything the agent does until it answers |
| Job | A background `terminal` command that outlived `FOREGROUND_WAIT` |
| Notice | A `[system notice]` turn announcing a finished job |
| Episode / Fact / Procedure | The three kinds of long-term memory |
| Hands-free | Voice loop that turns each spoken pause into a prompt |
| Protocol | The integer both backend and frontend must agree on |
| Event log | The numbered stream of events a web session records and replays |
