# Local LLM Agent (LangChain + LangGraph + Ollama)

A local, private AI assistant for your terminal, built on **LangChain** and **LangGraph** and running on **Ollama** (default model `qwen3:8b`). Your conversations and files stay on your machine: the only thing that goes out is a web search the agent runs, if you leave web search on (see [Web search](#web-search)).

Each message you send is routed either to a plain chat answer or to a **coding agent** that can read, search, edit and run things in a project directory. The agent can use a **RAG** knowledge base (Chroma) and a **long-term memory** (Postgres + pgvector) that remembers facts, events and procedures across sessions. You can talk to it by keyboard or, optionally, by **voice** with a wake word, speech recognition, speech output and speaker verification (it only obeys your voice).

> Developed and tested on **Windows 11, Python 3.12**. The code has POSIX branches (shell, process-group kill) but they are not tested.

---

## Contents

1. [Features](#features)
2. [Quick start](#quick-start)
3. [Using it](#using-it)
4. [Project layout](#project-layout)
5. [How a message flows](#how-a-message-flows)
6. [Tools and safety](#tools-and-safety)
7. [Background jobs](#background-jobs)
8. [Knowledge base (RAG)](#knowledge-base-rag)
9. [Web search](#web-search)
10. [Long-term memory](#long-term-memory)
11. [Conversation history and summaries](#conversation-history-and-summaries)
12. [Voice mode](#voice-mode)
13. [Configuration reference](#configuration-reference)
14. [Limits and defaults](#limits-and-defaults)
15. [Vietnamese](#vietnamese-tiếng-việt)
16. [Performance](#performance)
17. [Testing](#testing)
18. [Troubleshooting](#troubleshooting)
19. [Design notes and gotchas](#design-notes-and-gotchas)
20. [Extending the project](#extending-the-project)
21. [Known limitations](#known-limitations)
22. [License](#license)

---

## Features

- **Router**: a small structured-output call decides `direct` (general knowledge) or `code` (needs the machine or repository). It sees the last few messages, so "yes, do it" is routed correctly.
- **Coding agent** with 17 tools: list/read/search files, git status/diff/log/show, run shell commands, write/edit/delete/move files, search the knowledge base, search the web and read web pages, manage background jobs.
- **Human approval** before anything that changes state or runs a command (the graph pauses with `interrupt()`).
- **Path sandbox**: file tools cannot leave `PROJECT_ROOT`, and `.git` cannot be modified.
- **Live input**: you can keep typing while the agent works. Messages queue; a short "stop" message cancels the running turn immediately.
- **Background jobs**: a shell command that runs longer than 5 seconds continues in the background, and the agent is told when it finishes.
- **RAG** over `./documents/` (Chroma, BAAI/bge-small-en-v1.5 embeddings), synced idempotently at startup.
- **Long-term memory** in Postgres/pgvector: episodic, semantic (facts) and procedural memories, recalled before each turn and extracted after it.
- **History summarization** when the conversation grows past 5000 tokens.
- **Voice mode** (optional): wake word, voice activity detection, local speech-to-text, local text-to-speech, spoken tool approvals, speaker verification, all offline.
- **Vietnamese and English**: type or speak Vietnamese; the assistant answers in the language you used, in text and in a Vietnamese voice (see [Vietnamese](#vietnamese-tiếng-việt)).

---

## Quick start

### Prerequisites

| Need | Why |
|---|---|
| Python 3.12 | Runtime |
| [Ollama](https://ollama.com) running | Models |
| `ollama pull qwen3:8b` | Main model |
| `ollama pull qwen3:4b` | Summaries (override with `SUMMARY_MODEL`) |
| [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg` on PATH) | The `search_code` tool |
| Docker (optional) | Postgres for long-term memory |
| A microphone and speakers (optional) | Voice mode |

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# optional: voice mode
python -m pip install -r requirements_voice.txt
```

### Configure

```powershell
copy .env.example .env     # .env is gitignored
```

At minimum, `.env` needs `MODEL=qwen3:8b`. `MODEL` must be a key of `TOKENIZER_MAP` in `dependency.py` (currently only `qwen3:8b`, mapped to the `Qwen/Qwen3-8B` tokenizer). To use another Ollama model, add its tokenizer to that map. See the [configuration reference](#configuration-reference) for every variable.

### Optional: long-term memory database

```powershell
docker compose up -d       # pgvector/pgvector:pg17, db agent_memory, user agent
```

`compose.yaml` creates a container named `agent-postgres` with the credentials `agent` / `agent_password` on port 5432. Those are **local development defaults**; change them if the port is reachable from anywhere else. Tables and the `vector` extension are created automatically on first start.

Without Postgres the app still runs. A warning is logged and memory is simply off.

### Run

```powershell
python main.py              # keyboard chat; type exit or quit to leave
python main.py --voice      # keyboard + wake-word voice mode
```

**First run downloads models**: the Qwen3-8B tokenizer and bge-small embeddings (Hugging Face). Voice mode additionally downloads about 250 MB (see [Voice mode](#voice-mode)). Importing almost any module loads the models and syncs `./documents/`, so startup takes a moment.

---

## Using it

```
User: how is the router implemented?          <- code agent: reads files, answers
User: what is a docker volume?                <- direct answer (or uses the knowledge base)
User: run the tests                           <- asks approval, then runs a shell command
```

### Approvals

Before `terminal`, `write_file`, `edit_file`, `delete_file` or `move_file` run, the assistant prints what it wants to do and asks:

```
The assistant wants to run:
  terminal({"command": "python -m unittest"})
Allow? [y/N]:
```

`y` or `yes` approves. Anything else declines, and the agent is told the user declined. Nothing runs before approval.

### While the agent is working

You can keep typing. The prompt stays live.

| You type | Effect |
|---|---|
| A normal message | Queued; it runs after the current task (`[Queued ...]` is shown) |
| `stop`, `cancel`, `cancel that`, `never mind`, `forget it`, `abort`, `halt`, `/stop` (short variants like "okay stop now") | Cancels the running turn **immediately** and drops queued messages |
| `/jobs` | Lists background jobs |
| `/cancel <id>` | Cancels one background job |
| `exit` / `quit` | Leaves the program |

Only short, unambiguous messages count as "stop". A request that merely contains the word, such as "stop the docker container" or "how do I cancel a subscription", is treated as a normal queued request.

A cancelled turn can leave a tool call without a result, which models reject. The next turn first answers such calls with `ERROR: cancelled by the user.`, so the conversation stays valid.

---

## Project layout

```
main.py              Entry point: reads the keyboard, optionally starts voice mode
chat_loop.py         ChatLoop: worker thread, input queue, stop handling, approvals, job notices
turns.py             One conversation turn (UI-independent), cancellation, stop phrases, sentence streaming for speech
jobs.py              Background shell jobs (JobManager)
tools.py             All agent tools, the path sandbox, the approval list
web.py               Web search and page reading (DuckDuckGo, safe fetching, caching)
retrieval.py         Ranking shared by RAG and the page reader: BM25, rank fusion, passage splitting
language.py          English/Vietnamese detection and localized phrases
llm_options.py       Settings for every Ollama model (thinking off, keep-alive, context size)
dependency.py        Builds model, RAG, tokenizer, memory (import-time singleton)
app.py               Unfinished refactor stub (AgentApplication)

graph/
  graph.py           The LangGraph definition (nodes and edges)
  node.py            Node functions: recall, router, direct, code_tools, tool_execute, save, summarize
chains/
  router_chain.py    direct-vs-code classifier (structured output)
  direct_chain.py    Plain chat answer
  code_chain.py      Coding agent prompt, model bound to the tools
  summerize_chain.py History summarizer (spelling is intentional, see gotchas)
  rag_chain.py       Standalone RAG chain (not used by the graph)
data/
  state.py           Graph state: messages, search_file, memory_context
  route_query.py     Router schema (destination: direct | code)
  rag_data.py        RagOutput schema (unused)
rag/
  rag.py             Chroma vector store, chunking, idempotent document sync, search tool
memories/
  database.py        Engine, sessions, init_db (pgvector extension + tables)
  table.py           Three tables (384-dim vectors, HNSW cosine indexes)
  store.py           Persistence and vector search, near-duplicate handling
  extractor.py       LLM structured extraction of what to remember from one turn
  manager.py         recall() before a turn, remember() after it
voice/
  assistant.py       The voice state machine
  audio.py           Microphone capture (sounddevice) and speaker playback
  wakeword.py        openWakeWord engine + WakeWordDetector (hits, cooldown)
  vad.py             Silero VAD + CommandRecorder (speech start/end, timeouts)
  stt.py             faster-whisper transcription
  tts.py             Piper synthesis + Speaker (sentence streaming, no overlap)
  speaker.py         Speaker verification (voiceprint, embeddings, verifier)
  enroll.py          `python -m voice.enroll` voice enrollment / live test
  check.py           `python -m voice.check` microphone and wake-word diagnostic
  text.py            Wake-word stripping, filler/yes-no handling, speech cleanup
  config.py          VoiceConfig (env vars, validated)
documents/           Knowledge files indexed by RAG (.txt and .py)
tests/               unittest suite (see Testing)
compose.yaml         Postgres + pgvector
requirements.txt     Core dependencies (pinned)
requirements_voice.txt  Optional voice dependencies (pinned)
CLAUDE.md            Notes for Claude Code
```

---

## How a message flows

```
                         ┌─────────────────────────────────────────┐
 keyboard / voice ──────►│ ChatLoop (queue + worker thread)        │
                         └───────────────┬─────────────────────────┘
                                         ▼
                              turns.run_turn  ──►  LangGraph app
                                                        │
 START ─► recall_memory ─► router ─┬─► direct ────────────────────┐
                                   │                              │
                                   └─► code_tools ⇄ tool_execute  │
                                          │  (while the reply has │
                                          │   tool_calls)         │
                                          ▼                       ▼
                                      save_memory ◄───────────────┘
                                          ▼
                                    reverse_router ─► (history > 5000 tokens?) ─► summarize ─► END
                                          └──────────────────────────────────────────────────► END
```

1. **`recall_memory`** searches long-term memory with the newest message and puts the result in `memory_context`, which is injected into the system prompt. Failures are logged, never fatal.
2. **`router`** classifies the message as `direct` or `code`, using the last 4 messages of context (each cut to 300 characters).
3. **`direct`** answers from the model's knowledge. **`code_tools`** is the coding agent; if its reply contains tool calls, **`tool_execute`** runs them and loops back until the model answers in plain text.
4. **`tool_execute`** never raises. An unknown tool or a tool error becomes an `ERROR: ...` tool message, output is capped, and a turn is limited to 30 tool calls (after that the model is told to answer with what it has). If any call needs approval, the node asks **once, before anything runs**, with `interrupt()`. On resume the node re-executes from the top, so nothing with side effects may happen before the interrupt.
5. **`save_memory`** extracts and stores memories from the finished turn (if memory is enabled).
6. **`summarize`** runs only when the history is too long.

A turn costs at least two model calls (the router, then the answer), plus one per tool round. See [Performance](#performance).

**Graph state** (`data/state.py`): `messages` (with the `add_messages` reducer, so nodes append instead of overwrite), `search_file`, `memory_context`.

**Checkpointer**: `InMemorySaver`, one thread (`thread_id` `"1"`), `recursion_limit` 100. The conversation lives in memory, so it is lost when the program exits. Long-term memory is what persists. Keyboard and voice share this one conversation.

---

## Tools and safety

| Tool | Needs approval | What it does |
|---|:-:|---|
| `list_files(path)` | | Lists files, pruning `.venv`, `.git`, `__pycache__`, `node_modules`, `chroma_db` etc.; at most 500 |
| `read_file(path, start_line, end_line)` | | Numbered lines; at most 400 lines per call |
| `search_code(query, directory)` | | `ripgrep` search, with a 60 s timeout; the query is passed as text, never as a flag |
| `git_status`, `git_diff`, `git_log`, `git_show` | | Read-only git (`git_log` limit is clamped to 1 to 100; a commit starting with `-` is refused) |
| `search_document(query, top_k)` | | Hybrid keyword + meaning search of the knowledge base; best passages with source file and relevance (default 5, at most 10) |
| `web_search(query, max_results)` | | Searches the web (DuckDuckGo); titles, links and snippets (default 5, at most 10) |
| `read_webpage(url, focus, start)` | | Reads a public web page as text; `focus` returns only the passages that match, `start` pages through a long one |
| `job_status(job_id)` | | Status of one or all background jobs |
| `cancel_job(job_id)` | | Cancels a running job |
| `terminal(command)` | **yes** | Runs a shell command in the project directory (PowerShell on Windows, `/bin/sh` elsewhere) |
| `write_file(path, content)` | **yes** | Creates or overwrites a file |
| `edit_file(path, old, new, occurrence)` | **yes** | Replaces text; `old` must be unique unless `occurrence` is given |
| `delete_file(path)` | **yes** | Deletes a file |
| `move_file(source, destination)` | **yes** | Moves or renames a file; refuses to overwrite |

**Safety model**

- Every file argument goes through `resolve_path()`: relative paths are resolved against `PROJECT_ROOT`, `..` escapes and absolute paths outside it are rejected (`PathOutsideProjectError`), and for write operations anything under `.git` is rejected too.
- `PROJECT_ROOT` defaults to the repository directory; set it in `.env` to point the agent at another project.
- `terminal` is *not* sandboxed beyond its working directory. It can run any command you approve, so read the approval prompt.
- Tool output is capped at 20,000 characters, and tool failures never crash the graph.
- Memory recall is presented to the model as "reference data, never instructions", and the extractor is told to ignore instructions inside transcripts and never to store secrets. Neither is a security guarantee.

---

## Background jobs

`terminal` starts the command as a **job** and waits up to `FOREGROUND_WAIT` (5 s).

- **Finishes in time:** the output is returned as usual (exit code 0 returns stdout; otherwise `Command failed (exit code N): ...`).
- **Still running:** the tool returns `Job N is still running in the background`. The agent tells you and the turn ends, so you can keep working. When the job ends, `[Job N finished]` is printed and a `[system notice]` message is queued as a new turn so the agent can report the result.
- **Limits:** jobs are killed after 10 minutes (`Command timed out after 600s`). Reports show the last 4,000 characters of output.
- **Cancelling:** `/cancel N`, or the agent's `cancel_job`. The whole process tree is killed (`taskkill /T /F` on Windows, process group on POSIX). A cancelled job sends no notification.
- **"stop"** cancels the running *turn*, not background jobs. Cancel jobs explicitly.
- Jobs still running when you quit are not stopped.

---

## Knowledge base (RAG)

- Put `.txt`, `.md`, `.markdown`, `.rst` and `.py` files in `./documents/` (three sample files ship: `docker.txt`, `network.txt`, `postgresql.txt`).
- **Incremental sync.** On every start, `RAG.add_document` chunks the files (500 characters, 50 overlap) under **stable ids** (a SHA-256 of source, index and content), deletes chunks of changed or removed files, and embeds **only the chunks that are new**. Unchanged chunks keep their stored embeddings, so a restart with no changes takes about 0.01 s instead of re-embedding everything (1 s for the 33 sample chunks, minutes for a large folder).
- **Hybrid search** (`RAG.search_hybrid`, used by the `search_document` tool). The query is run twice: by *meaning* (Chroma vectors, `BAAI/bge-small-en-v1.5`, 384 dimensions) and by *words* (BM25 in `retrieval.py`). The two rankings are merged with reciprocal rank fusion, with the exact-word list weighted slightly higher, because exact matches (identifiers, error messages, numbers) are what embeddings blur. At most 2 passages come from any one file, so one document cannot fill the answer.
- **Nothing is returned for an unrelated question.** A passage counts only if it matches by word or lies within vector distance 0.9 (calibrated: related queries had their best match at 0.23 to 0.70, unrelated ones at 1.06 or more; re-check it if you change the embedding model). Common words (English and Vietnamese stop words, and words found in over half the passages) are ignored by the keyword side. "What is the capital of France" returns "No relevant passages found" instead of the least-bad document.
- Accents are folded for the keyword side, so `tep` finds `tệp`.
- Results look like `[1] docker.txt (relevance 1.00)` followed by the passage; relevance is relative to the best hit of that query.
- The vector store is **Chroma** persisted in `./chroma_db` (gitignored).

---

## Web search

The agent has two web tools (`web_search`, `read_webpage`) for anything the repository and your documents cannot answer: library documentation, error messages, recent versions.

- **`web_search`** uses DuckDuckGo through the `ddgs` package: no account, no API key. It returns numbered results (title, link, 300-character snippet), drops duplicate links and allows at most 2 results per site. Results are cached for 10 minutes.
- **`read_webpage`** downloads a page and reduces it to readable text with `lxml` (menus, scripts, footers and ads are removed; an `<article>` or `<main>` is preferred). With **`focus`** set to what the agent is looking for, it returns only the best-matching passages, found with the same BM25 ranking as the knowledge base, so a 14,000-character page costs about 1,000 characters of context. Without it the page comes in parts of 6,000 characters (`start` continues). Pages are cached for 10 minutes.
- The agent is told to search, then read the best result with `focus`, instead of guessing from a snippet, to name the source it used, and to try `search_document` first for questions about your own documents.

**Safety**

- Everything from the web is labelled *untrusted data*: the model is told never to follow instructions found in a page (prompt injection is the main risk of letting a model read the web; this reduces it but cannot eliminate it).
- **Only the public internet can be opened.** Private, loopback, link-local and other non-public addresses are refused (your own machine, your router, `169.254.169.254` cloud-metadata addresses, IPv4-mapped IPv6 forms), as are non-http(s) links (`file://`, `ftp://`) and links containing a user name or password. Redirects are followed by hand and every hop is checked again, so a page cannot send the agent to an internal address. One residual risk remains: a host whose DNS answer changes between the check and the connection (DNS rebinding).
- Pages are limited to 2 MB and 10 seconds; only text, HTML, XML, JSON and Markdown are read (PDFs and images are refused). Failures come back to the model as `ERROR: ...` text, never as a crash.
- **Privacy:** a web search sends the query text to DuckDuckGo, and opening a page contacts that site. The web tools do not ask for approval (they cannot change your files), so a model that follows a malicious page could put information into a query or link. Set `WEB_SEARCH_ENABLED=false` to switch both tools off; they then answer "turned off".

---

## Long-term memory

Three memory types in Postgres with 384-dimension pgvector columns and **HNSW cosine** indexes (`memories/table.py`):

| Table | Holds | Dedup rule |
|---|---|---|
| `episodic_memories` | What happened: event, task, outcome, importance (0 to 1), thread id | None (each turn can add one) |
| `semantic_memories` | Durable facts with category and confidence | A near-duplicate (cosine distance ≤ 0.08) is **overwritten with the newest wording** |
| `procedural_memories` | A task and the ordered steps that worked, with a success count | A similar task (distance ≤ 0.2) gets its steps replaced and its count bumped |

**Why facts are overwritten.** A fact and its negation ("uses tabs" / "uses spaces") embed almost identically (distance about 0.004), so keeping the old text would silently discard a correction.

**Recall** (`recall_memory`): embeds the newest message and returns up to 3 of each type within distance 0.4, formatted into the system prompt.

**Remembering** (`save_memory`), after a finished turn:
1. Skip unless the turn ends in a plain-text answer.
2. Skip trivial turns (fewer than 3 words and no tool use).
3. Search known facts related to the question (up to 5, distance ≤ 0.45).
4. An LLM structured-extraction call returns an optional episode, facts, procedures and the ids of known facts that this turn **supersedes**. Superseded facts are deleted first.
5. Everything is stored; failures are logged and never break the conversation.

The distance thresholds were calibrated against bge-small; re-check them if you change the embedding model (and `EMBEDDING_DIM`, which also fixes the table schema).

The database URL comes from `DATABASE_URL` or `DB_URL`. A plain `postgresql://` is rewritten to `postgresql+psycopg://` because only psycopg 3 is installed.

---

## Conversation history and summaries

After every turn, if the history exceeds **5,000 tokens** (`MAX_HISTORY_TOKENS`; it must fit the 8192-token context window together with the ~1,400-token tool prompt and the answer, or Ollama silently drops the oldest messages) (counted with the Qwen3 tokenizer), `summarize` condenses everything *before the latest turn* into one summary message using the small model (`SUMMARY_MODEL`, default `qwen3:4b`). The latest turn is kept whole, so a tool call is never separated from its result. Messages are replaced via `RemoveMessage` plus re-added copies with fresh ids so that order is preserved.

---

## Voice mode

Voice is **only another input and output** for the same `ChatLoop`. Transcripts go through the same `submit()` the keyboard uses and the answer is read aloud as it streams, so tools, memory, checkpointer and conversation are shared. The keyboard keeps working while voice runs.

### Install and run

```powershell
python -m pip install -r requirements_voice.txt
python main.py --voice          # or set VOICE_ENABLED=true in .env
```

First run downloads about 250 MB into `~/.cache/langchain-voice` (override with `VOICE_MODEL_DIR`): Whisper `base`, the Piper voice `en_US-lessac-medium`, the openWakeWord models and, with speaker verification, a 26 MB speaker model. After that it works offline.

### Components

| Job | Engine | Notes |
|---|---|---|
| Wake word | [openWakeWord](https://github.com/dscripka/openWakeWord) (ONNX) | Free, offline. Needs several consecutive frames above the threshold, then a 2 s cooldown, so one utterance fires once |
| Speech start/end | Silero VAD + `CommandRecorder` | Timeouts for no speech and for maximum length |
| Speech to text | faster-whisper | Runs **only after** the wake word; whisper's own VAD filter and a no-speech filter reduce hallucinations |
| Text to speech | Piper | Synthesizes the next sentence while one is playing |
| Audio I/O | sounddevice (PortAudio) | 16 kHz mono capture in 80 ms frames; the callback only copies into a bounded queue |
| Speaker check | WeSpeaker ResNet34 via sherpa-onnx | See [Voice enrollment](#voice-enrollment-speaker-verification) |

### Say "Hey Jarvis"

The bundled wake model is **`hey_jarvis`**. Saying only "Jarvis" will **not** trigger it. openWakeWord ships these pretrained models: `alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy`, `timer`, `weather`. Set `WAKE_WORD` to one of them, or to the path of a custom `.onnx` model you trained yourself (a plain "jarvis" would need that). Arbitrary words are not supported without a trained model.

### The state machine

```
WAITING_FOR_WAKE_WORD ──► WAKE_WORD_DETECTED ──► LISTENING_FOR_COMMAND ──► TRANSCRIBING ──► PROCESSING ──► SPEAKING
        ▲                         ▲                       ▲                     │               │             │
        │                         │                       └──── (only the wake  ┘               │             │
        │                         └────────── wake word heard while PROCESSING ◄────────────────┘             │
        └────────────────────────────────── RETURN_TO_STANDBY ◄──────────────────────────────────────────────┘
                                   (any state) ──► STOPPING
```

Transitions are an explicit table (`TRANSITIONS` in `voice/assistant.py`); an illegal transition raises `InvalidTransition`. The console shows `[STANDBY]`, `[ACTIVATED]`, `[LISTENING]`, `[TRANSCRIBING]`, `[PROCESSING]`, `[SPEAKING]` and `[STOPPING]`, plus `You said: ...`.

**Both interaction styles work**

- *One breath*: "Hey Jarvis, explain the architecture". About 240 ms of audio from before the detection is kept, and a leading wake word in the transcript is stripped.
- *Two steps*: "Hey Jarvis." (pause) "Explain the architecture". If only the wake word was heard, a short beep means "go ahead".

**While the agent works**, the wake word is still monitored: "Hey Jarvis, stop" cancels the running turn and "Hey Jarvis, <request>" queues one. It is **never** monitored while the assistant is speaking.

### Self-activation, approvals, privacy

- **No self-activation**: speech output never reaches the detector. After every playback the microphone queue is flushed and the detector reset. There is no echo cancellation, and no barge-in (you cannot talk over a reply).
- **Spoken approvals**: the assistant reads out the pending action and listens for the answer ("Yes, allow it" to approve). Silence, an unclear answer or an unrecognized voice **declines**. "Stop" cancels the turn.
- **Speech runs in parallel with the streamed text.** The answer is spoken sentence by sentence *while it is still being generated*, so speech starts after the first sentence, not after the last. Sentences are spoken in short bursts, and the wake word is still checked between bursts, so "Hey Jarvis, stop" works mid-answer and drops whatever has not been spoken yet (typing `stop` does the same). Nothing is spoken twice.
- **What is spoken**: only the answer. Tool output, the assistant's notes before a tool call (the coding agent's newest sentence is held back until the next one starts, in case a tool call follows), code blocks ("code omitted"), `<think>` blocks, links and markdown are left out. Speech stops after `VOICE_MAX_SPOKEN_CHARS` characters with "The rest is on screen." (in Vietnamese when the answer is Vietnamese).
- **Recognizer noise**: transcripts such as "you", "uh", "thank you" are treated as nothing heard rather than as commands.
- **Privacy**: standby audio is only scored by the wake-word model and discarded; it is not transcribed, stored or sent anywhere. Recorded commands are held in memory only. Everything runs locally.
- **Failures**: a missing or unplugged microphone is retried (5 attempts with backoff) and then voice mode stops cleanly; transcription, TTS and agent errors return to standby (agent errors are spoken as a short apology).

### Voice enrollment (speaker verification)

```powershell
python -m voice.enroll          # read 6 fixed phrases once
python -m voice.enroll --test   # speak, or have someone else speak, and see similarity scores
```

Enrollment turns each phrase into a speaker embedding and saves the average plus a threshold calibrated from how consistent your phrases were (`voiceprint.npz` in the model directory). Once that file exists, verification is **on automatically** (or force it with `SPEAKER_VERIFICATION=true/false`).

- The voice check runs after transcription finds a real command, on **all audio of the interaction** (wake word plus command), so short utterances are judged on enough speech.
- Rejected audio prints `[REJECTED] ...` and never reaches the agent; the transcript is not shown.
- Spoken approvals are checked too, so a stranger cannot approve a tool run.
- Under 0.6 s of speech, or any embedding error, **fails closed** (rejects).
- Tune with `SPEAKER_THRESHOLD` (higher is stricter).
- It is a **convenience lock, not authentication**: a recording of you or a cloned voice can pass. The wake word itself still fires for anyone; only commands are checked.

### Troubleshooting the microphone

```powershell
python -m voice.check            # default device, 12 s
python -m voice.check 3 20       # device 3, 20 s
```

It lists input devices, shows a live level meter and the wake-word score, and ends with a verdict (silent / hears sound but wake word too low / working). Set the working device with `MICROPHONE_DEVICE=<number>`. Prefer the number: a name matches several entries (one per Windows audio API) and PortAudio rejects an ambiguous name.

---

## Configuration reference

All settings are environment variables (`.env` is loaded with `python-dotenv`).

### Core

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | (required) | Ollama model; must be a key of `TOKENIZER_MAP` |
| `SUMMARY_MODEL` | `qwen3:4b` | Model for history summaries |
| `LLM_REASONING` | `false` | `true` lets Qwen3 "think" before answering: smarter but several times slower (see [Performance](#performance)) |
| `LLM_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after a message |
| `LLM_NUM_CTX` | Ollama default (8192) | Context window; smaller uses less GPU memory |
| `MAX_HISTORY_TOKENS` | `5000` | History is summarized above this; keep it below the context window minus about 3,000 |
| `DB_URL` / `DATABASE_URL` | none | Postgres URL; without it memory is disabled |
| `PROJECT_ROOT` | repository dir | Directory the coding tools may touch |
| `WEB_SEARCH_ENABLED` | `true` | `false` turns off `web_search` and `read_webpage` |

### Voice

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_ENABLED` | `false` | Same as `--voice` |
| `WAKE_WORD` | `hey_jarvis` | Built-in model name or path to a `.onnx` |
| `WAKE_WORD_THRESHOLD` | `0.5` | 0.05 to 1.0; raise if it triggers by accident |
| `WAKE_WORD_MIN_FRAMES` | `2` | Consecutive 80 ms frames above the threshold (1 to 10) |
| `WAKE_WORD_COOLDOWN` | `2.0` | Seconds ignored after a detection (0 to 30) |
| `SPEECH_RECOGNITION_MODEL` | `base` | faster-whisper model (`tiny`, `base`, `small`, ...) |
| `SPEECH_LANGUAGE` | `en` | Language you speak: `en`, `vi`, a list like `vi,en` (detects which, among just those), or `auto`. Anything but plain `en` defaults the Whisper model to `small` |
| `SPEECH_RECOGNITION_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SPEECH_RECOGNITION_COMPUTE_TYPE` | `int8` | CTranslate2 compute type |
| `TTS_VOICE` | `en_US-lessac-medium` | Piper voice for English replies, downloaded on first use |
| `TTS_VOICE_VI` | `vi_VN-vais1000-medium` | Piper voice for replies written in Vietnamese (downloaded only if one is spoken) |
| `VOICE_MAX_SPOKEN_CHARS` | `1200` | Longest reply read aloud (50 to 20000) |
| `VOICE_LISTENING_TIMEOUT` | `10` | Seconds to wait for you to start speaking (1 to 120) |
| `VOICE_MAX_RECORDING_DURATION` | `60` | Longest command (2 to 600) |
| `VOICE_END_SILENCE` | `0.8` | Silence that ends a command (0.2 to 5) |
| `VOICE_VAD_THRESHOLD` | `0.5` | Speech probability threshold (0.05 to 0.95) |
| `MICROPHONE_DEVICE` | system default | Device index or name |
| `AUDIO_OUTPUT_DEVICE` | system default | Device index or name |
| `VOICE_MODEL_DIR` | `~/.cache/langchain-voice` | Model cache and voiceprint location |
| `SPEAKER_VERIFICATION` | auto | Empty: on if a voiceprint exists; `true`/`false` to force |
| `SPEAKER_THRESHOLD` | calibrated | 0.1 to 0.95; stricter when higher |
| `VOICEPRINT_PATH` | `<VOICE_MODEL_DIR>/voiceprint.npz` | Voiceprint file |

Invalid voice values raise a clear error, but only when voice mode is requested, so a typo cannot break keyboard use. (`VOICE_ENABLED` with an unrecognized value is treated as off.)

---

## Limits and defaults

| Limit | Value | Where |
|---|---|---|
| History before summarizing | 5,000 tokens (`MAX_HISTORY_TOKENS`) | `graph/node.py` |
| Tool calls per turn | 30 | `graph/node.py` |
| Tool output | 20,000 characters | `graph/node.py` |
| Graph recursion limit | 100 | `main.py` |
| Router context | last 4 messages, 300 characters each | `graph/node.py` |
| `read_file` | 400 lines per call | `tools.py` |
| `list_files` | 500 files | `tools.py` |
| git / ripgrep timeout | 60 s | `tools.py` |
| Terminal foreground wait | 5 s | `tools.py` |
| Job timeout / report size | 600 s / 4,000 characters | `jobs.py` |
| RAG chunks | 500 characters, 50 overlap, top 5 | `rag/rag.py` |
| Memory recall | 3 per type, distance ≤ 0.4 | `memories/store.py` |
| Memory transcript | 6,000 characters (tool results cut to 300) | `memories/extractor.py` |

---

## Vietnamese (Tiếng Việt)

You can type and speak Vietnamese, and the assistant answers in the language you used.

**Typed chat.** Nothing to configure. The prompts tell the model to reply in your language, keeping file names, code and commands unchanged. In a UTF-8 terminal (and with piped input, which `main.py` decodes as UTF-8) Vietnamese letters work as typed. These typed shortcuts understand Vietnamese:

| Where | Vietnamese |
|---|---|
| Stop the running turn | `dừng`, `dừng lại`, `ngừng`, `hủy`, `hủy bỏ`, `thôi`, `thôi đi`, `bỏ đi`, `đừng làm nữa`, with polite words (`làm ơn`, `vui lòng`, `nhé`, `ngay`...). Accent-less typing works for `dung lai`, `huy bo`, `thoi di`, `ngung lai`. Like English, only short messages count: "dừng chân ở đâu thì đẹp" is a normal question |
| Approve a tool run | `có`, `vâng`, `dạ`, `đồng ý` (with or without accents). Anything else declines |

**Voice.** Set the language you speak:

```
SPEECH_LANGUAGE=vi,en     # Vietnamese and English, detected per sentence (recommended if you mix them)
SPEECH_LANGUAGE=vi        # Vietnamese only
```

- **Speech recognition** switches to Whisper's `small` model automatically (it is far more accurate than `base` for Vietnamese; about 460 MB, downloaded once). With a list like `vi,en` the language of each utterance is detected **among just those**. Forcing `vi` makes Whisper *translate* English speech into Vietnamese words, which is why the list form exists.
- **Speech output.** Replies written in Vietnamese are read with the Piper voice `TTS_VOICE_VI` (default `vi_VN-vais1000-medium`); English replies keep `TTS_VOICE`. The voice is chosen per reply from its letters, and a voice is downloaded only when first needed.
- **Wake word** stays English: say **"Hey Jarvis"**, then your Vietnamese command. Leading forms such as "Hey Jarvis", "Này Jarvis", "Ê Jarvis", "Xin chào Jarvis" and a trailing "Jarvis ơi" are removed from the transcript.
- **Spoken approvals and stop** work in Vietnamese: the assistant asks "Tôi muốn chạy ... Hãy nói đồng ý để cho phép." when you last spoke Vietnamese, and understands "có", "đồng ý", "cho phép", "được rồi" (approve) and "dừng lại" (stop). Silence or anything unclear still declines.
- **Recognizer noise** is ignored, including Whisper's well-known Vietnamese hallucination on silence ("Hãy subscribe cho kênh Ghiền Mì Gõ...") and fillers such as "ừm" and "cảm ơn các bạn".
- **Voice enrollment** (`python -m voice.enroll`) shows Vietnamese phrases when the first language in `SPEECH_LANGUAGE` is `vi`.

**Measured** with Piper-generated Vietnamese speech (not real voices): Whisper `small` recognized most sentences with one or two wrong words, `base` was clearly worse. Vietnamese and English commands both reached the agent correctly under `vi,en`, and Vietnamese replies used the Vietnamese voice.

**Limits.**
- **Memory and the knowledge base are English-oriented.** Both use the English `bge-small-en-v1.5` embedding model, so recalling Vietnamese memories, or searching the English documents with Vietnamese words, works poorly. For documents the agent is told to search in English. A multilingual embedding model would fix this, but changing it means re-indexing the knowledge base and re-checking the memory distance thresholds, so it is not done.
- **Text typed without accents** ("xin chao") cannot be told apart from English, so replies to it, spoken apologies and spoken prompts use English.
- **Accuracy on real speech** (accents, noise, fast speech) was not measured. If a word is misheard, say it again, or try `SPEECH_RECOGNITION_MODEL=medium` (slower).
- The `qwen3` model's Vietnamese is decent but weaker than its English; the prompts do not change that.

---

## Performance

Speed is set by how many tokens the model generates, and on a small GPU that is slow. Measured on a laptop with an RTX 3050 (4 GB VRAM), where `qwen3:8b` runs **64% on the CPU** and generates about **4 tokens per second**:

| What | Before | After |
|---|---|---|
| Router call (direct or code?) | 96 to 113 s | **3 to 10 s** |
| Code agent call that requests a tool | 158 s | **7 s** |
| Short plain answer | 56 to 235 s | **13 to 16 s** |
| "What is a docker volume?" (full turn) | minutes | **16 s** |
| "List the python files in a folder" (full turn) | minutes | **20 s** |

What changed:

- **Qwen3 "thinking" is off** (`LLM_REASONING=false`). The model was silently generating hundreds of hidden reasoning tokens before every answer, including the one-word routing call. Turn it back on for hard problems, and expect several times the wait.
- **The model stays loaded** for 30 minutes (`LLM_KEEP_ALIVE`); Ollama's default unloads it after 5 minutes and reloading a multi-GB model is slow.
- **History limit lowered to 5,000 tokens** so a prompt always fits the context window. Prompt processing itself is fast (about 470 tokens/s), so long history costs seconds, not minutes.
- **Plain answers are asked to be concise**, because every 100 words of reply costs about 25 s at this speed.

What still dominates: **long answers**. A detailed explanation of a file can take a couple of minutes at 4 tokens/s, and each tool round adds a model call. Memory extraction (a full model call after each non-trivial turn) adds more once Postgres is running. Check the split with `ollama ps`: anything other than `100% GPU` means part of the model is on the CPU. Ways to go faster, from least to most disruptive:

0. Shrinking the context window (`LLM_NUM_CTX=4096` or `2048`) was measured and barely helps: the GPU share only moved from 36% to 39% and speed stayed within noise, so the default is unchanged.
1. Ask for short answers ("briefly").
2. Use a model that fits fully in your GPU memory (for example a 3B model on 4 GB). Models in the Qwen3 family keep thinking even when asked not to, so `qwen3:4b` was not faster in practice.
3. Add a card or run on a machine with more VRAM.

---

## Testing

```powershell
python -m unittest discover -s tests -t .
```

The suite needs **no microphone, speaker, model download or network**. `tests/stubs.py` replaces `dependency` and `chains` so nothing loads Ollama, Hugging Face models or Chroma. The Postgres memory test is skipped when the database is unreachable.

| File | Covers |
|---|---|
| `test_graph.py` | `tool_execute` (errors, limits, dispatch), approval flow on the real graph, summarization |
| `test_tools.py` | Path sandbox, file/git tools, terminal and background jobs |
| `test_jobs.py` | JobManager: detach, notify, cancel, timeout, truncation |
| `test_chat_loop.py` | Stop phrases, queueing, stop and approval handling, real-graph turns, cancellation repair |
| `test_rag.py`, `test_memory.py` | Document sync, memory extraction and store |
| `test_streaming_speech.py` | Sentence streaming: sentence boundaries, code blocks, tool-call narration held back, the spoken-length limit, cancellation, ordering (speech starts while the model is still generating), nothing spoken twice, the wake word between bursts |
| `test_retrieval.py` | BM25 (stop words, common words, accent folding), rank fusion and weights, passage splitting, best-passage selection |
| `test_web.py` | URL safety (private/loopback/metadata addresses, redirects, schemes, credentials), fetching limits, HTML cleaning, page reading and paging, search formatting and caching, switch-off. No network is used |
| `test_voice.py` | The voice state machine with fake audio parts: wake word, false detection, delayed command, empty/failed transcription, self-activation, mic disconnection, agent errors, shutdown, real `ChatLoop` + graph |
| `test_voice_parts.py` | Wake-word detector, recorder, speaker (TTS) streaming and text helpers |
| `test_vietnamese.py` | Vietnamese detection, stop and approval phrases, wake-word forms, filler, per-language voice choice, Whisper language selection, localized prompts and apologies |
| `test_speaker.py` | Voiceprint, verifier, enrollment flow, assistant voice-check integration |
| `test_voice_engines.py` | **Opt-in, real models**: Piper speech through the real wake word, VAD, Whisper and speaker model (see below) |

Opt-in real-model tests (downloads about 250 MB once, about a minute to run):

```powershell
$env:RUN_VOICE_ENGINE_TESTS = "1"
python -m unittest tests.test_voice_engines
```

Not covered by automated tests: real human voices, real accuracy and false-activation rates, speaker echo into the microphone, and a full conversation with a live Ollama model.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `password authentication failed for user "agent"` at startup, then "Long-term memory disabled" | The app connected to the wrong server: on some machines another process also answers on `localhost:5432`. Check what owns the port, or map the container to another host port (`"5433:5432"` in `compose.yaml`) and update `DB_URL`. The app runs fine without memory. |
| `KeyError: 'MODEL'` | Create `.env` from `.env.example`. |
| `KeyError` for your model name | Add its tokenizer to `TOKENIZER_MAP` in `dependency.py`. |
| `search_code` says ripgrep is not installed | Install `rg` and put it on PATH. |
| Garbled characters on Windows | `main.py` reconfigures stdout to UTF-8; use a terminal font that has the glyphs. |
| Voice mode does not react | Say **"Hey Jarvis"** (not "Jarvis"); run `python -m voice.check`; check `MICROPHONE_DEVICE`. |
| Voice mode rejects you | Run `python -m voice.enroll --test`, lower `SPEAKER_THRESHOLD`, re-enroll in the place and at the distance you actually use, or set `SPEAKER_VERIFICATION=false`. |
| `Speaker verification is enabled but there is no voiceprint` | Run `python -m voice.enroll` or set `SPEAKER_VERIFICATION=false`. |
| `Voice mode needs extra packages` | `python -m pip install -r requirements_voice.txt`. |
| Voice triggers by accident | Raise `WAKE_WORD_THRESHOLD` or `WAKE_WORD_MIN_FRAMES`. |

---

## Design notes and gotchas

- **Import-time singleton.** `dependency.py` builds the model, RAG, tokenizer and memory manager when imported (`dependency = dependency_init()`). `chains/*`, `tools.py` and `graph/node.py` read it with `from dependency import dependency`, so importing almost any module loads models and syncs `./documents/`. `dependency.py` must **not** import `graph` (circular: the graph nodes import `dependency`); `main.py` compiles the graph. Tests work around this by installing stubs first.
- **Threads.** The main thread reads the keyboard (or runs the voice loop). `ChatLoop` runs turns one at a time on a single worker thread, because two turns on one checkpointer thread would interleave messages. Job watchers run on their own daemon threads. The PortAudio callback only copies audio into a queue, and a short-lived helper thread synthesizes speech while the previous sentence plays.
- **Spelling convention.** The summarization chain is named `summerize` in file and symbol names (`summerize_chain.py`, `summerize_chain`), but the graph nodes use `summarize`. Keep both as they are.
- **Approval re-execution.** `interrupt()` makes `tool_execute` re-run from the top on resume, so the node asks before doing anything with side effects.
- **Tool registry.** `TOOLS` in `tools.py` is the single source of truth: the model is bound to it and the graph dispatches through `TOOL_BY_NAME`, so a tool cannot be bound but missing from dispatch.
- **Memory is optional.** `dependency["memory"]` is `None` when Postgres is unreachable, and every memory call is guarded.
- **Web results are data.** `web.py` labels every result as untrusted and checks every URL (and every redirect hop) against the public internet before connecting.
- **Voice never imports heavy libraries unless used.** `voice/__init__.py` imports the audio and model libraries inside `build_assistant`, so keyboard mode works without `requirements_voice.txt`.
- **Fail-closed security checks.** Speaker verification rejects on too little audio or on errors; spoken approvals decline on silence or doubt.

---

## Extending the project

**Add a tool**

1. Define it with `@tool` in `tools.py`; any file argument must go through `resolve_path()`.
2. Add it to the `TOOLS` list at the bottom.
3. Add a line for it in the code prompt in `chains/code_chain.py`.
4. Add its name to `NEEDS_APPROVAL` if it changes state or runs commands.
5. Add tests in `tests/test_tools.py`.

**Use another Ollama model**: pull it, set `MODEL`, and add its tokenizer to `TOKENIZER_MAP`.

**Add documents to the knowledge base**: drop `.txt` or `.py` files into `documents/` and restart.

**Use another wake word**: set `WAKE_WORD` to a built-in name or a custom `.onnx` model path.

**Change the voice language or voice**: set `SPEECH_LANGUAGE` and pick a Piper voice name for `TTS_VOICE`.

---

## Known limitations

- **Conversation is in memory only.** History is lost when the program exits; only long-term memory persists.
- **Stop is cooperative.** Cancelling is checked between streamed chunks, so a single tool call that is already running finishes first (at most about 5 s for `terminal`, up to 60 s for git and search).
- **`terminal` is powerful.** It runs whatever you approve.
- **Unused leftovers.** `chains/rag_chain.py` and `data/rag_data.py` are not wired into the graph (RAG is exposed as the `search_document` tool instead). `app.py` / `AgentApplication` is an empty stub. `transformers` is loaded only to count tokens for the summarize trigger.
- **Vietnamese** memory and document search are weak (English-only embeddings); see [Vietnamese](#vietnamese-tiếng-việt).
- **Voice**: tested with synthetic speech and an idle microphone, not with a range of real voices. There is no barge-in and no echo cancellation. No plain "jarvis" model ships with openWakeWord. Speaker verification is not strong authentication.
- **Platform**: tested on Windows only.

---

## License

MIT. See [LICENSE](LICENSE).
