# Web chat frontend (`webui/`)

TypeScript browser app for `python -m backend` (API: `backend/CLAUDE.md`). No framework, no runtime dependencies; dev dependencies `typescript` and `@types/node`, pinned exactly (a test checks).

```powershell
npm --prefix webui install          # once
npm --prefix webui run build        # src/*.ts -> dist/*.js, public/* -> dist/ (backend serves dist/)
npm --prefix webui test             # node --test on test/*.test.ts (Node runs .ts directly)
npm --prefix webui run check        # tsc type-check of src and test
```

- `src/types.ts`: backend shapes (`ServerInfo`, `SessionInfo`, `ChatEvent`) and `PROTOCOL` (must equal `backend/server.py`). `src/api.ts`: `Api` client (`handshake`, sessions, `transcribe`, `speak`); never throws, returns `Result`. `src/state.ts`: pure reducer `apply(conversation, event)` (ignores already-seen ids, so a replay is harmless; `notice` events become notice items), `blocks()` (``` fences → code), sidebar helpers. `src/url.ts`: token from `?token=` (then kept in `sessionStorage`, removed from the address bar), session id from `#<id>`. `src/handsfree.ts` (pure, tested): `PressTracker` (hold/tap/double tap), `Endpointer` (speech start/end from levels; threshold follows the noise floor), `handsFreeAction`, `insertDictation`. `src/voice.ts`: `Recorder` (MediaRecorder, webm/opus, 60 s cap), `Player` (`onChange` pauses hands-free while it plays), `HandsFreeListener` (one mic, an Endpointer on AnalyserNode levels, one MediaRecorder file per utterance; not unit-tested: smoke-tested in headless Chrome with `--use-file-for-fake-audio-capture`), pure `replyToSpeak` (only `done` events after the history present when the session was opened). `src/main.ts`: DOM only; handshake first (no page without it), sidebar polls `/api/sessions` every 2 s, one `EventSource` for the active session; Talk / Hands-free buttons and "Read answers aloud" only when the handshake offers voice (choice kept in `localStorage`). Talk: hold (or hold Right Ctrl) = dictation via `api.dictate` inserted into the message box at the cursor, not sent; a short click records until the next click. Hands-free (button or double tap Right Ctrl): each utterance → `handsFreeAction` → queued prompt (sent when the session is idle; a 409 re-queues), approval answer, stop, or off; stopping hands-free keeps queued prompts, "stop" or switching sessions drops them; off after 2 min without speech.
- `public/`: `index.html`, `style.css`, copied to `dist/` by `scripts/copy-public.mjs`. `dist/` and `node_modules/` are gitignored.
- Rules: relative imports name the `.ts` file (`rewriteRelativeImportExtensions` turns them into `.js`); only erasable TypeScript (`erasableSyntaxOnly`: no enums/namespaces/parameter properties). Never `innerHTML`/`eval` (a test scans `src/`): model and web-page text is untrusted, inserted with `textContent` only. CSP forbids inline script/style: no `<script>` blocks, `style=` attributes or `on*=` handlers in HTML.

`tests/test_webui_frontend.py` runs `npm test`, `npm run check` and a build+serve check from the Python suite (skipped without `npm` or `node_modules`).
