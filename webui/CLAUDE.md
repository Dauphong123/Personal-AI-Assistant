# Web chat frontend (`webui/`)

TypeScript browser app for `python -m backend` (API: `backend/CLAUDE.md`). No framework, no runtime dependencies; dev dependencies `typescript` and `@types/node`, pinned exactly (a test checks).

```powershell
npm --prefix webui install          # once
npm --prefix webui run build        # src/*.ts -> dist/*.js, public/* -> dist/ (backend serves dist/)
npm --prefix webui test             # node --test on test/*.test.ts (Node runs .ts directly)
npm --prefix webui run check        # tsc type-check of src and test
```

- `src/types.ts`: backend shapes (`SessionInfo`, `ChatEvent`). `src/api.ts`: `Api` client; never throws, returns `Result`. `src/state.ts`: pure reducer `apply(conversation, event)` (ignores already-seen ids, so a replay is harmless), `blocks()` (``` fences → code), sidebar helpers. `src/url.ts`: token from `?token=` (then kept in `sessionStorage`, removed from the address bar), session id from `#<id>`. `src/main.ts`: DOM only; sidebar polls `/api/sessions` every 2 s, one `EventSource` for the active session.
- `public/`: `index.html`, `style.css`, copied to `dist/` by `scripts/copy-public.mjs`. `dist/` and `node_modules/` are gitignored.
- Rules: relative imports name the `.ts` file (`rewriteRelativeImportExtensions` turns them into `.js`); only erasable TypeScript (`erasableSyntaxOnly`: no enums/namespaces/parameter properties). Never `innerHTML`/`eval` (a test scans `src/`): model and web-page text is untrusted, inserted with `textContent` only. CSP forbids inline script/style: no `<script>` blocks, `style=` attributes or `on*=` handlers in HTML.

`tests/test_webui_frontend.py` runs `npm test`, `npm run check` and a build+serve check from the Python suite (skipped without `npm` or `node_modules`).
