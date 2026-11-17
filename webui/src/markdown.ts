// Markdown, the subset a chat model writes: headings, paragraphs, fenced code, lists (nested),
// quotes, rules, tables, and inline code, strong, emphasis, strikethrough and links. Pure: it
// returns a tree, and the page builds elements from it with textContent only, so nothing here
// can inject HTML. Half-streamed text is fine: unclosed markers stay literal, an unclosed fence
// is code to the end (marked open).

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "code"; text: string }
  | { kind: "strong" | "em" | "del"; children: Inline[] }
  | { kind: "link"; href: string; children: Inline[] };

export type Align = "left" | "right" | "center" | null;

export type MdBlock =
  | { kind: "paragraph"; inlines: Inline[] }
  | { kind: "heading"; level: number; inlines: Inline[] }
  | { kind: "code"; lang: string; text: string; open: boolean }
  | { kind: "list"; ordered: boolean; start: number; items: MdBlock[][] }
  | { kind: "quote"; blocks: MdBlock[] }
  | { kind: "rule" }
  | { kind: "table"; align: Align[]; header: Inline[][]; rows: Inline[][][] };

const MAX_BLOCK_DEPTH = 12; // deeper quotes/lists are shown as plain paragraphs
const MAX_INLINE_DEPTH = 6;
const TAB = 4;

// --- links ------------------------------------------------------------------------------

/** The link target if it is safe to open (http, https, mailto), else null. */
export function safeHref(raw: string): string | null {
  const url = raw.trim();

  if (/[\s\u0000-\u001f\u007f]/.test(url)) return null;

  return /^(https?:\/\/[^/\\]|mailto:[^/\\])/i.test(url) ? url : null;
}

// --- blocks -------------------------------------------------------------------------------

const FENCE = /^(\s*)(`{3,}|~{3,})(.*)$/;
const HEADING = /^ {0,3}(#{1,6})(?:\s+(.*))?$/;
const RULE = /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/;
const QUOTE = /^\s*>/;
const ITEM = /^(\s*)([-*+]|\d{1,9}[.)])([ \t]+|$)(.*)$/;
const TABLE_SEPARATOR = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

/** Columns of leading whitespace (a tab counts as four). */
function indentOf(line: string): number {
  let column = 0;

  for (const ch of line) {
    if (ch === " ") column += 1;
    else if (ch === "\t") column += TAB - (column % TAB);
    else break;
  }

  return column;
}

/** The line without up to `columns` columns of leading whitespace. */
function stripIndent(line: string, columns: number): string {
  let column = 0;
  let index = 0;

  while (index < line.length && column < columns) {
    const ch = line[index];

    if (ch === " ") column += 1;
    else if (ch === "\t") column += TAB - (column % TAB);
    else break;

    index += 1;
  }

  return line.slice(index);
}

const isBlank = (line: string): boolean => line.trim() === "";

function fenceOf(line: string): { indent: number; marker: string; lang: string } | null {
  const match = FENCE.exec(line);

  if (!match) return null;

  const marker = match[2] ?? "";
  const info = match[3] ?? "";

  if (marker.startsWith("`") && info.includes("`")) return null; // ```inline``` is a code span

  return { indent: indentOf(match[1] ?? ""), marker, lang: info.trim().split(/\s+/)[0] ?? "" };
}

function isTableStart(lines: readonly string[], index: number): boolean {
  const head = lines[index] ?? "";
  const separator = lines[index + 1] ?? "";
  // The separator row must have a pipe too, so "---" under a line of text is not a table.
  return head.includes("|") && separator.includes("|") && TABLE_SEPARATOR.test(separator);
}

/** Does this line start a block that ends a paragraph? */
function startsBlock(lines: readonly string[], index: number): boolean {
  const line = lines[index] ?? "";

  if (fenceOf(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line)) return true;

  const item = ITEM.exec(line);

  if (item && (item[4] ?? "").trim() !== "") {
    const marker = item[2] ?? "";
    // A numbered line interrupts a paragraph only as "1." (so "2024. Then" stays text).
    if (!/\d/.test(marker) || parseInt(marker, 10) === 1) return true;
  }

  return isTableStart(lines, index);
}

export function parseMarkdown(source: string): MdBlock[] {
  return parseBlocks(source.replace(/\r\n?/g, "\n").split("\n"), 0);
}

function parseBlocks(lines: readonly string[], depth: number): MdBlock[] {
  if (depth > MAX_BLOCK_DEPTH) {
    const text = lines.join("\n").trim();
    return text ? [{ kind: "paragraph", inlines: [{ kind: "text", text }] }] : [];
  }

  const out: MdBlock[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i] ?? "";

    if (isBlank(line)) {
      i += 1;
      continue;
    }

    const fence = fenceOf(line);

    if (fence) {
      const body: string[] = [];
      let open = true;
      i += 1;

      while (i < lines.length) {
        const current = lines[i] ?? "";
        const close = /^\s*(`{3,}|~{3,})\s*$/.exec(current)?.[1];

        if (close && close[0] === fence.marker[0] && close.length >= fence.marker.length) {
          open = false;
          i += 1;
          break;
        }

        body.push(stripIndent(current, fence.indent));
        i += 1;
      }

      const text = body.join("\n");
      out.push({ kind: "code", lang: fence.lang, text: open ? text.replace(/\n+$/, "") : text, open });
      continue;
    }

    const heading = HEADING.exec(line);

    if (heading) {
      const content = (heading[2] ?? "").replace(/(^|\s+)#+\s*$/, "").trim();
      out.push({ kind: "heading", level: (heading[1] ?? "#").length, inlines: parseInline(content) });
      i += 1;
      continue;
    }

    if (RULE.test(line)) {
      out.push({ kind: "rule" });
      i += 1;
      continue;
    }

    if (QUOTE.test(line)) {
      const inner: string[] = [];

      while (i < lines.length && QUOTE.test(lines[i] ?? "")) {
        inner.push((lines[i] ?? "").replace(/^\s*> ?/, ""));
        i += 1;
      }

      out.push({ kind: "quote", blocks: parseBlocks(inner, depth + 1) });
      continue;
    }

    if (ITEM.test(line)) {
      i = parseList(lines, i, depth, out);
      continue;
    }

    if (isTableStart(lines, i)) {
      i = parseTable(lines, i, out);
      continue;
    }

    const paragraph = [line.trim()];
    i += 1;

    while (i < lines.length && !isBlank(lines[i] ?? "") && !startsBlock(lines, i)) {
      paragraph.push((lines[i] ?? "").trim());
      i += 1;
    }

    out.push({ kind: "paragraph", inlines: parseInline(paragraph.join("\n")) });
  }

  return out;
}

/** A list starting at `start`; pushes it to `out` and returns the index after it. */
function parseList(lines: readonly string[], start: number, depth: number, out: MdBlock[]): number {
  const first = ITEM.exec(lines[start] ?? "");
  const baseIndent = indentOf(first?.[1] ?? "");
  const ordered = /\d/.test(first?.[2] ?? "");
  const items: string[][] = [];
  let current: string[] = [];
  let contentIndent = 0;
  let previousBlank = false;
  let i = start;

  while (i < lines.length) {
    const line = lines[i] ?? "";

    if (isBlank(line)) {
      let next = i + 1;

      while (next < lines.length && isBlank(lines[next] ?? "")) next += 1;

      const following = lines[next];

      if (following === undefined) break;

      const item = ITEM.exec(following);
      const continues =
        indentOf(following) > baseIndent ||
        (item !== null && !RULE.test(following) && /\d/.test(item[2] ?? "") === ordered);

      if (!continues) break;

      current.push("");
      previousBlank = true;
      i += 1;
      continue;
    }

    const indent = indentOf(line);
    const item = ITEM.exec(line);

    if (item && indent <= baseIndent + 1 && !RULE.test(line)) {
      if (/\d/.test(item[2] ?? "") !== ordered) break; // another kind of list starts

      const marker = item[2] ?? "";
      const gap = (item[3] ?? "").length;
      contentIndent = indent + marker.length + (gap >= 1 && gap <= 4 ? gap : 1);
      current = [item[4] ?? ""];
      items.push(current);
      previousBlank = false;
      i += 1;
      continue;
    }

    if (indent > baseIndent) {
      current.push(stripIndent(line, contentIndent));
      previousBlank = false;
      i += 1;
      continue;
    }

    if (!previousBlank && !startsBlock(lines, i)) {
      current.push(line.trim()); // a lazy continuation line
      i += 1;
      continue;
    }

    break;
  }

  const startNumber = ordered ? parseInt(first?.[2] ?? "1", 10) : 1;
  out.push({ kind: "list", ordered, start: startNumber, items: items.map((item) => parseBlocks(item, depth + 1)) });
  return i;
}

/** Cells of one table row: outer pipes dropped, split on pipes that are not escaped. */
function splitRow(line: string): string[] {
  let row = line.trim();

  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|") && !row.endsWith("\\|")) row = row.slice(0, -1);

  const cells: string[] = [];
  let cell = "";

  for (let index = 0; index < row.length; index += 1) {
    const ch = row[index];

    if (ch === "\\" && row[index + 1] === "|") {
      cell += "\\|";
      index += 1;
    } else if (ch === "|") {
      cells.push(cell.trim());
      cell = "";
    } else {
      cell += ch;
    }
  }

  cells.push(cell.trim());
  return cells;
}

function parseTable(lines: readonly string[], start: number, out: MdBlock[]): number {
  const header = splitRow(lines[start] ?? "");
  const align = splitRow(lines[start + 1] ?? "").map((cell): Align => {
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");
    return left && right ? "center" : left ? "left" : right ? "right" : null;
  });
  const width = header.length;
  const rows: Inline[][][] = [];
  let i = start + 2;

  while (i < lines.length && !isBlank(lines[i] ?? "") && (lines[i] ?? "").includes("|")) {
    const cells = splitRow(lines[i] ?? "").slice(0, width);
    rows.push(Array.from({ length: width }, (_, column) => parseInline(cells[column] ?? "")));
    i += 1;
  }

  out.push({
    kind: "table",
    align: Array.from({ length: width }, (_, column) => align[column] ?? null),
    header: header.map((cell) => parseInline(cell)),
    rows,
  });
  return i;
}

// --- inline -------------------------------------------------------------------------------

const PUNCTUATION = /[!-/:-@[-`{-~]/;
const isSpace = (ch: string | undefined): boolean => ch === undefined || /\s/.test(ch);
const isWordChar = (ch: string | undefined): boolean => ch !== undefined && /[\p{L}\p{N}]/u.test(ch);

function pushText(out: Inline[], text: string): void {
  if (!text) return;

  const last = out.at(-1);

  if (last?.kind === "text") {
    out[out.length - 1] = { kind: "text", text: last.text + text };
  } else {
    out.push({ kind: "text", text });
  }
}

function pushAll(out: Inline[], items: readonly Inline[]): void {
  for (const item of items) {
    if (item.kind === "text") pushText(out, item.text);
    else out.push(item);
  }
}

function runLength(src: string, at: number, ch: string): number {
  let end = at;

  while (src[end] === ch) end += 1;

  return end - at;
}

/** Where the code span opened by the backtick run at `at` closes, or -1. */
function codeSpanEnd(src: string, at: number): { close: number; run: number } | null {
  const run = runLength(src, at, "`");
  let j = at + run;

  while (j < src.length) {
    if (src[j] === "`") {
      const length = runLength(src, j, "`");

      if (length === run) return { close: j, run };

      j += length;
    } else {
      j += 1;
    }
  }

  return null;
}

/** Index of the closing delimiter (`width` copies of `ch`) after `from`, or -1. Skips
 * escapes and code spans; a longer closing run closes at its end (`***` ends `*a **b***`). */
function findClose(src: string, from: number, ch: string, width: number): number {
  let j = from;

  while (j < src.length) {
    const c = src[j];

    if (c === "\\") {
      j += 2;
      continue;
    }

    if (c === "`") {
      const span = codeSpanEnd(src, j);
      j = span ? span.close + span.run : j + runLength(src, j, "`");
      continue;
    }

    if (c === ch) {
      const run = runLength(src, j, ch);
      const candidate = run === width ? j : run >= 3 ? j + run - width : -1;
      const closes =
        candidate > from &&
        !isSpace(src[j - 1]) &&
        !(ch === "_" && isWordChar(src[j + run]));

      if (closes) return candidate;

      j += run;
      continue;
    }

    j += 1;
  }

  return -1;
}

/** A [label](destination) starting at `at` ("["), or null if it is not one (yet). */
function linkAt(src: string, at: number): { end: number; label: string; url: string } | null {
  let depth = 0;
  let j = at;

  for (; j < src.length; j += 1) {
    const c = src[j];

    if (c === "\\") {
      j += 1;
    } else if (c === "[") {
      depth += 1;
    } else if (c === "]") {
      depth -= 1;

      if (depth === 0) break;
    }
  }

  if (j >= src.length || src[j + 1] !== "(") return null;

  let k = j + 2;
  let parens = 1;

  for (; k < src.length; k += 1) {
    const c = src[k];

    if (c === "\\") {
      k += 1;
    } else if (c === "(") {
      parens += 1;
    } else if (c === ")") {
      parens -= 1;

      if (parens === 0) break;
    }
  }

  if (k >= src.length) return null;

  const destination = /^(<[^>]*>|\S+)(?:\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?$/.exec(src.slice(j + 2, k).trim());

  if (!destination) return null;

  const url = (destination[1] ?? "").replace(/^<(.*)>$/, "$1");
  return { end: k + 1, label: src.slice(at + 1, j), url };
}

/** A bare http(s) URL at `at`, without trailing punctuation or an unmatched ")". */
function bareUrlAt(src: string, at: number): string | null {
  const match = /^https?:\/\/[^\s<>]+/i.exec(src.slice(at));

  if (!match) return null;

  let url = match[0];

  for (;;) {
    const last = url.at(-1) ?? "";
    const unmatched = last === ")" && (url.match(/\(/g)?.length ?? 0) < (url.match(/\)/g)?.length ?? 0);

    if (".,;:!?'\"*_~]".includes(last) || unmatched) {
      url = url.slice(0, -1);
    } else {
      break;
    }
  }

  return /^https?:\/\/./i.test(url) ? url : null;
}

export function parseInline(src: string, depth = 0): Inline[] {
  const out: Inline[] = [];

  if (depth > MAX_INLINE_DEPTH) {
    pushText(out, src);
    return out;
  }

  let i = 0;

  while (i < src.length) {
    const c = src[i] ?? "";

    if (c === "\\" && i + 1 < src.length && PUNCTUATION.test(src[i + 1] ?? "")) {
      pushText(out, src[i + 1] ?? "");
      i += 2;
      continue;
    }

    if (c === "`") {
      const span = codeSpanEnd(src, i);

      if (span) {
        let text = src.slice(i + span.run, span.close);

        if (text.length > 2 && text.startsWith(" ") && text.endsWith(" ") && text.trim() !== "") {
          text = text.slice(1, -1);
        }

        out.push({ kind: "code", text });
        i = span.close + span.run;
      } else {
        const run = runLength(src, i, "`");
        pushText(out, src.slice(i, i + run));
        i += run;
      }

      continue;
    }

    const image = c === "!" && src[i + 1] === "[";

    if (c === "[" || image) {
      const link = linkAt(src, image ? i + 1 : i);

      if (link) {
        const href = safeHref(link.url);
        const children = image ? [{ kind: "text" as const, text: link.label }] : parseInline(link.label, depth + 1);

        if (href) {
          out.push({ kind: "link", href, children });
        } else {
          pushAll(out, children);
        }

        i = link.end;
        continue;
      }
    }

    if (c === "<") {
      const auto = /^<(https?:\/\/[^\s<>]+)>/i.exec(src.slice(i));

      if (auto?.[1]) {
        out.push({ kind: "link", href: auto[1], children: [{ kind: "text", text: auto[1] }] });
        i += auto[0].length;
        continue;
      }
    }

    if ((c === "h" || c === "H") && !isWordChar(src[i - 1])) {
      const url = bareUrlAt(src, i);

      if (url) {
        out.push({ kind: "link", href: url, children: [{ kind: "text", text: url }] });
        i += url.length;
        continue;
      }
    }

    if (c === "*" || c === "_" || c === "~") {
      const run = runLength(src, i, c);
      const canOpen = !isSpace(src[i + run]) && !(c === "_" && isWordChar(src[i - 1]));
      let matched = false;

      if (canOpen && c === "~") {
        const close = run === 2 ? findClose(src, i + 2, "~", 2) : -1;

        if (close >= 0) {
          out.push({ kind: "del", children: parseInline(src.slice(i + 2, close), depth + 1) });
          i = close + 2;
          matched = true;
        }
      } else if (canOpen) {
        if (run >= 2) {
          const close = findClose(src, i + 2, c, 2);

          if (close >= 0) {
            out.push({ kind: "strong", children: parseInline(src.slice(i + 2, close), depth + 1) });
            i = close + 2;
            matched = true;
          }
        }

        if (!matched && run !== 2) {
          const close = findClose(src, i + 1, c, 1);

          if (close >= 0) {
            out.push({ kind: "em", children: parseInline(src.slice(i + 1, close), depth + 1) });
            i = close + 1;
            matched = true;
          }
        }
      }

      if (!matched) {
        pushText(out, src.slice(i, i + run));
        i += run;
      }

      continue;
    }

    pushText(out, c);
    i += 1;
  }

  return out;
}
