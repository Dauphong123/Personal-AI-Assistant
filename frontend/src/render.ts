// Markdown tree (markdown.ts) -> DOM. Text only ever goes in through textContent or text
// nodes; link targets were checked by safeHref.

import type { Inline, MdBlock } from "./markdown.ts";
import { describeTool } from "./state.ts";
import type { CallView, ToolItem } from "./state.ts";

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);

  if (className) node.className = className;

  return node;
}

const SVG = "http://www.w3.org/2000/svg";

/** A small "Copy" button; the page's click handler finds what to copy by its class. */
export function copyButton(className: string, label: string): HTMLButtonElement {
  const button = el("button", `copy ${className}`);
  button.type = "button";
  button.setAttribute("aria-label", label);

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(SVG, "path");
  path.setAttribute("d", "M9 9h10v11H9zM5 15V4h10");
  svg.append(path);

  const text = el("span", "label");
  text.textContent = "Copy";
  button.append(svg, text);
  return button;
}

/** A small text+icon button for a message (Edit, Retry); the page's click handler acts on
 * its class. Styled like Copy. */
export function actionButton(className: string, text: string, d: string, title: string): HTMLButtonElement {
  const button = el("button", `copy action ${className}`);
  button.type = "button";
  button.title = title;
  const label = el("span", "label");
  label.textContent = text;
  button.append(icon("", d), label);
  return button;
}

export const ICONS = {
  edit: "M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4",
  retry: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5",
};

function icon(className: string, d: string): SVGSVGElement {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", className ? `icon ${className}` : "icon");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(SVG, "path");
  path.setAttribute("d", d);
  svg.append(path);
  return svg;
}

const TOOL_MARKS: Record<Exclude<ToolItem["status"], "running">, [string, string]> = {
  done: ["M5 12.5l4.5 4.5L19 7.5", "Done"],
  error: ["M12 7v6M12 16.5v.5M4 12a8 8 0 1 0 16 0a8 8 0 1 0-16 0", "Failed"],
  stopped: ["M7 12h10", "Stopped"],
};

/** One tool call as a card: what it does and to what; the output opens on a click. */
export function renderTool(item: ToolItem): HTMLElement {
  const view = describeTool(item.name, item.args);
  const card = el("details", `activity ${item.status}`);
  const summary = el("summary");
  const title = el("span", "activity-title");
  title.textContent = item.status === "running" ? view.doing : view.done;
  const state = el("span", "activity-state");

  if (item.status === "running") {
    const spinner = el("span", "activity-spinner");
    spinner.setAttribute("aria-hidden", "true");
    summary.append(spinner, title);
    state.textContent = "Working…";
  } else {
    const [d, label] = TOOL_MARKS[item.status];
    summary.append(icon("activity-mark", d), title);
    state.textContent = item.status === "done" ? "" : label;
  }

  if (view.target !== null) {
    const target = el("code", "activity-target");
    target.textContent = view.target;
    target.title = view.target;
    summary.append(target);
  }

  summary.append(state);

  if (item.text) {
    summary.append(icon("activity-chevron", "M9 6l6 6-6 6"));
    const pre = el("pre");
    pre.textContent = item.text;
    card.append(summary, pre);
  } else {
    // Nothing to open yet: a click must not toggle an empty card.
    summary.addEventListener("click", (event) => event.preventDefault());
    card.append(summary);
  }

  return card;
}

export function renderInlines(inlines: readonly Inline[]): Node[] {
  return inlines.map((inline): Node => {
    switch (inline.kind) {
      case "text":
        return document.createTextNode(inline.text);
      case "code": {
        const code = el("code");
        code.textContent = inline.text;
        return code;
      }
      case "strong":
      case "em":
      case "del": {
        const node = el(inline.kind === "strong" ? "strong" : inline.kind === "em" ? "em" : "del");
        node.append(...renderInlines(inline.children));
        return node;
      }
      case "link": {
        const link = el("a");
        link.href = inline.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.append(...renderInlines(inline.children));
        return link;
      }
    }
  });
}

function renderListItem(blocks: readonly MdBlock[]): HTMLLIElement {
  const li = el("li");
  const [only] = blocks;

  // A tight item (one paragraph) holds its text directly, without a paragraph margin.
  if (blocks.length === 1 && only?.kind === "paragraph") {
    li.append(...renderInlines(only.inlines));
  } else {
    li.append(...renderBlocks(blocks));
  }

  return li;
}

export function renderBlocks(blocks: readonly MdBlock[]): HTMLElement[] {
  return blocks.map((block): HTMLElement => {
    switch (block.kind) {
      case "paragraph": {
        const p = el("p");
        p.append(...renderInlines(block.inlines));
        return p;
      }
      case "heading": {
        // Message headings sit below the page's own h1-h3.
        const level = Math.min(6, Math.max(4, block.level + 3)) as 4 | 5 | 6;
        const heading = el(`h${level}`, `md-h${block.level}`);
        heading.append(...renderInlines(block.inlines));
        return heading;
      }
      case "code": {
        // <figure class="code"> <div class="code-head">lang [Copy]</div> <pre><code>…</code></pre>
        const figure = el("figure", block.open ? "code open" : "code");
        const head = el("div", "code-head");
        const lang = el("span", "code-lang");
        lang.textContent = block.lang || "code";
        head.append(lang);

        if (!block.open) head.append(copyButton("copy-code", "Copy code"));

        const pre = el("pre", "code-block");
        const code = el("code");
        code.textContent = block.text;
        pre.append(code);
        figure.append(head, pre);
        return figure;
      }
      case "list": {
        const list = block.ordered ? el("ol") : el("ul");

        if (block.ordered && block.start !== 1 && list instanceof HTMLOListElement) list.start = block.start;

        list.append(...block.items.map(renderListItem));
        return list;
      }
      case "quote": {
        const quote = el("blockquote");
        quote.append(...renderBlocks(block.blocks));
        return quote;
      }
      case "rule":
        return el("hr");
      case "table": {
        const wrap = el("div", "table-wrap");
        const table = el("table");
        const head = el("thead");
        const headRow = el("tr");
        const align = (column: number): string => (block.align[column] ? `align-${block.align[column]}` : "");

        block.header.forEach((cell, column) => {
          const th = el("th", align(column));
          th.scope = "col";
          th.append(...renderInlines(cell));
          headRow.append(th);
        });

        head.append(headRow);
        const body = el("tbody");

        for (const row of block.rows) {
          const tr = el("tr");

          row.forEach((cell, column) => {
            const td = el("td", align(column));
            td.append(...renderInlines(cell));
            tr.append(td);
          });

          body.append(tr);
        }

        table.append(head, body);
        wrap.append(table);
        return wrap;
      }
    }
  });
}

// --- approval card -------------------------------------------------------------------------


const FIELD_TONE: Record<string, string> = { Replace: "old", With: "new" };

/** One card per tool call: what it does, the file it touches, and its arguments. */
export function renderCalls(calls: readonly CallView[]): HTMLElement[] {
  return calls.map((call) => {
    const card = el("article", call.danger ? "call danger" : "call");
    const head = el("div", "call-head");
    const title = el("span", "call-title");
    title.textContent = call.title;
    head.append(title);

    if (call.target !== null) {
      const target = el("code", "call-target");
      target.textContent = call.target;
      head.append(target);
    }

    card.append(head);
    const labelled = call.fields.length > 1;

    for (const field of call.fields) {
      const row = el("div", `call-field ${FIELD_TONE[field.label] ?? ""}`.trim());

      if (labelled || !field.code) {
        const label = el("span", "call-label");
        label.textContent = field.label;
        row.append(label);
      }

      if (field.code) {
        const pre = el("pre");
        pre.textContent = field.text;
        row.append(pre);
      } else {
        const value = el("span", "call-value");
        value.textContent = field.text;
        row.append(value);
      }

      card.append(row);
    }

    return card;
  });
}
