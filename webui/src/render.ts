// Markdown tree (markdown.ts) -> DOM. Text only ever goes in through textContent or text
// nodes; link targets were checked by safeHref.

import type { Inline, MdBlock } from "./markdown.ts";

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);

  if (className) node.className = className;

  return node;
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
        const pre = el("pre", block.open ? "code-block open" : "code-block");
        const code = el("code");

        if (block.lang) pre.dataset["lang"] = block.lang;

        code.textContent = block.text;
        pre.append(code);
        return pre;
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
