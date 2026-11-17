import assert from "node:assert/strict";
import { test } from "node:test";

import { parseInline, parseMarkdown, safeHref } from "../src/markdown.ts";
import type { Inline, MdBlock } from "../src/markdown.ts";

const text = (value: string): Inline => ({ kind: "text", text: value });
const para = (...inlines: Inline[]): MdBlock => ({ kind: "paragraph", inlines });

// --- blocks ---------------------------------------------------------------------------

test("plain text is one paragraph; a blank line starts the next", () => {
  assert.deepEqual(parseMarkdown("one\ntwo\n\nthree"), [para(text("one\ntwo")), para(text("three"))]);
});

test("empty and blank input give no blocks", () => {
  assert.deepEqual(parseMarkdown(""), []);
  assert.deepEqual(parseMarkdown("  \n\n "), []);
});

test("headings carry their level and inline content", () => {
  assert.deepEqual(parseMarkdown("# Title\n### Sub **bold** ##"), [
    { kind: "heading", level: 1, inlines: [text("Title")] },
    { kind: "heading", level: 3, inlines: [text("Sub "), { kind: "strong", children: [text("bold")] }] },
  ]);
});

test("a hash without a space is not a heading", () => {
  assert.deepEqual(parseMarkdown("#hashtag"), [para(text("#hashtag"))]);
});

test("fenced code keeps its language and text exactly", () => {
  assert.deepEqual(parseMarkdown("Run:\n```python\nprint('**x**')\n  indented\n```\nDone."), [
    para(text("Run:")),
    { kind: "code", lang: "python", text: "print('**x**')\n  indented", open: false },
    para(text("Done.")),
  ]);
});

test("an unclosed fence is code to the end, marked open (still streaming)", () => {
  assert.deepEqual(parseMarkdown("```ts\nconst a = 1;\n"), [{ kind: "code", lang: "ts", text: "const a = 1;", open: true }]);
  assert.deepEqual(parseMarkdown("```"), [{ kind: "code", lang: "", text: "", open: true }]);
});

test("tilde fences and a longer closing fence work", () => {
  assert.deepEqual(parseMarkdown("~~~\na\n~~~~"), [{ kind: "code", lang: "", text: "a", open: false }]);
});

test("bullet lists, tight", () => {
  assert.deepEqual(parseMarkdown("- one\n* two\n+ three"), [
    {
      kind: "list",
      ordered: false,
      start: 1,
      items: [[para(text("one"))], [para(text("two"))], [para(text("three"))]],
    },
  ]);
});

test("ordered lists keep their start number", () => {
  const [list] = parseMarkdown("3. c\n4) d");
  assert.equal(list?.kind, "list");
  assert.ok(list?.kind === "list");
  assert.equal(list.ordered, true);
  assert.equal(list.start, 3);
  assert.equal(list.items.length, 2);
});

test("nested lists by indentation", () => {
  assert.deepEqual(parseMarkdown("1. outer\n   - inner a\n   - inner b\n2. next"), [
    {
      kind: "list",
      ordered: true,
      start: 1,
      items: [
        [
          para(text("outer")),
          { kind: "list", ordered: false, start: 1, items: [[para(text("inner a"))], [para(text("inner b"))]] },
        ],
        [para(text("next"))],
      ],
    },
  ]);
});

test("a two-space nested list under a numbered item still nests", () => {
  const [list] = parseMarkdown("1. outer\n  - inner\n2. next");
  assert.ok(list?.kind === "list");
  assert.equal(list.items.length, 2);
  assert.equal(list.items[0]?.[1]?.kind, "list");
});

test("list items survive blank lines between them and keep code blocks", () => {
  const [list, after] = parseMarkdown("- a\n\n- b\n\n  ```\n  x\n  ```\n\nafter");
  assert.ok(list?.kind === "list");
  assert.equal(list.items.length, 2);
  assert.deepEqual(list.items[1], [para(text("b")), { kind: "code", lang: "", text: "x", open: false }]);
  assert.deepEqual(after, para(text("after")));
});

test("a lazy line continues the list item's paragraph", () => {
  assert.deepEqual(parseMarkdown("- first line\nsecond line"), [
    { kind: "list", ordered: false, start: 1, items: [[para(text("first line\nsecond line"))]] },
  ]);
});

test("a bullet line interrupts a paragraph", () => {
  const result = parseMarkdown("Steps:\n- a\n- b");
  assert.deepEqual(result[0], para(text("Steps:")));
  assert.equal(result[1]?.kind, "list");
});

test("a number that is not 1 does not interrupt a paragraph", () => {
  assert.deepEqual(parseMarkdown("It was\n2024. Then"), [para(text("It was\n2024. Then"))]);
});

test("block quotes parse their inside as markdown", () => {
  assert.deepEqual(parseMarkdown("> **Note**\n> second"), [
    { kind: "quote", blocks: [para({ kind: "strong", children: [text("Note")] }, text("\nsecond"))] },
  ]);
});

test("rules", () => {
  assert.deepEqual(parseMarkdown("a\n\n---\n\nb"), [para(text("a")), { kind: "rule" }, para(text("b"))]);
  assert.deepEqual(parseMarkdown("* * *"), [{ kind: "rule" }]);
});

test("tables: header, alignment and rows (short rows padded, long rows cut)", () => {
  assert.deepEqual(parseMarkdown("| Name | Size |\n|:-----|-----:|\n| a | `1` |\n| b |\n| c | 3 | extra |"), [
    {
      kind: "table",
      align: ["left", "right"],
      header: [[text("Name")], [text("Size")]],
      rows: [
        [[text("a")], [{ kind: "code", text: "1" }]],
        [[text("b")], []],
        [[text("c")], [text("3")]],
      ],
    },
  ]);
});

test("a pipe line without a separator row is just a paragraph", () => {
  assert.deepEqual(parseMarkdown("a | b\nc | d"), [para(text("a | b\nc | d"))]);
});

test("escaped pipes stay inside a table cell", () => {
  const [table] = parseMarkdown("| a \\| b | c |\n|---|---|");
  assert.ok(table?.kind === "table");
  assert.deepEqual(table.header[0], [text("a | b")]);
});

test("tabs count as four spaces for nesting", () => {
  const [list] = parseMarkdown("- outer\n\t- inner");
  assert.ok(list?.kind === "list");
  assert.equal(list.items[0]?.[1]?.kind, "list");
});

test("very deep nesting does not blow the stack", () => {
  const deep = ">".repeat(5000) + " x";
  assert.doesNotThrow(() => parseMarkdown(deep));
  const nested = Array.from({ length: 2000 }, (_, i) => " ".repeat(i * 2) + "- x").join("\n");
  assert.doesNotThrow(() => parseMarkdown(nested));
});

// --- inline ---------------------------------------------------------------------------

test("strong, emphasis and strikethrough", () => {
  assert.deepEqual(parseInline("**b** *i* __b2__ _i2_ ~~s~~"), [
    { kind: "strong", children: [text("b")] },
    text(" "),
    { kind: "em", children: [text("i")] },
    text(" "),
    { kind: "strong", children: [text("b2")] },
    text(" "),
    { kind: "em", children: [text("i2")] },
    text(" "),
    { kind: "del", children: [text("s")] },
  ]);
});

test("emphasis nests", () => {
  assert.deepEqual(parseInline("**bold *and italic***"), [
    { kind: "strong", children: [text("bold "), { kind: "em", children: [text("and italic")] }] },
  ]);
});

test("underscores inside words are not emphasis", () => {
  assert.deepEqual(parseInline("snake_case_name and my__dunder__x"), [text("snake_case_name and my__dunder__x")]);
});

test("stars around spaces or alone are literal", () => {
  assert.deepEqual(parseInline("2 * 3 * 4"), [text("2 * 3 * 4")]);
  assert.deepEqual(parseInline("** not bold**"), [text("** not bold**")]);
});

test("unclosed markers stay literal (half-streamed text)", () => {
  assert.deepEqual(parseInline("**bol"), [text("**bol")]);
  assert.deepEqual(parseInline("see `cod"), [text("see `cod")]);
  assert.deepEqual(parseInline("[link](exa"), [text("[link](exa")]);
  // The half-written URL still reads as a link.
  assert.deepEqual(parseInline("[link](http://exa"), [
    text("[link]("),
    { kind: "link", href: "http://exa", children: [text("http://exa")] },
  ]);
});

test("code spans are literal and may use longer backtick runs", () => {
  assert.deepEqual(parseInline("`**x**` and ``a ` b``"), [
    { kind: "code", text: "**x**" },
    text(" and "),
    { kind: "code", text: "a ` b" },
  ]);
});

test("backslash escapes punctuation", () => {
  assert.deepEqual(parseInline("\\*not em\\* \\`x\\`"), [text("*not em* `x`")]);
});

test("links keep safe targets", () => {
  assert.deepEqual(parseInline("see [the docs](https://example.com/a_b?x=1)."), [
    text("see "),
    { kind: "link", href: "https://example.com/a_b?x=1", children: [text("the docs")] },
    text("."),
  ]);
});

test("unsafe link targets become plain text", () => {
  assert.deepEqual(parseInline("[click](javascript:alert(1))"), [text("click")]);
  assert.deepEqual(parseInline("[x](data:text/html,hi)"), [text("x")]);
  assert.deepEqual(parseInline("[x](/relative)"), [text("x")]);
});

test("images become links to the image, never loaded", () => {
  assert.deepEqual(parseInline("![a cat](https://example.com/cat.png)"), [
    { kind: "link", href: "https://example.com/cat.png", children: [text("a cat")] },
  ]);
});

test("bare and angle-bracket URLs become links, trailing punctuation left out", () => {
  assert.deepEqual(parseInline("Go to https://example.com/x. Or <http://a.io/b>"), [
    text("Go to "),
    { kind: "link", href: "https://example.com/x", children: [text("https://example.com/x")] },
    text(". Or "),
    { kind: "link", href: "http://a.io/b", children: [text("http://a.io/b")] },
  ]);
  const wiki = "https://en.wikipedia.org/wiki/A_(b)";
  assert.deepEqual(parseInline(`(see ${wiki})`), [
    text("(see "),
    { kind: "link", href: wiki, children: [text(wiki)] },
    text(")"),
  ]);
});

test("safeHref allows only http(s) and mailto", () => {
  assert.equal(safeHref("https://a.b/c"), "https://a.b/c");
  assert.equal(safeHref("HTTP://A.B"), "HTTP://A.B");
  assert.equal(safeHref("mailto:a@b.c"), "mailto:a@b.c");
  assert.equal(safeHref(" javascript:alert(1)"), null);
  assert.equal(safeHref("java\nscript:x"), null);
  assert.equal(safeHref("file:///etc/passwd"), null);
  assert.equal(safeHref("//evil.com"), null);
});

test("adjacent text pieces are merged", () => {
  assert.deepEqual(parseInline("a * b"), [text("a * b")]);
});
