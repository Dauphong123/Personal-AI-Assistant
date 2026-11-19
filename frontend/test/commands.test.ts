// Command palette and slash commands: matching and parsing, without the DOM.
import assert from "node:assert/strict";
import { test } from "node:test";

import { parseSlash, rank, score, SLASH_COMMANDS, slashSuggestions } from "../src/commands.ts";
import type { PaletteItem } from "../src/commands.ts";

const items: PaletteItem[] = [
  { id: "new", title: "New session" },
  { id: "rename", title: "Rename session" },
  { id: "clear", title: "Clear conversation", keywords: "reset wipe" },
  { id: "theme", title: "Switch theme", keywords: "dark light" },
  { id: "go:1", title: "Database migration notes", detail: "Session" },
];

test("score: prefix beats word start beats substring beats scattered letters; no match is -1", () => {
  const prefix = score("ren", "Rename session");
  const word = score("ses", "Rename session");
  const inside = score("nam", "Rename session");
  const scattered = score("rns", "Rename session");
  assert.ok(prefix > word && word > inside && inside > scattered && scattered > 0, `${prefix} ${word} ${inside} ${scattered}`);
  assert.equal(score("xyz", "Rename session"), -1);
  assert.equal(score("REN", "rename"), score("ren", "Rename"), "case does not matter");
});

test("rank: best matches first, keywords count, no query keeps the order", () => {
  assert.deepEqual(rank("", items).map((item) => item.id), ["new", "rename", "clear", "theme", "go:1"]);
  assert.deepEqual(rank("se", items).map((item) => item.id).slice(0, 2), ["new", "rename"]);
  assert.deepEqual(rank("dark", items).map((item) => item.id), ["theme"]);
  assert.deepEqual(rank("migr", items).map((item) => item.id), ["go:1"]);
  assert.deepEqual(rank("zzz", items), []);
});

test("rank: equal scores keep their original order", () => {
  const same: PaletteItem[] = [
    { id: "a", title: "Session one" },
    { id: "b", title: "Session two" },
  ];
  assert.deepEqual(rank("session", same).map((item) => item.id), ["a", "b"]);
});

test("parseSlash: known commands only; anything else is an ordinary message", () => {
  assert.deepEqual(parseSlash("/new"), { name: "new", arg: "" });
  assert.deepEqual(parseSlash("  /CLEAR  "), { name: "clear", arg: "" });
  assert.deepEqual(parseSlash("/rename Release plan  "), { name: "rename", arg: "Release plan" });
  assert.deepEqual(parseSlash("/rename"), { name: "rename", arg: "" });
  assert.equal(parseSlash("/voice what is in this folder?"), null, "a project path is a question");
  assert.equal(parseSlash("/clear the build cache"), null, "a command that takes nothing, with words after it");
  assert.equal(parseSlash("explain /new"), null);
  assert.equal(parseSlash("/"), null);
});

test("slashSuggestions: while the first word is being typed", () => {
  assert.deepEqual(slashSuggestions("/").map((c) => c.name), SLASH_COMMANDS.map((c) => c.name));
  assert.deepEqual(slashSuggestions("/re").map((c) => c.name), ["rename"]);
  assert.deepEqual(slashSuggestions("/RE").map((c) => c.name), ["rename"]);
  assert.deepEqual(slashSuggestions("/rename x"), []);
  assert.deepEqual(slashSuggestions("hello"), []);
  assert.deepEqual(slashSuggestions("/zz"), []);
});

test("every slash command has a usage line and help", () => {
  for (const command of SLASH_COMMANDS) {
    assert.ok(command.usage.startsWith(`/${command.name}`), command.name);
    assert.ok(command.help.length > 0, command.name);
  }
});
