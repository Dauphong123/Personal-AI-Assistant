import assert from "node:assert/strict";
import { test } from "node:test";

import { shortcut } from "../src/keys.ts";
import type { KeyContext } from "../src/keys.ts";

const base: KeyContext = {
  key: "Escape",
  ctrl: false,
  alt: false,
  meta: false,
  typing: false,
  menuOpen: false,
  drawerOpen: false,
  dialogOpen: false,
  renaming: false,
  busy: false,
};

test("Escape closes the innermost thing first", () => {
  assert.equal(shortcut({ ...base, menuOpen: true, drawerOpen: true, busy: true }), "close-menu");
  assert.equal(shortcut({ ...base, drawerOpen: true, busy: true }), "close-drawer");
  assert.equal(shortcut({ ...base, busy: true }), "stop");
  assert.equal(shortcut(base), null);
});

test("Escape is left alone by an open dialog and while renaming", () => {
  // The dialog closes itself on Escape; the rename box cancels itself.
  assert.equal(shortcut({ ...base, dialogOpen: true, busy: true }), null);
  assert.equal(shortcut({ ...base, renaming: true, busy: true }), null);
});

test("Escape stops a turn even from the message box", () => {
  assert.equal(shortcut({ ...base, typing: true, busy: true }), "stop");
});

test("slash focuses the message box unless typing or modified", () => {
  assert.equal(shortcut({ ...base, key: "/" }), "focus-input");
  assert.equal(shortcut({ ...base, key: "/", typing: true }), null);
  assert.equal(shortcut({ ...base, key: "/", ctrl: true }), null);
  assert.equal(shortcut({ ...base, key: "/", dialogOpen: true }), null);
});

test("other keys do nothing", () => {
  assert.equal(shortcut({ ...base, key: "a" }), null);
  assert.equal(shortcut({ ...base, key: "Enter", busy: true }), null);
});
