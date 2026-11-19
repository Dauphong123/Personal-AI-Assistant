import assert from "node:assert/strict";
import { test } from "node:test";

import { nextTheme, parseTheme, themeLabel } from "../src/theme.ts";

test("a stored choice is read back; anything else means follow the system", () => {
  assert.equal(parseTheme("light"), "light");
  assert.equal(parseTheme("dark"), "dark");
  assert.equal(parseTheme("system"), "system");
  assert.equal(parseTheme(null), "system");
  assert.equal(parseTheme("purple"), "system");
});

test("the button cycles system, light, dark", () => {
  assert.equal(nextTheme("system"), "light");
  assert.equal(nextTheme("light"), "dark");
  assert.equal(nextTheme("dark"), "system");
});

test("the label says what is in use", () => {
  assert.equal(themeLabel("system"), "Theme: system");
  assert.equal(themeLabel("light"), "Theme: light");
  assert.equal(themeLabel("dark"), "Theme: dark");
});
