import assert from "node:assert/strict";
import { test } from "node:test";

import { sessionFromHash, tokenFrom } from "../src/url.ts";

test("token comes from the query and is removed from it", () => {
  assert.deepEqual(tokenFrom("?token=abc&x=1", null), { token: "abc", search: "?x=1" });
  assert.deepEqual(tokenFrom("?token=abc", null), { token: "abc", search: "" });
});

test("the query token wins over a stored one (a new server launch)", () => {
  assert.deepEqual(tokenFrom("?token=new", "old"), { token: "new", search: "" });
});

test("without a query token the stored one is used", () => {
  assert.deepEqual(tokenFrom("", "kept"), { token: "kept", search: "" });
});

test("no token anywhere is an empty token", () => {
  assert.deepEqual(tokenFrom("", null), { token: "", search: "" });
  assert.deepEqual(tokenFrom("?token=", null), { token: "", search: "" });
});

test("session id comes from the hash only when it looks like one", () => {
  assert.equal(sessionFromHash("#abcdef012345"), "abcdef012345");
  assert.equal(sessionFromHash("abcdef012345"), "abcdef012345");
  assert.equal(sessionFromHash("#nope"), null);
  assert.equal(sessionFromHash(""), null);
  assert.equal(sessionFromHash("#ABCDEF012345"), null);
});
