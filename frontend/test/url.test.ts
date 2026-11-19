import assert from "node:assert/strict";
import { test } from "node:test";

import { apiFrom, sessionFromHash, tokenFrom } from "../src/url.ts";

test("the API address comes from the query and is removed from it", () => {
  assert.deepEqual(apiFrom("?api=http://127.0.0.1:8000&x=1", null), { api: "http://127.0.0.1:8000", search: "?x=1" });
  assert.deepEqual(apiFrom("?api=http%3A%2F%2Flocalhost%3A8000%2F", null), { api: "http://localhost:8000", search: "" });
});

test("the query API wins over a stored one, which is used otherwise", () => {
  assert.deepEqual(apiFrom("?api=http://127.0.0.1:9000", "http://127.0.0.1:8000"), { api: "http://127.0.0.1:9000", search: "" });
  assert.deepEqual(apiFrom("", "http://127.0.0.1:8000"), { api: "http://127.0.0.1:8000", search: "" });
});

test("no API address means the page's own origin", () => {
  assert.deepEqual(apiFrom("", null), { api: "", search: "" });
});

test("anything but a plain http(s) origin is ignored", () => {
  for (const bad of ["javascript:alert(1)", "ftp://x", "http://x/app", "http://u:p@x", "http://x?a=1", "nope", "//x"]) {
    assert.deepEqual(apiFrom(`?${new URLSearchParams({ api: bad }).toString()}`, null), { api: "", search: "" }, bad);
    assert.deepEqual(apiFrom(`?${new URLSearchParams({ api: bad }).toString()}`, "http://127.0.0.1:8000").api, "http://127.0.0.1:8000", bad);
  }
  assert.equal(apiFrom("", "javascript:alert(1)").api, "");
});

test("token and API address are both taken out of the launch link", () => {
  const launch = tokenFrom("?api=http://127.0.0.1:8000&token=abc", null);
  assert.equal(launch.token, "abc");
  assert.deepEqual(apiFrom(launch.search, null), { api: "http://127.0.0.1:8000", search: "" });
});

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
