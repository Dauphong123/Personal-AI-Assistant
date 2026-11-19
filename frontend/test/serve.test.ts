import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, before, test } from "node:test";

import { contentSecurityPolicy, parseArgs, startServer, staticFile } from "../scripts/serve.ts";
import type { Server } from "node:http";

const API = "http://127.0.0.1:8000";

let dist = "";
let server: Server;
let port = 0;

before(async () => {
  const root = mkdtempSync(join(tmpdir(), "frontend-serve-"));
  dist = join(root, "dist");
  mkdirSync(join(dist, "lib"), { recursive: true });
  writeFileSync(join(dist, "index.html"), "<html>page</html>");
  writeFileSync(join(dist, "main.js"), "// js");
  writeFileSync(join(dist, "style.css"), "/* css */");
  writeFileSync(join(dist, "lib", "api.js"), "// api");
  writeFileSync(join(dist, "notes.txt"), "not served");
  writeFileSync(join(root, "outside.js"), "outside");
  server = await startServer({ dist, api: API, host: "127.0.0.1", port: 0 });
  port = (server.address() as AddressInfo).port;
});

after(() => {
  server.close();
  rmSync(dirname(dist), { recursive: true, force: true });
});

interface Answer {
  status: number;
  headers: Record<string, string | string[] | undefined>;
  body: string;
}

function get(path: string, options: { method?: string; host?: string } = {}): Promise<Answer> {
  return new Promise((resolve, reject) => {
    const req = request(
      { host: "127.0.0.1", port, path, method: options.method ?? "GET", headers: { Host: options.host ?? `127.0.0.1:${port}` } },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => (body += chunk));
        res.on("end", () => resolve({ status: res.statusCode ?? 0, headers: res.headers, body }));
      },
    );
    req.on("error", reject);
    req.end();
  });
}

test("the page is served with a CSP that only lets it reach this origin and the agent API", async () => {
  const answer = await get("/");
  assert.equal(answer.status, 200);
  assert.equal(answer.body, "<html>page</html>");
  assert.match(String(answer.headers["content-type"]), /text\/html/);
  const csp = String(answer.headers["content-security-policy"]);
  assert.match(csp, /connect-src 'self' http:\/\/127\.0\.0\.1:8000;/);
  assert.match(csp, /script-src 'self';/);
  assert.doesNotMatch(csp, /unsafe-inline/);
  assert.equal(answer.headers["x-content-type-options"], "nosniff");
  assert.equal(answer.headers["referrer-policy"], "no-referrer");
  assert.equal(answer.headers["cache-control"], "no-store");
});

test("the launch link's query string does not change what is served", async () => {
  const answer = await get(`/?api=${encodeURIComponent(API)}&token=abc`);
  assert.equal(answer.status, 200);
  assert.equal(answer.body, "<html>page</html>");
});

test("built files are served with their type", async () => {
  assert.match(String((await get("/main.js")).headers["content-type"]), /text\/javascript/);
  assert.match(String((await get("/style.css")).headers["content-type"]), /text\/css/);
  assert.equal((await get("/lib/api.js")).status, 200);
});

test("unknown types, missing files, hidden names and traversal are 404", async () => {
  for (const path of ["/notes.txt", "/nope.js", "/lib", "/.hidden.js", "/../outside.js", "/%2e%2e/outside.js",
    "/lib%2f..%2f..%2foutside.js", "/..%5coutside.js"]) {
    assert.equal((await get(path)).status, 404, path);
  }
});

test("only GET and HEAD", async () => {
  assert.equal((await get("/", { method: "POST" })).status, 405);
  const head = await get("/", { method: "HEAD" });
  assert.equal(head.status, 200);
  assert.equal(head.body, "");
});

test("a foreign Host header is refused (DNS rebinding)", async () => {
  assert.equal((await get("/", { host: "evil.example" })).status, 403);
  assert.equal((await get("/", { host: `localhost:${port}` })).status, 200);
});

test("staticFile keeps every path inside the build directory", () => {
  assert.equal(staticFile(dist, "/"), join(dist, "index.html"));
  assert.equal(staticFile(dist, "/lib/api.js"), join(dist, "lib", "api.js"));
  assert.equal(staticFile(dist, "/../outside.js"), null);
  assert.equal(staticFile(dist, "/a\u0000.js"), null);
  assert.equal(staticFile(dist, "/%E0%A4%A.js"), null); // bad escape
});

test("a missing build explains how to build", async () => {
  const unbuilt = await startServer({ dist: join(dist, "absent"), api: API, host: "127.0.0.1", port: 0 });
  const saved = port;
  port = (unbuilt.address() as AddressInfo).port;

  try {
    const answer = await get("/");
    assert.equal(answer.status, 503);
    assert.match(answer.body, /npm --prefix frontend run build/);
  } finally {
    port = saved;
    unbuilt.close();
  }
});

test("the CSP names the API origin in connect-src only", () => {
  const csp = contentSecurityPolicy("http://localhost:9000");
  assert.match(csp, /connect-src 'self' http:\/\/localhost:9000;/);
  assert.match(csp, /media-src 'self' blob:;/);
  assert.match(csp, /frame-ancestors 'none'/);
});

test("parseArgs: loopback, port 5173 and the backend's default API address", () => {
  assert.deepEqual(parseArgs([]), { host: "127.0.0.1", port: 5173, api: "http://127.0.0.1:8000" });
  assert.deepEqual(parseArgs(["--port", "3000", "--api", "http://localhost:9000/", "--host", "localhost"]), {
    host: "localhost",
    port: 3000,
    api: "http://localhost:9000",
  });
});

test("parseArgs refuses a bad API address, port, host or option", () => {
  assert.throws(() => parseArgs(["--api", "http://x/app"]), /--api/);
  assert.throws(() => parseArgs(["--api", "javascript:alert(1)"]), /--api/);
  assert.throws(() => parseArgs(["--port", "70000"]), /--port/);
  assert.throws(() => parseArgs(["--host", "0.0.0.0"]), /loopback/);
  assert.throws(() => parseArgs(["--nope"]), /--nope/);
});
