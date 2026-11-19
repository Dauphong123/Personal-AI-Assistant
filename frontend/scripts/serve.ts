// The frontend's own web server: serves the built page (dist/) on its own origin, separate from
// the agent API (python -m backend), which the page reaches only over HTTP. Node built-ins only.
//
//   npm --prefix frontend start [-- --api http://127.0.0.1:8000] [--port 5173] [--host 127.0.0.1]
//
// The page carries the API token, so this server is loopback only, checks the Host header (DNS
// rebinding), serves only known file types from dist/ and sends a CSP that lets the page connect
// to this origin and the agent API origin (--api) and nowhere else.

import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { apiOrigin } from "../src/url.ts";

export const DEFAULT_PORT = 5173; // backend/__main__.py DEFAULT_FRONTEND_ORIGINS uses this port
export const DEFAULT_API = "http://127.0.0.1:8000"; // python -m backend's default address
export const DIST = fileURLToPath(new URL("../dist/", import.meta.url));

const LOOPBACK = new Set(["127.0.0.1", "localhost"]);
const BUILD_HINT = "The frontend is not built. Run: npm --prefix frontend install && npm --prefix frontend run build";
const TEXT = "text/plain; charset=utf-8";

// Only these file types are served (as in backend/server.py).
const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

export interface Options {
  host: string;
  port: number;
  api: string;
}

/** Command-line options; throws an Error naming the bad option. */
export function parseArgs(argv: readonly string[]): Options {
  const options: Options = { host: "127.0.0.1", port: DEFAULT_PORT, api: DEFAULT_API };

  for (let i = 0; i < argv.length; i += 2) {
    const name = argv[i];
    const value = argv[i + 1];

    if (name === "--api") {
      const origin = apiOrigin(value ?? null);

      if (origin === null) {
        throw new Error(`--api must be the agent API's origin, e.g. ${DEFAULT_API} (got ${String(value)})`);
      }

      options.api = origin;
    } else if (name === "--port") {
      const port = Number(value);

      if (!Number.isInteger(port) || port < 0 || port > 65535) {
        throw new Error(`--port must be a number from 0 to 65535 (got ${String(value)})`);
      }

      options.port = port;
    } else if (name === "--host") {
      if (value === undefined || !LOOPBACK.has(value)) {
        throw new Error("--host must be a loopback address (127.0.0.1 or localhost): the page carries the API token");
      }

      options.host = value;
    } else {
      throw new Error(`Unknown option ${String(name)}. Options: --api ORIGIN, --port N, --host 127.0.0.1|localhost`);
    }
  }

  return options;
}

/** The page may load only its own files and connect only to itself and the agent API. */
export function contentSecurityPolicy(api: string): string {
  return (
    `default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self' ${api}; ` +
    "img-src 'self' data:; media-src 'self' blob:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
  );
}

/** The build file a URL path names, or null. Hidden names, '..', backslashes, bad escapes and
 * unlisted file types are refused; the real path (links followed) must stay inside dist. */
export function staticFile(dist: string, urlPath: string): string | null {
  let path: string;

  try {
    path = decodeURIComponent(urlPath.split("?")[0] ?? "");
  } catch {
    return null;
  }

  if (path.includes("\\") || path.includes("\0")) return null;

  const parts = (path.replace(/^\/+/, "") || "index.html").split("/");

  if (parts.some((part) => !part || part.startsWith("."))) return null;

  if (!Object.hasOwn(CONTENT_TYPES, extname(parts[parts.length - 1] ?? "").toLowerCase())) return null;

  const file = resolve(dist, ...parts);

  try {
    const root = realpathSync(dist);
    const real = realpathSync(file);

    if (!real.startsWith(root + sep) || !statSync(real).isFile()) return null;
  } catch {
    return null; // missing
  }

  return file;
}

function handle(req: IncomingMessage, res: ServerResponse, dist: string, api: string, hosts: Set<string>): void {
  const send = (status: number, body: string | Buffer, type: string): void => {
    res.writeHead(status, {
      "Content-Type": type,
      "Content-Length": Buffer.byteLength(body),
      "Content-Security-Policy": contentSecurityPolicy(api),
      "X-Content-Type-Options": "nosniff",
      "Cache-Control": "no-store",
      "Referrer-Policy": "no-referrer",
    });
    res.end(req.method === "HEAD" ? undefined : body);
  };

  if (!hosts.has(req.headers.host ?? "")) return send(403, "Forbidden host", TEXT);

  if (req.method !== "GET" && req.method !== "HEAD") {
    res.setHeader("Allow", "GET, HEAD");
    return send(405, "Use GET", TEXT);
  }

  if (!existsSync(dist)) return send(503, BUILD_HINT, TEXT);

  const file = staticFile(dist, req.url ?? "/");

  if (file === null) return send(404, "Not found", TEXT);

  let body: Buffer;

  try {
    body = readFileSync(file);
  } catch {
    return send(404, "Not found", TEXT);
  }

  send(200, body, CONTENT_TYPES[extname(file).toLowerCase()] ?? TEXT);
}

/** Listen on host:port (0 = any free port) and serve dist; resolves once listening. */
export function startServer(options: Options & { dist: string }): Promise<Server> {
  if (!LOOPBACK.has(options.host)) {
    return Promise.reject(new Error("The frontend server listens on loopback only (127.0.0.1 or localhost)."));
  }

  let hosts = new Set<string>();
  const server = createServer((req, res) => handle(req, res, options.dist, options.api, hosts));

  return new Promise((resolveStart, rejectStart) => {
    server.once("error", rejectStart);
    server.listen(options.port, options.host, () => {
      server.off("error", rejectStart);
      const { port } = server.address() as AddressInfo;
      hosts = new Set([...LOOPBACK].map((name) => `${name}:${port}`));
      resolveStart(server);
    });
  });
}

async function main(): Promise<void> {
  let options: Options;

  try {
    options = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error((error as Error).message);
    process.exit(2);
  }

  try {
    await startServer({ ...options, dist: DIST });
  } catch (error) {
    console.error(`Could not listen on ${options.host}:${options.port} (${(error as Error).message}). Try another --port.`);
    process.exit(1);
  }

  console.log(`Frontend on http://${options.host}:${options.port} (agent API: ${options.api})`);
  console.log("Open the link that python -m backend prints: it carries the API address and the token.");

  if (!existsSync(join(DIST, "index.html"))) console.log(BUILD_HINT);
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  void main();
}
