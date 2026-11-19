// Reading the launch token and the selected session from the page address.

const SESSION_ID = /^[0-9a-f]{12}$/;

/** The token from the query string (preferred: it is from the latest launch) or the stored
 * one, and the query string without it, to put back into the address bar. */
export function tokenFrom(search: string, stored: string | null): { token: string; search: string } {
  const params = new URLSearchParams(search);
  const fromQuery = params.get("token") ?? "";
  params.delete("token");
  const rest = params.toString();
  return { token: fromQuery || stored || "", search: rest ? `?${rest}` : "" };
}

/** A plain http(s) origin ("http://127.0.0.1:8000"), or null for anything else (paths, users,
 * queries, other schemes): the page sends its token there, so nothing looser is accepted. */
function apiOrigin(value: string | null): string | null {
  if (!value) return null;

  let url: URL;

  try {
    url = new URL(value);
  } catch {
    return null;
  }

  const plain = url.pathname === "/" && !url.search && !url.hash && !url.username && !url.password;
  return (url.protocol === "http:" || url.protocol === "https:") && plain && !value.includes("?") ? url.origin : null;
}

/** The agent API's address from the query string (`?api=`, from the backend's launch link) or the
 * stored one, and the query string without it. "" means the page's own origin. */
export function apiFrom(search: string, stored: string | null): { api: string; search: string } {
  const params = new URLSearchParams(search);
  const fromQuery = apiOrigin(params.get("api"));
  params.delete("api");
  const rest = params.toString();
  return { api: fromQuery ?? apiOrigin(stored) ?? "", search: rest ? `?${rest}` : "" };
}

/** The session id in a `#<id>` hash, or null. */
export function sessionFromHash(hash: string): string | null {
  const id = hash.startsWith("#") ? hash.slice(1) : hash;
  return SESSION_ID.test(id) ? id : null;
}
