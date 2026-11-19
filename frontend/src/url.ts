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

/** The session id in a `#<id>` hash, or null. */
export function sessionFromHash(hash: string): string | null {
  const id = hash.startsWith("#") ? hash.slice(1) : hash;
  return SESSION_ID.test(id) ? id : null;
}
