// Command palette (Ctrl+K) and slash commands in the message box: matching and parsing only,
// no DOM, so it runs under node --test. main.ts builds the items and runs what is chosen.

export interface PaletteItem {
  id: string; // what main.ts runs: an action name, or "go:<session id>"
  title: string;
  detail?: string; // shown faintly on the right ("Session", "Ctrl+K")
  keywords?: string; // other words that should find it
}

/** How well `query` matches `text`: -1 = not at all; a start-of-text match scores highest,
 * then the start of a word, then anywhere, then the letters in order with gaps. */
export function score(query: string, text: string): number {
  const q = query.trim().toLowerCase();
  const t = text.toLowerCase();

  if (!q) return 0;
  if (t.startsWith(q)) return 400 - t.length;

  const at = t.indexOf(q);

  if (at > 0 && /[\s\-_/.:]/.test(t[at - 1] ?? "")) return 300 - at;
  if (at > 0) return 200 - at;

  // Letters in order: "rns" finds "Rename session". Fewer gaps score higher.
  let position = -1;
  let gaps = 0;

  for (const char of q) {
    const next = t.indexOf(char, position + 1);
    if (next === -1) return -1;
    gaps += next - position - 1;
    position = next;
  }

  return Math.max(1, 100 - gaps);
}

/** Items matching `query`, best first (ties keep their order); all of them when it is empty. */
export function rank(query: string, items: readonly PaletteItem[]): PaletteItem[] {
  if (!query.trim()) return [...items];

  return items
    .map((item, index) => {
      const own = score(query, item.title);
      const other = item.keywords ? score(query, item.keywords) : -1;
      return { item, index, best: Math.max(own, other > 0 ? Math.floor(other * 0.9) : other) };
    })
    .filter((entry) => entry.best > 0)
    .sort((a, b) => b.best - a.best || a.index - b.index)
    .map((entry) => entry.item);
}

export interface SlashCommand {
  name: string;
  usage: string;
  help: string;
  takesArg: boolean;
}

export const SLASH_COMMANDS: readonly SlashCommand[] = [
  { name: "new", usage: "/new", help: "Start a new session", takesArg: false },
  { name: "rename", usage: "/rename <name>", help: "Rename this session", takesArg: true },
  { name: "clear", usage: "/clear", help: "Clear this conversation", takesArg: false },
  { name: "delete", usage: "/delete", help: "Delete this session", takesArg: false },
  { name: "stop", usage: "/stop", help: "Stop the running answer", takesArg: false },
  { name: "model", usage: "/model [name] [effort]", help: "Show or switch this session's model", takesArg: true },
  { name: "theme", usage: "/theme", help: "Switch between system, light and dark", takesArg: false },
];

/** A slash command typed in the message box, or null for an ordinary message. Only the names
 * above count, so "/voice what is here?" (a project path) is still a question, and so is a
 * command that takes nothing followed by words ("/clear the build cache"). */
export function parseSlash(text: string): { name: string; arg: string } | null {
  const match = /^\/([a-z]+)(?:\s+([\s\S]*))?$/i.exec(text.trim());

  if (!match) return null;

  const name = (match[1] ?? "").toLowerCase();
  const arg = (match[2] ?? "").trim();
  const command = SLASH_COMMANDS.find((c) => c.name === name);

  if (!command || (!command.takesArg && arg)) return null;

  return { name, arg };
}

/** Commands to suggest while the first word after "/" is being typed. */
export function slashSuggestions(text: string): SlashCommand[] {
  const match = /^\/([a-z]*)$/i.exec(text);

  if (!match) return [];

  const typed = (match[1] ?? "").toLowerCase();
  return SLASH_COMMANDS.filter((c) => c.name.startsWith(typed));
}
