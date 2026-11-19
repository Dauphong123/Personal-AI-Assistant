// Light, dark or follow the system. The choice is kept in localStorage; the page sets
// data-theme on <html> (style.css has the colours for each).

export type Theme = "system" | "light" | "dark";

const ORDER: readonly Theme[] = ["system", "light", "dark"];

export function parseTheme(stored: string | null): Theme {
  return ORDER.find((theme) => theme === stored) ?? "system";
}

export function nextTheme(current: Theme): Theme {
  return ORDER[(ORDER.indexOf(current) + 1) % ORDER.length] ?? "system";
}

export function themeLabel(theme: Theme): string {
  return `Theme: ${theme}`;
}
