// Keyboard shortcuts, decided without the DOM so they can be tested.
// Escape closes the innermost open thing, else stops a running turn; "/" jumps to the box.

export interface KeyContext {
  key: string;
  ctrl: boolean;
  alt: boolean;
  meta: boolean;
  typing: boolean; // focus is in a text field
  menuOpen: boolean;
  drawerOpen: boolean;
  dialogOpen: boolean; // a modal <dialog> handles its own keys
  renaming: boolean; // the title is being edited (Escape cancels that)
  busy: boolean; // a turn is running
}

export type Shortcut = "close-menu" | "close-drawer" | "stop" | "focus-input";

export function shortcut(context: KeyContext): Shortcut | null {
  if (context.dialogOpen) return null;

  if (context.key === "Escape") {
    if (context.renaming) return null;
    if (context.menuOpen) return "close-menu";
    if (context.drawerOpen) return "close-drawer";
    if (context.busy) return "stop";
    return null;
  }

  const modified = context.ctrl || context.alt || context.meta;

  if (context.key === "/" && !modified && !context.typing) return "focus-input";

  return null;
}
