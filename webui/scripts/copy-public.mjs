// Copy the static files in public/ next to the compiled scripts in dist/.
import { cpSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
cpSync(join(root, "public"), join(root, "dist"), { recursive: true });
