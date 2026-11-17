"""The TypeScript frontend in webui/: its own tests and type-check run from the Python suite
(skipped when Node or webui/node_modules is missing), plus static checks of the sources."""

import re
import shutil
import subprocess
import unittest
from pathlib import Path

WEBUI = Path(__file__).resolve().parent.parent / "webui"
SRC = WEBUI / "src"
NPM = shutil.which("npm")
INSTALLED = (WEBUI / "node_modules" / "typescript").is_dir()


def npm(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [NPM or "npm", *args], cwd=WEBUI, capture_output=True, text=True, timeout=300, shell=False
    )


@unittest.skipUnless(NPM and INSTALLED, "needs npm and `npm --prefix webui install`")
class NodeToolchainTests(unittest.TestCase):
    def test_frontend_unit_tests_pass(self):
        result = npm("test")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])

    def test_typescript_type_checks(self):
        result = npm("run", "check")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])


class SourceTests(unittest.TestCase):
    def sources(self) -> dict[str, str]:
        return {path.name: path.read_text(encoding="utf-8") for path in SRC.glob("*.ts")}

    def test_sources_are_typescript(self):
        names = set(self.sources())
        self.assertTrue({"api.ts", "state.ts", "types.ts"} <= names, names)
        self.assertEqual(list(SRC.glob("*.js")), [])

    def test_never_writes_html(self):
        # Model and web-page text is untrusted: it may only be inserted as text.
        for name, code in self.sources().items():
            for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
                self.assertNotIn(sink, code, f"{sink} in {name}")

    def test_relative_imports_name_ts_files(self):
        # Node runs the tests on the .ts sources; tsc rewrites the extension for the browser.
        for name, code in self.sources().items():
            for spec in re.findall(r"""from\s+["'](\.[^"']+)["']""", code):
                self.assertTrue(spec.endswith(".ts"), f"{name} imports {spec}")

    def test_dev_dependencies_are_pinned(self):
        import json

        package = json.loads((WEBUI / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(package.get("dependencies", {}), {})
        for name, version in package["devDependencies"].items():
            self.assertRegex(version, r"^\d+\.\d+\.\d+$", name)
        self.assertTrue((WEBUI / "package-lock.json").is_file())


if __name__ == "__main__":
    unittest.main()
