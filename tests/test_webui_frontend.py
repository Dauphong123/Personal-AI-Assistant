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
        self.assertTrue({"api.ts", "state.ts", "types.ts", "url.ts", "voice.ts", "main.ts"} <= names, names)
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


PUBLIC = WEBUI / "public"
HTML = (PUBLIC / "index.html").read_text(encoding="utf-8") if (PUBLIC / "index.html").exists() else ""
CSS = (PUBLIC / "style.css").read_text(encoding="utf-8") if (PUBLIC / "style.css").exists() else ""
MAIN = (SRC / "main.ts").read_text(encoding="utf-8") if (SRC / "main.ts").exists() else ""

CONTROLS = [
    "sidebar", "sessions", "new-session", "toggle-sidebar", "session-title", "status", "rename", "clear-chat",
    "delete-session", "scroll", "empty", "log", "alert", "approval", "approval-calls", "allow", "deny",
    "form", "input", "send", "stop", "mic", "speak-label", "speak-answers",
    "session-menu", "session-menu-list", "title-input", "confirm", "confirm-ok", "confirm-cancel", "jump",
]


class PageTests(unittest.TestCase):
    def test_files_exist(self):
        for path in (PUBLIC / "index.html", PUBLIC / "style.css", SRC / "main.ts", SRC / "url.ts"):
            self.assertTrue(path.exists(), path)

    def test_has_every_control(self):
        for control in CONTROLS:
            self.assertIn(f'id="{control}"', HTML, control)

    def test_is_a_complete_document(self):
        self.assertTrue(HTML.lstrip().lower().startswith("<!doctype html>"))
        self.assertIn('<meta charset="utf-8">', HTML.lower())
        self.assertIn('name="viewport"', HTML)
        self.assertIn('<html lang="en">', HTML)

    def test_loads_the_built_module_and_style_from_own_origin(self):
        self.assertIn('<script type="module" src="/main.js"></script>', HTML)
        self.assertIn('href="/style.css"', HTML)

    def test_works_under_a_strict_csp(self):
        # The server forbids inline script and style, so none may be used.
        self.assertIsNone(re.search(r"<script(?![^>]*\bsrc=)", HTML), "inline <script>")
        self.assertIsNone(re.search(r"<style", HTML, re.I), "inline <style>")
        self.assertIsNone(re.search(r"\sstyle\s*=", HTML, re.I), "style attribute")
        self.assertIsNone(re.search(r"\son[a-z]+\s*=", HTML, re.I), "inline event handler")

    def test_no_external_resources(self):
        self.assertIsNone(re.search(r"(?:src|href)\s*=\s*[\"']https?:", HTML))
        self.assertNotIn("@import", CSS)
        self.assertNotIn("url(http", CSS)

    def test_every_element_main_looks_up_exists_in_the_page(self):
        used = set(re.findall(r"""byId(?:<[^>]+>)?\(\s*["']([\w-]+)["']""", MAIN))
        self.assertTrue(used)
        for element in used:
            self.assertIn(f'id="{element}"', HTML, element)

    def test_session_state_classes_do_not_reuse_panel_classes(self):
        # A badge with class "approval" picked up the approval panel's box styles.
        self.assertIn("state-${session.state}", MAIN)
        for state in ("running", "waiting", "approval"):
            self.assertIn(f".session-state.state-{state}", CSS)

    def test_no_browser_popups(self):
        # Rename and confirmations are part of the page (a prompt() box blocks the whole tab).
        for popup in ("window.prompt", "window.confirm", "window.alert", "prompt(", "confirm(\"", "alert("):
            self.assertNotIn(popup, MAIN, popup)

    def test_session_actions_are_a_menu(self):
        menu = HTML[HTML.index('id="session-menu-list"'):]
        menu = menu[:menu.index("</div>")]
        for item in ("rename", "clear-chat", "delete-session"):
            self.assertIn(f'id="{item}"', menu, item)
        self.assertEqual(menu.count('role="menuitem"'), 3)

    def test_token_is_kept_out_of_the_address_bar(self):
        self.assertIn("history.replaceState", MAIN)


class AccessibilityTests(unittest.TestCase):
    def test_icon_only_buttons_have_a_name(self):
        # A button whose only content is an icon needs aria-label (screen readers, voice control).
        for match in re.finditer(r"<button\b([^>]*)>(.*?)</button>", HTML, re.S):
            attrs, body = match.groups()
            text = re.sub(r"<svg\b.*?</svg>", "", body, flags=re.S)
            text = re.sub(r"<[^>]+>", "", text).strip()
            self.assertTrue(text or "aria-label=" in attrs, f"unnamed button: {attrs.strip()}")

    def test_icons_are_hidden_from_screen_readers(self):
        for svg in re.findall(r"<svg\b[^>]*>", HTML):
            self.assertIn('aria-hidden="true"', svg)

    def test_thinking_placeholder_and_examples_exist(self):
        for element in ("thinking", "empty-text", "examples"):
            self.assertIn(f'id="{element}"', HTML, element)
        self.assertIn("data-prompt=", HTML)


def custom_properties(block: str) -> set[str]:
    return set(re.findall(r"(--[\w-]+)\s*:", block))


class StyleTests(unittest.TestCase):
    def test_dark_theme_defines_every_colour(self):
        light = CSS[CSS.index(":root {"):CSS.index("}", CSS.index(":root {"))]
        dark_start = CSS.index(":root:not([data-theme=\"light\"])")
        dark = CSS[dark_start:CSS.index("}", dark_start)]
        forced_start = CSS.index(':root[data-theme="dark"]')
        forced = CSS[forced_start:CSS.index("}", forced_start)]
        colours = {name for name in custom_properties(light) if not name.startswith(("--sans", "--display", "--mono", "--radius", "--column"))}
        self.assertTrue(colours)
        self.assertEqual(colours, custom_properties(dark))
        self.assertEqual(custom_properties(dark), custom_properties(forced))

    def test_supports_dark_mode_and_reduced_motion(self):
        self.assertIn("prefers-color-scheme: dark", CSS)
        self.assertIn("prefers-reduced-motion", CSS)

    def test_keyboard_focus_is_visible(self):
        self.assertIn(":focus-visible", CSS)

    def test_adapts_to_narrow_screens(self):
        self.assertIn("@media (max-width", CSS)

    def test_hidden_attribute_is_honoured(self):
        self.assertIn("[hidden]", CSS)


@unittest.skipUnless(NPM and INSTALLED, "needs npm and `npm --prefix webui install`")
class BuildTests(unittest.TestCase):
    def test_build_makes_a_page_the_backend_serves(self):
        import http.client
        import threading

        from tests import stubs

        stubs.install()
        from backend.server import STATIC_DIR, make_server
        from backend.sessions import SessionManager

        result = npm("run", "build")
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])
        dist = WEBUI / "dist"
        self.assertEqual(STATIC_DIR, dist)

        for name in ("index.html", "style.css", "main.js", "api.js", "state.js", "url.js", "voice.js"):
            self.assertTrue((dist / name).is_file(), name)

        main_js = (dist / "main.js").read_text(encoding="utf-8")
        self.assertIn('from "./state.js"', main_js)  # rewritten from .ts for the browser
        self.assertNotIn('.ts"', main_js)

        server = make_server(SessionManager(app=None, run=lambda *a, **k: False), "t", port=0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        try:
            for path, needle in (("/", b'id="log"'), ("/main.js", b"/api/"), ("/style.css", b"focus-visible")):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                self.assertEqual(resp.status, 200, path)
                if path == "/main.js":
                    body = (dist / "api.js").read_bytes()  # routes live in the api module
                self.assertIn(needle, body, path)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
