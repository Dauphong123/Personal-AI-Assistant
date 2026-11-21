"""F036: the models the user can switch between, and where the pick is saved."""

import json
import tempfile
import unittest
from pathlib import Path

from model_choice import (
    ModelChoice,
    available_models,
    check_available,
    startup_choice,
    default_choice,
    load_settings,
    ollama_models,
    parse_choice,
    privacy_notice,
    save_default,
)

CLAUDE = ["claude-code:opus", "claude-code:sonnet", "claude-code:haiku", "claude-code:fable"]
ENV = {"MODEL": "qwen3:8b"}


def tags(*models: tuple[str, str]):
    body = json.dumps({"models": [{"name": n, "modified_at": t} for n, t in models]}).encode()
    return lambda url: body


def down(url: str) -> bytes:
    raise OSError("connection refused")


class ParseChoiceTests(unittest.TestCase):
    def test_claude_with_and_without_effort(self):
        self.assertEqual(parse_choice("claude-code:opus high"), ModelChoice("claude-code:opus", "high"))
        self.assertEqual(parse_choice("claude-code:opus"), ModelChoice("claude-code:opus", "medium"))

    def test_bare_alias_means_claude(self):
        self.assertEqual(parse_choice(" sonnet "), ModelChoice("claude-code:sonnet", "medium"))

    def test_ollama_name_has_no_effort(self):
        self.assertEqual(parse_choice("qwen3:4b"), ModelChoice("qwen3:4b", None))

    def test_bad_input_names_the_usage(self):
        for text in ["claude-code:gpt", "claude-code:opus extreme", "qwen3:4b high", "", "a b c"]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, r"/model <name> \[effort\]"):
                parse_choice(text)

    def test_choice_properties(self):
        claude = ModelChoice("claude-code:sonnet", "medium")
        self.assertTrue(claude.is_claude)
        self.assertEqual(claude.alias, "sonnet")
        self.assertEqual(claude.label(), "claude-code:sonnet (medium)")
        local = ModelChoice("qwen3:8b", None)
        self.assertFalse(local.is_claude)
        self.assertIsNone(local.alias)
        self.assertEqual(local.label(), "qwen3:8b")


class ListTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "model_settings.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_ollama_list_hides_embeddings_newest_first(self):
        fetch = tags(
            ("nomic-embed-text", "2026-11-02T00:00:00Z"),
            ("qwen3:4b", "2026-11-01T00:00:00Z"),
            ("qwen3:8b", "2026-11-03T00:00:00Z"),
        )
        self.assertEqual(ollama_models(ENV, fetch), ["qwen3:8b", "qwen3:4b"])

    def test_ollama_host_is_used(self):
        seen: list[str] = []

        def fetch(url: str) -> bytes:
            seen.append(url)
            return b'{"models": []}'

        ollama_models({"OLLAMA_HOST": "http://box:9999/"}, fetch)
        self.assertEqual(seen, ["http://box:9999/api/tags"])

    def test_live_list_is_saved_and_claude_follows(self):
        models = available_models(ENV, tags(("qwen3:8b", "2026-11-03T00:00:00Z")), self.path)
        self.assertEqual(models, ["qwen3:8b", *CLAUDE])
        self.assertEqual(load_settings(self.path)["ollama_models"], ["qwen3:8b"])

    def test_list_falls_back_when_ollama_is_down(self):
        self.path.write_text(json.dumps({"ollama_models": ["qwen3:4b"]}), encoding="utf-8")
        self.assertEqual(available_models(ENV, down, self.path), ["qwen3:4b", "qwen3:8b", *CLAUDE])

    def test_nothing_saved_and_ollama_down_still_lists_model(self):
        self.assertEqual(available_models(ENV, down, self.path), ["qwen3:8b", *CLAUDE])


class DefaultChoiceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "model_settings.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_missing_file_means_model(self):
        self.assertEqual(default_choice(ENV, self.path), ModelChoice("qwen3:8b", None))

    def test_bad_files_warn_and_fall_back_to_model(self):
        for text in ["not json", '{"chat_model": "claude-code:gpt"}', "[1, 2]", '{"chat_model": 5}']:
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertLogs("model_choice", "WARNING"):
                    self.assertEqual(default_choice(ENV, self.path), ModelChoice("qwen3:8b", None))

    def test_a_saved_ollama_model_that_is_gone_falls_back_to_model(self):
        save_default(ModelChoice("qwen3:4b", None), self.path)
        with self.assertLogs("model_choice", "WARNING"):
            self.assertEqual(default_choice(ENV, self.path, installed=["qwen3:8b"]), ModelChoice("qwen3:8b", None))
        self.assertEqual(default_choice(ENV, self.path, installed=["qwen3:4b"]), ModelChoice("qwen3:4b", None))
        self.assertEqual(default_choice(ENV, self.path), ModelChoice("qwen3:4b", None), "no list: not checked")

    def test_startup_choice_checks_the_live_list_when_ollama_answers(self):
        save_default(ModelChoice("qwen3:4b", None), self.path)
        with self.assertLogs("model_choice", "WARNING"):
            self.assertEqual(startup_choice(ENV, self.path, tags(("qwen3:8b", "2026-11-03T00:00:00Z"))), ModelChoice("qwen3:8b", None))
        self.assertEqual(startup_choice(ENV, self.path, down), ModelChoice("qwen3:4b", None))

    def test_check_available(self):
        check_available(ModelChoice("qwen3:4b", None), ["qwen3:4b"])
        check_available(ModelChoice("claude-code:opus", "low"), ["qwen3:4b"])
        with self.assertRaisesRegex(ValueError, "not installed"):
            check_available(ModelChoice("qwen3:8bb", None), ["qwen3:8b"])

    def test_saved_choice_is_used(self):
        save_default(ModelChoice("claude-code:opus", "high"), self.path)
        self.assertEqual(default_choice(ENV, self.path), ModelChoice("claude-code:opus", "high"))

    def test_save_default_keeps_other_keys(self):
        self.path.write_text(json.dumps({"ollama_models": ["a"]}), encoding="utf-8")
        save_default(ModelChoice("claude-code:opus", "high"), self.path)
        self.assertEqual(
            load_settings(self.path),
            {"ollama_models": ["a"], "chat_model": "claude-code:opus", "effort": "high"},
        )


class PrivacyTests(unittest.TestCase):
    def test_privacy_notice_only_for_claude(self):
        self.assertIn("Anthropic", privacy_notice(ModelChoice("claude-code:haiku", "low")) or "")
        self.assertIsNone(privacy_notice(ModelChoice("qwen3:8b", None)))


if __name__ == "__main__":
    unittest.main()
