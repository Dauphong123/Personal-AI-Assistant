import tempfile
import unittest

from settings import check_settings, max_history_tokens, memory_idle_seconds, settings_error

GOOD = {"MODEL": "qwen3:8b"}


class CheckSettingsTests(unittest.TestCase):
    def test_a_minimal_env_is_fine(self):
        self.assertEqual(check_settings(GOOD), [])

    def test_every_problem_is_reported_at_once(self):
        problems = check_settings(
            {
                "MODEL": "llama9",
                "MAX_HISTORY_TOKENS": "lots",
                "MEMORY_IDLE_SECONDS": "-1",
                "WEB_SEARCH_ENABLED": "ture",
                "LOG_LEVEL": "loud",
            }
        )
        text = "\n".join(problems)
        for name in ("MODEL", "MAX_HISTORY_TOKENS", "MEMORY_IDLE_SECONDS", "WEB_SEARCH_ENABLED", "LOG_LEVEL"):
            self.assertIn(name, text)
        self.assertEqual(len(problems), 5)

    def test_flags_accept_the_usual_spellings(self):
        for value in ("true", "False", "1", "0", "yes", "off", ""):
            env = {**GOOD, "VOICE_ENABLED": value, "LLM_REASONING": value, "LOG_TIMING": value}
            self.assertEqual(check_settings(env), [], value)

    def test_voice_enabled_typo_is_reported(self):
        self.assertIn("VOICE_ENABLED", "\n".join(check_settings({**GOOD, "VOICE_ENABLED": "yess"})))

    def test_bad_context_size_is_reported(self):
        self.assertIn("LLM_NUM_CTX", "\n".join(check_settings({**GOOD, "LLM_NUM_CTX": "big"})))

    def test_project_root_must_be_an_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(check_settings({**GOOD, "PROJECT_ROOT": tmp}), [])
        problems = check_settings({**GOOD, "PROJECT_ROOT": tmp + "-missing"})
        self.assertIn("PROJECT_ROOT", "\n".join(problems))

    def test_the_error_lists_the_problems_and_points_at_env(self):
        message = settings_error(["MODEL is not set", "LOG_LEVEL must be ..."])
        self.assertIn(".env", message)
        self.assertIn("- MODEL is not set", message)
        self.assertIn("- LOG_LEVEL must be ...", message)


class ParsedValueTests(unittest.TestCase):
    def test_history_default_and_value(self):
        self.assertEqual(max_history_tokens({}), 5000)
        self.assertEqual(max_history_tokens({"MAX_HISTORY_TOKENS": " 3000 "}), 3000)

    def test_history_must_be_a_positive_whole_number(self):
        for value in ("0", "-5", "1.5", "abc"):
            with self.assertRaises(ValueError) as caught:
                max_history_tokens({"MAX_HISTORY_TOKENS": value})
            self.assertIn("MAX_HISTORY_TOKENS", str(caught.exception))

    def test_idle_default_and_value(self):
        self.assertEqual(memory_idle_seconds({}), 3.0)
        self.assertEqual(memory_idle_seconds({"MEMORY_IDLE_SECONDS": "0.5"}), 0.5)
        self.assertEqual(memory_idle_seconds({"MEMORY_IDLE_SECONDS": "0"}), 0.0)

    def test_idle_must_be_a_non_negative_number(self):
        for value in ("-1", "soon", "nan"):
            with self.assertRaises(ValueError) as caught:
                memory_idle_seconds({"MEMORY_IDLE_SECONDS": value})
            self.assertIn("MEMORY_IDLE_SECONDS", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
