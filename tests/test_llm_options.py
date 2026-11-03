import unittest

from llm_options import ollama_options


class OllamaOptionsTests(unittest.TestCase):
    def test_defaults_turn_thinking_off_and_keep_the_model_loaded(self):
        self.assertEqual(ollama_options({}), {"reasoning": False, "keep_alive": "30m"})

    def test_reasoning_can_be_turned_back_on(self):
        for value in ["true", "TRUE", "1", "yes", " on "]:
            self.assertTrue(ollama_options({"LLM_REASONING": value})["reasoning"], value)

        for value in ["false", "0", "no", ""]:
            self.assertFalse(ollama_options({"LLM_REASONING": value})["reasoning"], value)

    def test_junk_values_are_rejected_with_the_variable_named(self):
        with self.assertRaisesRegex(ValueError, "LLM_REASONING"):
            ollama_options({"LLM_REASONING": "maybe"})

        for bad in ["8k", "4096.0", "-1", "0", "100", "abc"]:
            with self.assertRaisesRegex(ValueError, "LLM_NUM_CTX"):
                ollama_options({"LLM_NUM_CTX": bad})

    def test_keep_alive_and_context_are_configurable(self):
        options = ollama_options({"LLM_KEEP_ALIVE": "2h", "LLM_NUM_CTX": "4096"})
        self.assertEqual(options["keep_alive"], "2h")
        self.assertEqual(options["num_ctx"], 4096)
        self.assertNotIn("num_ctx", ollama_options({"LLM_NUM_CTX": " "}))
        self.assertEqual(ollama_options({"LLM_KEEP_ALIVE": " "})["keep_alive"], "30m")


if __name__ == "__main__":
    unittest.main()
