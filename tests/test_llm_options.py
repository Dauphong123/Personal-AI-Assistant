import unittest

from llm_options import TOKENIZER_MAP, model_name, ollama_options


class ModelNameTests(unittest.TestCase):
    def test_a_supported_model_is_returned(self):
        self.assertEqual(model_name({"MODEL": " qwen3:8b "}), "qwen3:8b")
        self.assertIn("qwen3:8b", TOKENIZER_MAP)

    def test_qwen3_4b_is_supported_with_the_shared_qwen3_tokenizer(self):
        # F025: 2.3x faster on this GPU (14.2 vs 6.2 tok/s); the Qwen3 sizes share one tokenizer,
        # so the one already downloaded for qwen3:8b counts its tokens.
        self.assertEqual(model_name({"MODEL": "qwen3:4b"}), "qwen3:4b")
        self.assertEqual(TOKENIZER_MAP["qwen3:4b"], TOKENIZER_MAP["qwen3:8b"])

    def test_missing_model_says_how_to_fix_it(self):
        for env in [{}, {"MODEL": ""}, {"MODEL": "   "}]:
            with self.assertRaisesRegex(ValueError, r"MODEL is not set.*\.env.*qwen3:8b"):
                model_name(env)

    def test_unsupported_model_lists_the_supported_ones(self):
        with self.assertRaisesRegex(ValueError, r"'llama3' is not supported.*qwen3:8b.*TOKENIZER_MAP"):
            model_name({"MODEL": "llama3"})


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
