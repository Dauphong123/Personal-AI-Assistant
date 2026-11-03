"""Replace `dependency` and `chains` so tests don't load Ollama, HF models or Chroma."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

from langchain_core.tools import tool


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class _Rag:
    def get_search_document_tools(self):
        @tool
        def search_document(query: str) -> str:
            """Search the indexed knowledge documents."""
            return "doc"

        return search_document


def install():
    if "dependency" not in sys.modules:
        module = ModuleType("dependency")
        module.dependency = {  # type: ignore[attr-defined]
            "model": MagicMock(),
            "rag": _Rag(),
            "tokenizer": _Tokenizer(),
            "memory": None,
        }
        sys.modules["dependency"] = module

    if "chains" not in sys.modules:
        sys.modules["chains"] = MagicMock()
