"""Chat chains for Ollama models picked per turn (F036). MODEL keeps the chains in `chains/`; any other
installed model gets the same prompts on its own ChatOllama, built once."""

import threading

_CACHE: dict[str, tuple] = {}
_LOCK = threading.Lock()


def _build(name: str) -> tuple:
    # Imported here: `chains` loads the main model at import, which tests replace with stubs.
    from langchain_ollama import ChatOllama

    from chains.code_chain import code_prompt
    from chains.direct_chain import direct_prompt
    from llm_options import ollama_options
    from tools import TOOLS

    model = ChatOllama(model=name, **ollama_options())
    return direct_prompt | model, code_prompt | model.bind_tools(TOOLS)


def chains_for(name: str) -> tuple:
    """(direct_chain, code_chain) for an installed Ollama model."""
    with _LOCK:
        if name not in _CACHE:
            _CACHE[name] = _build(name)
        return _CACHE[name]
