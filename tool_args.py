"""Repair a tool call's arguments before the tool runs.

A small local model gets tool arguments slightly wrong, and the transport changes some of them:
langchain_ollama parses every string argument that looks like JSON into a dict or list
(`_parse_arguments_from_tool_call`, a band-aid for an Ollama bug), so `write_file` of a .json
file received a dict and failed validation however often the model retried.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool


def _types(prop: dict[str, Any]) -> set[str]:
    """The JSON schema types a property accepts ("string", "array", ...)."""
    options = prop.get("anyOf", [prop])
    return {option["type"] for option in options if "type" in option}


def _properties(tool: BaseTool) -> dict[str, dict[str, Any]]:
    return tool.get_input_jsonschema().get("properties", {})


def prepare(tool: BaseTool, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The arguments to call `tool` with, and an error message instead when they cannot work."""
    properties = _properties(tool)
    fixed = dict(args)

    for name, value in args.items():
        accepts = _types(properties.get(name, {}))

        # A dict or list where only text fits was JSON text before langchain_ollama parsed it.
        if isinstance(value, (dict, list)) and "string" in accepts and not accepts & {"object", "array"}:
            fixed[name] = json.dumps(value, indent=2, ensure_ascii=False)

    return fixed, None
