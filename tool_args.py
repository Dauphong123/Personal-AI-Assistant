"""Repair a tool call's arguments before the tool runs.

A small local model gets tool arguments slightly wrong, and the transport changes some of them:
langchain_ollama parses every string argument that looks like JSON into a dict or list
(`_parse_arguments_from_tool_call`, a band-aid for an Ollama bug), so `write_file` of a .json
file received a dict and failed validation however often the model retried.
"""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError


# Names a model may use for a parameter, from other agents' tools and from this project's own
# inconsistent directory names. A name is renamed only when exactly one parameter matches.
SYNONYMS = (
    {"path", "file", "file_path", "filepath", "filename", "file_name"},
    {"path", "directory", "dir", "folder", "repo", "repo_path", "repository", "cwd"},
    {"old", "old_text", "old_string", "old_str", "find"},
    {"new", "new_text", "new_string", "new_str", "replacement", "replace"},
    {"content", "contents", "text", "file_content", "data"},
    {"command", "cmd", "shell_command"},
    {"query", "pattern", "search", "search_query", "keyword"},
    {"source", "src", "source_path", "from_path", "old_path"},
    {"destination", "dest", "destination_path", "target", "to_path", "new_path"},
)

PYTHON_TYPES = {
    "string": "str", "integer": "int", "number": "float", "boolean": "bool",
    "array": "list", "object": "dict", "null": "None",
}  # fmt: skip


def _type_list(prop: dict[str, Any]) -> list[str]:
    """The JSON schema types a property accepts ("string", "array", ...), in schema order."""
    options = prop.get("anyOf", [prop])
    return [option["type"] for option in options if "type" in option]


def _types(prop: dict[str, Any]) -> set[str]:
    return set(_type_list(prop))


def signature(tool: BaseTool) -> str:
    """The tool as a Python call, e.g. `read_file(path: str, start_line: int | None = None)`."""
    schema = tool.get_input_jsonschema()
    required = set(schema.get("required", []))
    parts = []

    for name, prop in schema.get("properties", {}).items():
        types = " | ".join(PYTHON_TYPES.get(t, t) for t in _type_list(prop))
        part = f"{name}: {types}" if types else name

        if name not in required:
            part += f" = {prop.get('default')!r}"

        parts.append(part)

    return f"{tool.name}({', '.join(parts)})"


def _rename(name: str, free: set[str]) -> str | None:
    """The one unused parameter `name` stands for, if there is exactly one."""
    matches = {p for p in free if any(name in group and p in group for group in SYNONYMS)}
    return matches.pop() if len(matches) == 1 else None


def _properties(tool: BaseTool) -> dict[str, dict[str, Any]]:
    return tool.get_input_jsonschema().get("properties", {})


def prepare(tool: BaseTool, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The arguments to call `tool` with, and an error message instead when they cannot work."""
    properties = _properties(tool)
    free = set(properties) - set(args)
    fixed: dict[str, Any] = {}

    for name, value in args.items():
        if name not in properties:
            real = _rename(name, free)

            if real is None:
                # pydantic would drop it silently and the tool would run on its default.
                return args, (
                    f"ERROR: {tool.name} got unknown argument '{name}'. Expected: {signature(tool)}"
                )

            free.discard(real)
            name = real

        fixed[name] = value
        accepts = _types(properties.get(name, {}))

        # A dict or list where only text fits was JSON text before langchain_ollama parsed it.
        if isinstance(value, (dict, list)) and "string" in accepts and not accepts & {"object", "array"}:
            fixed[name] = json.dumps(value, indent=2, ensure_ascii=False)

    schema = tool.get_input_schema()

    try:
        # Checked here, not by invoke(), so a bad call never reaches the approval prompt.
        if issubclass(schema, BaseModel):  # pydantic v1 schemas are left to invoke()
            schema.model_validate(fixed)
    except ValidationError as e:
        return fixed, f"ERROR: {tool.name} got bad arguments: {_problems(e)}. Expected: {signature(tool)}"

    return fixed, None


def _problems(error: ValidationError) -> str:
    """Pydantic's errors in one line: `field: message (got value)`."""
    problems = []

    for item in error.errors():
        problem = f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"

        if item["type"] != "missing":
            value = repr(item["input"])
            problem += f" (got {value if len(value) <= 60 else value[:57] + '...'})"

        problems.append(problem)

    return "; ".join(problems)
