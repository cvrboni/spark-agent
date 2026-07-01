from __future__ import annotations

from pathlib import Path
from typing import Any

from spark_agent.core.sandbox import build_sandbox, validate_read_path
from spark_agent.core.types import ToolSpec

type JsonObject = dict[str, Any]

DEFAULT_MAX_BYTES = 40_000
DEFAULT_TIMEOUT_S = 30.0


def workspace_tool_definitions() -> list[JsonObject]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List repository files under a directory without reading contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "max_entries": {"type": "integer", "minimum": 1, "maximum": 500, "default": 120},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a bounded prefix of a text file in the current repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_bytes": {
                            "type": "integer",
                            "minimum": 1024,
                            "maximum": 250000,
                            "default": DEFAULT_MAX_BYTES,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": (
                    "Apply a unified diff to repository files with git apply. Use this for code "
                    "edits after inspecting the target files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": (
                    "Run an allowlisted local validation command in the repository. Prefer this "
                    "for tests, compile checks, ruff, and git status/diff."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 12,
                        },
                        "timeout_s": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 120,
                            "default": DEFAULT_TIMEOUT_S,
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def workspace_tool_specs(
    repo_root: Path | None = None,
    *,
    sandbox_backend: str = "local",
    sandbox_image: str = "python:3.13-slim",
) -> list[ToolSpec]:
    root = (repo_root or Path.cwd()).resolve()
    sandbox = build_sandbox(root, backend=sandbox_backend, image=sandbox_image)
    definitions = {item["function"]["name"]: item for item in workspace_tool_definitions()}
    return [
        ToolSpec(
            name="list_files",
            definition=definitions["list_files"],
            handler=lambda arguments: _list_files_handler(arguments, root),
        ),
        ToolSpec(
            name="read_file",
            definition=definitions["read_file"],
            handler=lambda arguments: _read_file_handler(arguments, root),
        ),
        ToolSpec(
            name="apply_patch",
            definition=definitions["apply_patch"],
            handler=lambda arguments: _apply_patch_handler(arguments, sandbox),
        ),
        ToolSpec(
            name="run_command",
            definition=definitions["run_command"],
            handler=lambda arguments: _run_command_handler(arguments, sandbox),
        ),
    ]


async def _list_files_handler(arguments: JsonObject, root: Path) -> JsonObject:
    path = _resolve_under_root(root, str(arguments.get("path", ".")))
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    max_entries = max(1, min(int(arguments.get("max_entries", 120)), 500))
    entries: list[JsonObject] = []
    for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name))[:max_entries]:
        if _is_ignored(child):
            continue
        entries.append(
            {
                "path": str(child.relative_to(root)),
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return {"path": str(path.relative_to(root)), "entries": entries, "truncated": len(entries) >= max_entries}


async def _read_file_handler(arguments: JsonObject, root: Path) -> JsonObject:
    path = _resolve_under_root(root, str(arguments["path"]))
    if not path.is_file():
        raise FileNotFoundError(str(path))
    validate_read_path(path, root)
    max_bytes = max(1024, min(int(arguments.get("max_bytes", DEFAULT_MAX_BYTES)), 250_000))
    data = path.read_bytes()[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    return {
        "path": str(path.relative_to(root)),
        "content": text,
        "truncated": path.stat().st_size > max_bytes,
        "bytes_read": len(data),
    }


async def _apply_patch_handler(arguments: JsonObject, sandbox) -> JsonObject:
    patch = str(arguments["patch"])
    result = await sandbox.apply_patch(patch)
    payload = result.to_json(stdout_chars=4000, stderr_chars=4000)
    payload["ok"] = result.returncode == 0
    return payload


async def _run_command_handler(arguments: JsonObject, sandbox) -> JsonObject:
    raw_command = arguments["command"]
    if not isinstance(raw_command, list) or not all(isinstance(item, str) for item in raw_command):
        raise ValueError("command must be a list of strings")
    command = [str(item) for item in raw_command]
    timeout_s = max(1.0, min(float(arguments.get("timeout_s", DEFAULT_TIMEOUT_S)), 120.0))
    result = await sandbox.run_command(command, timeout_s=timeout_s)
    return result.to_json(stdout_chars=12000, stderr_chars=12000)


def _resolve_under_root(root: Path, value: str) -> Path:
    path = (root / value).expanduser().resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes repository root: {value}") from exc
    return path


def _is_ignored(path: Path) -> bool:
    return any(part in {".git", ".venv", "__pycache__", "node_modules", ".spark-agent"} for part in path.parts)
