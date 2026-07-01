from __future__ import annotations

import ast
import asyncio
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from spark_agent.core.types import ToolSpec

type JsonObject = dict[str, Any]
type JsonArray = list[JsonObject]


DEFAULT_MAX_MATCHES = 10
DEFAULT_CONTEXT_LINES = 1
DEFAULT_MAX_BYTES = 24_000


def tool_definitions() -> list[JsonObject]:
    return [
        {
            "type": "function",
            "function": {
                "name": "repo_grep",
                "description": (
                    "Search a repository with ripgrep and return only bounded matching snippets. "
                    "Use this before requesting full files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "root": {"type": "string", "default": "."},
                        "glob": {"type": "string", "default": "*"},
                        "context_lines": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 8,
                            "default": DEFAULT_CONTEXT_LINES,
                        },
                        "max_matches": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": DEFAULT_MAX_MATCHES,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "view_file_outline",
                "description": (
                    "Return a compact outline of classes, functions, and methods in a code file "
                    "without loading the full file into context."
                ),
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
    ]


def codebase_tool_specs(repo_root: Path | None = None) -> list[ToolSpec]:
    base_root = (repo_root or Path.cwd()).resolve()
    definitions = {item["function"]["name"]: item for item in tool_definitions()}
    return [
        ToolSpec(
            name="repo_grep",
            definition=definitions["repo_grep"],
            handler=lambda arguments: _repo_grep_handler(arguments, base_root),
        ),
        ToolSpec(
            name="view_file_outline",
            definition=definitions["view_file_outline"],
            handler=lambda arguments: _view_file_outline_handler(arguments, base_root),
        ),
    ]


async def _repo_grep_handler(arguments: JsonObject, repo_root: Path) -> JsonObject:
    root = _resolve_under_root(repo_root, str(arguments.get("root", ".")))
    return await repo_grep(
        query=str(arguments["query"]),
        root=root,
        glob=str(arguments.get("glob", "*")),
        context_lines=int(arguments.get("context_lines", DEFAULT_CONTEXT_LINES)),
        max_matches=int(arguments.get("max_matches", DEFAULT_MAX_MATCHES)),
    )


async def _view_file_outline_handler(arguments: JsonObject, repo_root: Path) -> JsonObject:
    path = _resolve_under_root(repo_root, str(arguments["path"]))
    return await view_file_outline(
        path=path,
        max_bytes=int(arguments.get("max_bytes", DEFAULT_MAX_BYTES)),
    )


async def repo_grep(
    query: str,
    *,
    root: str | Path = ".",
    glob: str = "*",
    context_lines: int = DEFAULT_CONTEXT_LINES,
    max_matches: int = DEFAULT_MAX_MATCHES,
    timeout_s: float = 10.0,
) -> JsonObject:
    if not query:
        raise ValueError("query must be non-empty")
    root_path = _safe_root(root)
    bounded_context = max(0, min(context_lines, 8))
    bounded_matches = max(1, min(max_matches, 200))

    if shutil.which("rg"):
        return await _repo_grep_rg(
            query=query,
            root=root_path,
            glob=glob,
            context_lines=bounded_context,
            max_matches=bounded_matches,
            timeout_s=timeout_s,
        )
    return await _repo_grep_python(
        query=query,
        root=root_path,
        glob=glob,
        context_lines=bounded_context,
        max_matches=bounded_matches,
    )


async def view_file_outline(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> JsonObject:
    file_path = _safe_file(path)
    bounded_bytes = max(1024, min(max_bytes, 250_000))
    data = _read_prefix(file_path, bounded_bytes)
    text = data.decode("utf-8", errors="replace")
    if file_path.suffix == ".py":
        outline = _python_outline(text)
        parser = "python_ast"
    else:
        outline = _generic_outline(text)
        parser = "generic_text"
    return {
        "path": str(file_path),
        "parser": parser,
        "truncated": file_path.stat().st_size > bounded_bytes,
        "symbols": outline,
    }


async def _repo_grep_rg(
    *,
    query: str,
    root: Path,
    glob: str,
    context_lines: int,
    max_matches: int,
    timeout_s: float,
) -> JsonObject:
    process = await asyncio.create_subprocess_exec(
        "rg",
        "--json",
        "--line-number",
        "--hidden",
        "--no-heading",
        "--context",
        str(context_lines),
        "--max-count",
        str(max_matches),
        "--glob",
        glob,
        query,
        str(root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise TimeoutError(f"repo_grep timed out after {timeout_s}s") from exc

    if process.returncode not in {0, 1}:
        raise RuntimeError(stderr.decode("utf-8", errors="replace")[:1000])

    matches = _parse_rg_json_lines(stdout.decode("utf-8", errors="replace"), max_matches)
    return {
        "query": query,
        "root": str(root),
        "engine": "ripgrep",
        "match_count": len(matches),
        "matches": matches,
    }


async def _repo_grep_python(
    *,
    query: str,
    root: Path,
    glob: str,
    context_lines: int,
    max_matches: int,
) -> JsonObject:
    matches = _scan_files(query, root, glob, context_lines, max_matches)
    return {
        "query": query,
        "root": str(root),
        "engine": "python_fallback",
        "match_count": len(matches),
        "matches": matches,
    }


def _parse_rg_json_lines(output: str, max_matches: int) -> JsonArray:
    import json

    results: JsonArray = []
    current_context: list[JsonObject] = []
    for raw_line in output.splitlines():
        event = json.loads(raw_line)
        event_type = event.get("type")
        data = event.get("data", {})
        if event_type == "context":
            current_context.append(_rg_line(data))
            current_context = current_context[-8:]
        elif event_type == "match":
            result = _rg_line(data)
            result["before"] = list(current_context)
            results.append(result)
            current_context = []
            if len(results) >= max_matches:
                break
    return results


def _rg_line(data: JsonObject) -> JsonObject:
    path = data.get("path", {}).get("text", "")
    lines = data.get("lines", {}).get("text", "")
    return {
        "path": path,
        "line_number": data.get("line_number"),
        "text": lines.rstrip("\n"),
    }


def _scan_files(
    query: str,
    root: Path,
    glob: str,
    context_lines: int,
    max_matches: int,
) -> JsonArray:
    results: JsonArray = []
    for path in _iter_candidate_files(root, glob):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines):
            if query not in line:
                continue
            start = max(0, index - context_lines)
            end = min(len(lines), index + context_lines + 1)
            results.append(
                {
                    "path": str(path),
                    "line_number": index + 1,
                    "text": line,
                    "context": [
                        {"line_number": line_no + 1, "text": lines[line_no]}
                        for line_no in range(start, end)
                    ],
                }
            )
            if len(results) >= max_matches:
                return results
    return results


def _iter_candidate_files(root: Path, glob: str) -> Iterable[Path]:
    for path in root.rglob(glob):
        if not path.is_file():
            continue
        if any(part in {".git", ".venv", "__pycache__", "node_modules"} for part in path.parts):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        yield path


def _python_outline(text: str) -> JsonArray:
    tree = ast.parse(text)
    symbols: JsonArray = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [
                        {
                            "kind": "method",
                            "name": item.name,
                            "line": item.lineno,
                            "async": isinstance(item, ast.AsyncFunctionDef),
                            "args": _args_outline(item.args),
                        }
                        for item in node.body
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    ],
                }
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "line": node.lineno,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "args": _args_outline(node.args),
                }
            )
    return symbols


def _args_outline(args: ast.arguments) -> list[str]:
    positional = [arg.arg for arg in [*args.posonlyargs, *args.args]]
    keyword_only = [f"{arg.arg}=..." for arg in args.kwonlyargs]
    variadic = [f"*{args.vararg.arg}"] if args.vararg else []
    kw_variadic = [f"**{args.kwarg.arg}"] if args.kwarg else []
    return positional + variadic + keyword_only + kw_variadic


def _generic_outline(text: str) -> JsonArray:
    symbols: JsonArray = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("class ", "struct ", "interface ", "enum ")):
            symbols.append({"kind": "type", "line": line_no, "signature": stripped[:240]})
        elif any(
            stripped.startswith(prefix)
            for prefix in ("def ", "async def ", "function ", "fn ", "func ")
        ):
            symbols.append({"kind": "function", "line": line_no, "signature": stripped[:240]})
    return symbols


def _safe_root(root: str | Path) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"repository root not found: {path}")
    return path


def _safe_file(path: str | Path) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    return file_path


def _resolve_under_root(root: Path, value: str) -> Path:
    path = (root / value).expanduser().resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes repository root: {value}") from exc
    return path


def _read_prefix(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as file:
        return file.read(max_bytes)
