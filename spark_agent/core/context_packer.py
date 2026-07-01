from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spark_agent.core.repo_index import RepoFileRecord, RepoIndex

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass(frozen=True, slots=True)
class ContextPacker:
    root: Path
    index: RepoIndex

    @classmethod
    def from_root(cls, root: Path, *, use_cache: bool = True) -> ContextPacker:
        index = RepoIndex.build(root, use_cache=use_cache)
        return cls(root=root.resolve(), index=index)

    def pack(self, task: str, *, max_files: int = 6, max_chars: int = 8_000) -> str:
        max_files = max(1, max_files)
        max_chars = max(1_000, max_chars)
        selected = self._select_records(task, max_files=max_files)
        rendered = self._render(task, selected)
        while len(rendered) > max_chars and len(selected) > 1:
            selected = selected[:-1]
            rendered = self._render(task, selected)
        if len(rendered) <= max_chars:
            return rendered
        return f"{rendered[:max_chars]}\n...[repository snapshot truncated to {max_chars} chars]"

    def _select_records(self, task: str, *, max_files: int) -> list[RepoFileRecord]:
        tokens = _query_tokens(task)
        scored = [
            (_score_record(record, tokens), record.path, record)
            for record in self.index.records
            if _is_context_candidate(record)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for score, _path, record in scored[:max_files] if score > 0] or [
            record for _score, _path, record in scored[:max_files]
        ]

    def _render(self, task: str, selected: list[RepoFileRecord]) -> str:
        payload = {
            "root": ".",
            "index": {
                "file_count": len(self.index.records),
                "top_level": self.index.top_level_entries(),
            },
            "query_tokens": sorted(_query_tokens(task))[:24],
            "selected_files": [self._file_context(record) for record in selected],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _file_context(self, record: RepoFileRecord) -> dict[str, Any]:
        path = self.root / record.path
        base: dict[str, Any] = {
            "path": record.path,
            "language": record.language,
            "size": record.size,
            "sha256": record.sha256[:16],
        }
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:24_000]
        except OSError as exc:
            base["error"] = str(exc)
            return base
        outline = _outline_text(record, text)
        if outline:
            base["outline"] = outline
        else:
            base["preview"] = _compact_preview(text)
        return base


def _query_tokens(task: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(task)}


def _is_context_candidate(record: RepoFileRecord) -> bool:
    if record.language in {"python", "markdown", "toml", "json", "yaml"}:
        return True
    return record.path in {"README", "README.md", "LICENSE"}


def _score_record(record: RepoFileRecord, tokens: set[str]) -> int:
    path = record.path.lower()
    score = 0
    for token in tokens:
        if token in path:
            score += 20
    if path.startswith("spark_agent/core/"):
        score += 12
    elif path in {"spark_agent/cli.py", "spark_agent/config.py"}:
        score += 10
    elif path.startswith("spark_agent/tools/"):
        score += 8
    elif path.startswith("tests/"):
        score += 4
    if path.endswith(("readme.md", "pyproject.toml")):
        score += 6
    if record.language == "python":
        score += 4
    return score


def _outline_text(record: RepoFileRecord, text: str) -> list[dict[str, Any]]:
    if record.language == "python":
        return _python_outline(text)
    return _generic_outline(text)


def _python_outline(text: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    symbols: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "line": node.lineno,
                    "methods": [
                        {"name": item.name, "line": item.lineno}
                        for item in node.body
                        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    ][:20],
                }
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            symbols.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "line": node.lineno,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                }
            )
    return symbols[:60]


def _generic_outline(text: str) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("# ", "## ", "### ")):
            symbols.append({"kind": "heading", "line": line_no, "text": stripped[:160]})
        elif stripped.startswith(("[project]", "[provider]", "[agent]", "[tool.")):
            symbols.append({"kind": "section", "line": line_no, "text": stripped[:160]})
    return symbols[:60]


def _compact_preview(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:40])[:2_000]
