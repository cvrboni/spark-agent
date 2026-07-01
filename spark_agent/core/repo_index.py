from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from spark_agent.core.sandbox import IGNORED_PATH_PARTS

CACHE_PATH = ".spark-agent/index/files.json"
MAX_INDEXED_FILE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class RepoFileRecord:
    path: str
    size: int
    mtime_ns: int
    sha256: str
    language: str


@dataclass(frozen=True, slots=True)
class RepoIndex:
    root: Path
    records: tuple[RepoFileRecord, ...]

    @classmethod
    def build(cls, root: Path, *, use_cache: bool = True) -> Self:
        resolved_root = root.resolve()
        cached = _load_cache(resolved_root) if use_cache else {}
        next_cache: dict[str, dict[str, Any]] = {}
        records: list[RepoFileRecord] = []
        for path in _iter_indexable_files(resolved_root):
            relative = path.relative_to(resolved_root).as_posix()
            stat = path.stat()
            cached_item = cached.get(relative)
            if (
                cached_item
                and cached_item.get("size") == stat.st_size
                and cached_item.get("mtime_ns") == stat.st_mtime_ns
            ):
                sha256 = str(cached_item["sha256"])
            else:
                sha256 = _sha256_file(path)
            record = RepoFileRecord(
                path=relative,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=sha256,
                language=_detect_language(path),
            )
            records.append(record)
            next_cache[relative] = asdict(record)
        records.sort(key=lambda item: item.path)
        if use_cache:
            _write_cache(resolved_root, next_cache)
        return cls(root=resolved_root, records=tuple(records))

    def top_level_entries(self) -> list[str]:
        entries = {
            record.path.split("/", 1)[0]
            for record in self.records
            if record.path and not record.path.startswith(".pytest")
        }
        return sorted(entries)


def _iter_indexable_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in IGNORED_PATH_PARTS for part in relative_parts):
            continue
        try:
            if path.stat().st_size > MAX_INDEXED_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _load_cache(root: Path) -> dict[str, dict[str, Any]]:
    path = root / CACHE_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    files = data.get("files")
    if not isinstance(files, dict):
        return {}
    return {str(key): dict(value) for key, value in files.items() if isinstance(value, dict)}


def _write_cache(root: Path, files: dict[str, dict[str, Any]]) -> None:
    path = root / CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "files": files}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".json": "json",
        ".md": "markdown",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
    }.get(suffix, suffix.removeprefix(".") or "text")
