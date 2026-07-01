from __future__ import annotations

import json

from spark_agent.core.repo_index import CACHE_PATH, RepoIndex


def test_repo_index_builds_cache_and_ignores_internal_dirs(tmp_path) -> None:
    source = tmp_path / "spark_agent" / "core" / "sample.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n", encoding="utf-8")
    internal = tmp_path / ".spark-agent" / "sessions" / "x.jsonl"
    internal.parent.mkdir(parents=True)
    internal.write_text("{}", encoding="utf-8")

    index = RepoIndex.build(tmp_path)

    assert [record.path for record in index.records] == ["spark_agent/core/sample.py"]
    cache = tmp_path / CACHE_PATH
    assert cache.exists()
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert "spark_agent/core/sample.py" in data["files"]


def test_repo_index_updates_hash_when_file_changes(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = RepoIndex.build(tmp_path).records[0]

    source.write_text("value = 22\n", encoding="utf-8")
    second = RepoIndex.build(tmp_path).records[0]

    assert first.sha256 != second.sha256
