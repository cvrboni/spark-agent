from __future__ import annotations

import json

from spark_agent.core.context_packer import ContextPacker


def test_context_packer_selects_task_relevant_file(tmp_path) -> None:
    core = tmp_path / "spark_agent" / "core"
    tools = tmp_path / "spark_agent" / "tools"
    core.mkdir(parents=True)
    tools.mkdir(parents=True)
    (core / "executor.py").write_text("class AgentExecutor:\n    pass\n", encoding="utf-8")
    (tools / "workspace.py").write_text("def workspace_tool_specs():\n    return []\n", encoding="utf-8")

    packed = ContextPacker.from_root(tmp_path).pack("fix AgentExecutor retry", max_files=1)
    data = json.loads(packed)

    assert data["selected_files"][0]["path"] == "spark_agent/core/executor.py"
    assert data["selected_files"][0]["outline"][0]["name"] == "AgentExecutor"


def test_context_packer_respects_character_budget(tmp_path) -> None:
    for index in range(10):
        path = tmp_path / f"file_{index}.py"
        path.write_text(f"def function_{index}():\n    return {index}\n", encoding="utf-8")

    packed = ContextPacker.from_root(tmp_path).pack("function", max_files=10, max_chars=1_200)

    assert len(packed) <= 1_260
    assert "selected_files" in packed
