from __future__ import annotations

import pytest

from spark_agent.tools.workspace import workspace_tool_specs


def _handler(specs, name: str):
    return next(spec.handler for spec in specs if spec.name == name)


@pytest.mark.asyncio
async def test_workspace_tools_read_files_under_root(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("print('ok')\n", encoding="utf-8")
    specs = workspace_tool_specs(tmp_path)

    result = await _handler(specs, "read_file")({"path": "sample.py"})

    assert result["path"] == "sample.py"
    assert "print" in result["content"]


@pytest.mark.asyncio
async def test_workspace_tools_reject_path_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    specs = workspace_tool_specs(tmp_path)

    with pytest.raises(PermissionError):
        await _handler(specs, "read_file")({"path": str(outside)})


@pytest.mark.asyncio
async def test_workspace_tools_reject_sensitive_file_reads(tmp_path) -> None:
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=secret\n", encoding="utf-8")
    specs = workspace_tool_specs(tmp_path)

    with pytest.raises(PermissionError, match="sensitive file"):
        await _handler(specs, "read_file")({"path": ".env"})


@pytest.mark.asyncio
async def test_workspace_tools_reject_internal_file_reads(tmp_path) -> None:
    internal = tmp_path / ".spark-agent" / "sessions.jsonl"
    internal.parent.mkdir()
    internal.write_text("{}", encoding="utf-8")
    specs = workspace_tool_specs(tmp_path)

    with pytest.raises(PermissionError, match="internal directory"):
        await _handler(specs, "read_file")({"path": ".spark-agent/sessions.jsonl"})


@pytest.mark.asyncio
async def test_run_command_rejects_non_allowlisted_command(tmp_path) -> None:
    specs = workspace_tool_specs(tmp_path)

    with pytest.raises(PermissionError):
        await _handler(specs, "run_command")({"command": ["rm", "-rf", "."]})
