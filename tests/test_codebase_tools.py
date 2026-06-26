from __future__ import annotations

import textwrap

import pytest

from spark_agent.tools.codebase import repo_grep, tool_definitions, view_file_outline


@pytest.mark.asyncio
async def test_view_file_outline_returns_python_symbols(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        textwrap.dedent(
            """
            class Worker:
                def run(self, value: str) -> str:
                    return value

            async def build() -> Worker:
                return Worker()
            """
        ),
        encoding="utf-8",
    )

    result = await view_file_outline(source)

    assert result["parser"] == "python_ast"
    assert result["symbols"][0]["kind"] == "class"
    assert result["symbols"][0]["name"] == "Worker"
    assert result["symbols"][1]["async"] is True


@pytest.mark.asyncio
async def test_repo_grep_returns_bounded_matches(tmp_path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("alpha\nneedle one\nbeta\nneedle two\n", encoding="utf-8")

    result = await repo_grep("needle", root=tmp_path, glob="*.py", max_matches=1)

    assert result["match_count"] == 1
    assert "needle" in result["matches"][0]["text"]


def test_tool_definitions_are_openai_function_tools() -> None:
    definitions = tool_definitions()

    assert {item["function"]["name"] for item in definitions} == {
        "repo_grep",
        "view_file_outline",
    }
    assert all(item["type"] == "function" for item in definitions)

