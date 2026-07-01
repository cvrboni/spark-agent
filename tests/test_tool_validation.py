from __future__ import annotations

import pytest

from spark_agent.core.tool_validation import ToolArgumentValidationError, validate_tool_arguments
from spark_agent.tools.workspace import workspace_tool_definitions


def _definition(name: str):
    return next(item for item in workspace_tool_definitions() if item["function"]["name"] == name)


def test_validate_tool_arguments_accepts_valid_payload() -> None:
    arguments = validate_tool_arguments(
        "run_command",
        _definition("run_command"),
        {"command": ["pytest"], "timeout_s": 5},
    )

    assert arguments == {"command": ["pytest"], "timeout_s": 5}


def test_validate_tool_arguments_rejects_extra_properties() -> None:
    with pytest.raises(ToolArgumentValidationError, match="unexpected arguments"):
        validate_tool_arguments(
            "read_file",
            _definition("read_file"),
            {"path": "README.md", "unexpected": True},
        )


def test_validate_tool_arguments_rejects_wrong_type() -> None:
    with pytest.raises(ToolArgumentValidationError, match=r"run_command\.command"):
        validate_tool_arguments(
            "run_command",
            _definition("run_command"),
            {"command": "pytest"},
        )
