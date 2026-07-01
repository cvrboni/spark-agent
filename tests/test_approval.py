from __future__ import annotations

import pytest

from spark_agent.core.approval import (
    RISKY_TOOLS,
    ApprovalPolicy,
    request_approval,
    summarize_tool_action,
)


def test_risky_tools_include_patch_and_run() -> None:
    assert "apply_patch" in RISKY_TOOLS
    assert "run_command" in RISKY_TOOLS
    assert "read_file" not in RISKY_TOOLS


def test_summarize_tool_action_for_patch() -> None:
    summary = summarize_tool_action("apply_patch", {"patch": "--- a\n+++ b\n@@\n+line\n"})
    assert "apply_patch" in summary
    assert "+++ b" in summary


def test_request_approval_auto_always_allows() -> None:
    assert request_approval(
        "apply_patch",
        {"patch": "x"},
        policy=ApprovalPolicy.AUTO,
        interactive=False,
    )


def test_request_approval_never_blocks() -> None:
    assert not request_approval(
        "run_command",
        {"command": ["pytest"]},
        policy=ApprovalPolicy.NEVER,
        interactive=True,
    )


def test_request_approval_interactive_without_tty_skips(monkeypatch) -> None:
    monkeypatch.setattr("spark_agent.core.approval.sys.stdin.isatty", lambda: False)
    assert not request_approval(
        "apply_patch",
        {"patch": "x"},
        policy=ApprovalPolicy.INTERACTIVE,
        interactive=True,
    )


def test_approval_policy_from_value_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="approval_policy"):
        ApprovalPolicy.from_value("maybe")
