from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

type JsonObject = dict[str, Any]

RISKY_TOOLS = frozenset({"apply_patch", "run_command"})


class ApprovalPolicy(StrEnum):
    AUTO = "auto"
    INTERACTIVE = "interactive"
    NEVER = "never"

    @classmethod
    def from_value(cls, value: str) -> ApprovalPolicy:
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"approval_policy must be one of: {', '.join(member.value for member in cls)}"
            ) from exc


def summarize_tool_action(name: str, arguments: Mapping[str, Any]) -> str:
    if name == "apply_patch":
        patch = str(arguments.get("patch", ""))
        lines = patch.strip().splitlines()
        preview = "\n".join(lines[:12])
        if len(lines) > 12:
            preview += f"\n... ({len(lines) - 12} more lines)"
        return f"apply_patch:\n{preview}"
    if name == "run_command":
        command = arguments.get("command", [])
        if isinstance(command, list):
            return f"run_command: {' '.join(str(item) for item in command)}"
        return f"run_command: {command!r}"
    return f"{name}: {json.dumps(dict(arguments), ensure_ascii=False)[:500]}"


def request_approval(
    name: str,
    arguments: Mapping[str, Any],
    *,
    policy: ApprovalPolicy,
    interactive: bool,
) -> bool:
    if name not in RISKY_TOOLS:
        return True
    if policy is ApprovalPolicy.AUTO:
        return True
    if policy is ApprovalPolicy.NEVER:
        return False
    if not interactive or not sys.stdin.isatty():
        print(
            f"[spark-agent] approval required for {name} but stdin is not interactive; skipping.",
            file=sys.stderr,
            flush=True,
        )
        return False
    summary = summarize_tool_action(name, arguments)
    print(f"\n[spark-agent] approval required:\n{summary}\n", file=sys.stderr, flush=True)
    try:
        answer = input("Approve? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}
