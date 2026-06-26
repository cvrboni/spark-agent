from __future__ import annotations

import pytest

from spark_agent.core.prompt_engine import (
    DYNAMIC_HEADER,
    STATIC_HEADER,
    PromptBlock,
    PromptEngine,
)


def test_static_hash_is_stable_when_dynamic_events_are_appended() -> None:
    engine = PromptEngine(
        static_blocks=[
            PromptBlock("system", "stable system"),
            PromptBlock.from_jsonable("tools", {"b": 2, "a": 1}),
        ],
    )
    static_hash = engine.static_hash
    static_prefix = engine.static_prefix

    engine.append_user_message("first turn")
    engine.append_assistant_message(content="thinking", reasoning_content="internal plan")
    engine.append_tool_result(tool_call_id="call_1", name="repo_grep", content={"ok": True})

    assert engine.static_hash == static_hash
    assert engine.static_prefix == static_prefix
    assert engine.render_messages()[0]["content"].startswith(STATIC_HEADER)
    assert DYNAMIC_HEADER in engine.render_messages()[0]["content"]


def test_static_json_blocks_are_canonical() -> None:
    first = PromptBlock.from_jsonable("tools", {"z": [3, 2], "a": {"b": True}})
    second = PromptBlock.from_jsonable("tools", {"a": {"b": True}, "z": [3, 2]})

    assert first.content == second.content


def test_dynamic_role_validation() -> None:
    engine = PromptEngine(static_blocks=[PromptBlock("system", "stable")])

    with pytest.raises(ValueError, match="Unsupported chat role"):
        engine.append_event({"role": "developer", "content": "not supported"})

