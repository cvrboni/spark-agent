from __future__ import annotations

import pytest

from spark_agent.core.model_adapter import ModelAdapter, ModelAdapterError


def test_model_adapter_normalizes_structured_tool_calls() -> None:
    message = {
        "content": None,
        "reasoning": "inspect first",
        "tool_calls": [
            {
                "function": {
                    "name": "repo_grep",
                    "arguments": '{"query":"PromptEngine"}',
                }
            }
        ],
    }

    normalized = ModelAdapter().normalize_message(message)

    assert normalized.reasoning_content == "inspect first"
    assert normalized.tool_calls[0]["type"] == "function"
    assert normalized.tool_calls[0]["function"]["name"] == "repo_grep"
    assert normalized.tool_calls[0]["function"]["arguments"] == '{"query":"PromptEngine"}'
    assert normalized.tool_calls[0]["id"].startswith("call_")


def test_model_adapter_accepts_tagged_tool_call_fallback() -> None:
    message = {
        "content": '<tool_call name="repo_grep">{"query":"PromptEngine"}</tool_call>',
    }

    normalized = ModelAdapter().normalize_message(message)

    assert normalized.content is None
    assert normalized.tool_calls[0]["function"] == {
        "name": "repo_grep",
        "arguments": {"query": "PromptEngine"},
    }


def test_model_adapter_rejects_invalid_tagged_tool_call_json() -> None:
    with pytest.raises(ModelAdapterError, match="invalid JSON"):
        ModelAdapter().normalize_message({"content": "<tool_call>{bad json}</tool_call>"})
