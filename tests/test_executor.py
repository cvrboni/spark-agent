from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from spark_agent.core.executor import AgentExecutor, VLLMClientConfig
from spark_agent.core.prompt_engine import PromptBlock, PromptEngine
from spark_agent.core.types import ToolSpec


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.closed = False
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.last_url = url
        self.last_headers = kwargs.get("headers")
        return FakeResponse(self.payload)

    async def aclose(self) -> None:
        self.closed = True


class FlakyClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    async def post(self, _url: str, **_kwargs: Any) -> FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("temporary local provider outage")
        return FakeResponse(self.payload)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_executor_consumes_structured_tool_calls_in_order() -> None:
    prompt = PromptEngine(static_blocks=[PromptBlock("system", "stable")])
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "reasoning_content": "I should inspect targeted files.",
                    "tool_calls": [
                        {
                            "id": "call_slow",
                            "function": {"name": "slow", "arguments": {"value": "first"}},
                        },
                        {
                            "id": "call_fast",
                            "function": {"name": "fast", "arguments": {"value": "second"}},
                        },
                    ],
                }
            }
        ]
    }

    async def slow(arguments: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        return {"value": arguments["value"]}

    async def fast(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"value": arguments["value"]}

    executor = AgentExecutor(
        prompt_engine=prompt,
        config=VLLMClientConfig(),
        tools=[
            ToolSpec(
                name="slow",
                definition={"type": "function", "function": {"name": "slow"}},
                handler=slow,
            ),
            ToolSpec(
                name="fast",
                definition={"type": "function", "function": {"name": "fast"}},
                handler=fast,
            ),
        ],
        client=FakeClient(payload),  # type: ignore[arg-type]
    )

    turn = await executor.step()
    events = prompt.dynamic_events

    assert turn.reasoning_content == "I should inspect targeted files."
    assert events[-2]["tool_call_id"] == "call_slow"
    assert events[-1]["tool_call_id"] == "call_fast"


def test_vllm_config_accepts_root_or_v1_base_url() -> None:
    assert (
        VLLMClientConfig(base_url="http://llm.example:8000").chat_completions_url
        == "http://llm.example:8000/v1/chat/completions"
    )
    assert (
        VLLMClientConfig(base_url="http://llm.example:8000/v1").chat_completions_url
        == "http://llm.example:8000/v1/chat/completions"
    )


def test_executor_accepts_reasoning_alias_from_vllm() -> None:
    turn = AgentExecutor._message_to_turn({"content": None, "reasoning": "hidden trace"})

    assert turn.reasoning_content == "hidden trace"


def test_executor_truncates_large_tool_results() -> None:
    prompt = PromptEngine(static_blocks=[PromptBlock("system", "stable")])
    executor = AgentExecutor(
        prompt_engine=prompt,
        config=VLLMClientConfig(max_tool_result_chars=500),
        client=FakeClient({"choices": [{"message": {"content": "ok"}}]}),  # type: ignore[arg-type]
    )

    result = executor._truncate_tool_result({"payload": "x" * 2000})

    assert isinstance(result, dict)
    assert result["truncated"] is True
    assert len(result["preview"]) <= 500


def test_executor_respects_context_budget_for_tool_results() -> None:
    prompt = PromptEngine(static_blocks=[PromptBlock("system", "stable")])
    prompt.append_user_message("x" * 5000)
    executor = AgentExecutor(
        prompt_engine=prompt,
        config=VLLMClientConfig(
            max_tool_result_chars=6000,
            repo_context_budget=5200,
        ),
        client=FakeClient({"choices": [{"message": {"content": "ok"}}]}),  # type: ignore[arg-type]
    )

    result = executor._truncate_tool_result({"payload": "y" * 4000})

    assert isinstance(result, dict)
    assert result.get("truncated") is True or result.get("error")


@pytest.mark.asyncio
async def test_executor_validates_tool_arguments_before_handler() -> None:
    prompt = PromptEngine(static_blocks=[PromptBlock("system", "stable")])
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "function": {
                                "name": "strict_tool",
                                "arguments": {"path": "README.md", "extra": True},
                            },
                        }
                    ],
                }
            }
        ]
    }
    called = False

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"ok": True, "arguments": arguments}

    executor = AgentExecutor(
        prompt_engine=prompt,
        config=VLLMClientConfig(),
        tools=[
            ToolSpec(
                name="strict_tool",
                definition={
                    "type": "function",
                    "function": {
                        "name": "strict_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                },
                handler=handler,
            )
        ],
        client=FakeClient(payload),  # type: ignore[arg-type]
    )

    await executor.step()

    assert called is False
    assert "unexpected arguments" in prompt.dynamic_events[-1]["content"]


@pytest.mark.asyncio
async def test_executor_sends_bearer_auth_header() -> None:
    prompt = PromptEngine(static_blocks=[PromptBlock("system", "stable")])
    payload = {"choices": [{"message": {"content": "ok"}}]}
    client = FakeClient(payload)
    executor = AgentExecutor(
        prompt_engine=prompt,
        config=VLLMClientConfig(
            base_url="http://llm.example:8000/v1",
            model="deepseek-v4-flash",
            api_key="secret",
        ),
        client=client,  # type: ignore[arg-type]
    )

    await executor.step()

    assert client.last_url == "http://llm.example:8000/v1/chat/completions"
    assert client.last_headers == {"Authorization": "Bearer secret"}


@pytest.mark.asyncio
async def test_executor_retries_transient_network_errors() -> None:
    prompt = PromptEngine(static_blocks=[PromptBlock("system", "stable")])
    client = FlakyClient({"choices": [{"message": {"content": "ok"}}]})
    executor = AgentExecutor(
        prompt_engine=prompt,
        config=VLLMClientConfig(max_retries=1, retry_backoff_s=0),
        client=client,  # type: ignore[arg-type]
    )

    turn = await executor.step()

    assert turn.content == "ok"
    assert client.calls == 2
