from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from spark_agent.core.prompt_engine import JsonValue, PromptEngine
from spark_agent.core.types import JsonObject, ToolSpec


class VLLMRequestError(RuntimeError):
    """Raised when the vLLM OpenAI-compatible endpoint cannot complete a request."""


@dataclass(frozen=True, slots=True)
class VLLMClientConfig:
    base_url: str = "http://localhost:8000"
    model: str = "deepseek-ai/DeepSeek-V4"
    api_key: str | None = None
    timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    temperature: float = 0.0
    max_tokens: int = 2048
    max_tool_result_chars: int = 6_000

    @property
    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    @property
    def headers(self) -> dict[str, str]:
        api_key = self.api_key or os.getenv("SPARK_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {}
        return {"Authorization": f"Bearer {api_key}"}


@dataclass(frozen=True, slots=True)
class AgentTurn:
    content: str | None
    reasoning_content: str | None
    tool_calls: tuple[JsonObject, ...] = ()
    raw_message: JsonObject = field(default_factory=dict)


class AgentExecutor:
    """Thought -> Action -> Observation executor for vLLM DeepSeek tool-calling."""

    def __init__(
        self,
        *,
        prompt_engine: PromptEngine,
        config: VLLMClientConfig,
        tools: Sequence[ToolSpec] = (),
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.prompt_engine = prompt_engine
        self.config = config
        self._tool_specs = {tool.name: tool for tool in tools}
        timeout = httpx.Timeout(
            timeout=config.timeout_s,
            connect=config.connect_timeout_s,
        )
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> AgentExecutor:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def run(self, user_task: str, *, max_turns: int = 8) -> str:
        self.prompt_engine.append_user_message(user_task)
        final_answer: str | None = None
        for _ in range(max_turns):
            turn = await self.step()
            if not turn.tool_calls:
                final_answer = turn.content or ""
                break
        if final_answer is None:
            raise RuntimeError(f"Agent did not finish within {max_turns} turns")
        return final_answer

    async def step(self) -> AgentTurn:
        completion = await self._chat_completion()
        message = self._extract_message(completion)
        turn = self._message_to_turn(message)
        self.prompt_engine.append_assistant_message(
            content=turn.content,
            reasoning_content=turn.reasoning_content,
            tool_calls=turn.tool_calls,
        )
        if turn.tool_calls:
            await self._execute_tool_calls(turn.tool_calls)
        return turn

    async def _chat_completion(self) -> JsonObject:
        payload: JsonObject = {
            "model": self.config.model,
            "messages": self.prompt_engine.render_messages(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self._tool_specs:
            payload["tools"] = [spec.definition for spec in self._tool_specs.values()]
            payload["tool_choice"] = "auto"

        try:
            response = await self._client.post(
                self.config.chat_completions_url,
                json=payload,
                headers=self.config.headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise VLLMRequestError(
                f"vLLM request timed out after {self.config.timeout_s:g}s ({type(exc).__name__})"
            ) from exc
        except httpx.NetworkError as exc:
            detail = str(exc) or type(exc).__name__
            raise VLLMRequestError(f"vLLM network error: {detail}") from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:1000]
            raise VLLMRequestError(
                f"vLLM returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise VLLMRequestError("vLLM returned invalid JSON") from exc

    @staticmethod
    def _extract_message(completion: Mapping[str, Any]) -> JsonObject:
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLLMRequestError("vLLM response does not contain choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise VLLMRequestError("vLLM response choice does not contain a message")
        return dict(message)

    @staticmethod
    def _message_to_turn(message: Mapping[str, Any]) -> AgentTurn:
        tool_calls = message.get("tool_calls") or ()
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, str):
            raise VLLMRequestError("tool_calls must be a sequence")
        return AgentTurn(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content") or message.get("reasoning"),
            tool_calls=tuple(dict(tool_call) for tool_call in tool_calls),
            raw_message=dict(message),
        )

    async def _execute_tool_calls(self, tool_calls: Sequence[JsonObject]) -> None:
        results: list[tuple[str, str, JsonValue | str] | None] = [None] * len(tool_calls)

        async def run_one(index: int, tool_call: JsonObject) -> None:
            tool_call_id = str(tool_call.get("id", ""))
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                results[index] = (tool_call_id, "unknown", {"error": "missing function payload"})
                return

            name = str(function.get("name", ""))
            spec = self._tool_specs.get(name)
            if spec is None:
                results[index] = (tool_call_id, name, {"error": f"unknown tool: {name}"})
                return

            try:
                arguments = self._decode_arguments(function.get("arguments", {}))
                result = await spec.handler(arguments)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            results[index] = (tool_call_id, name, result)

        async with asyncio.TaskGroup() as task_group:
            for index, tool_call in enumerate(tool_calls):
                task_group.create_task(run_one(index, tool_call))

        for item in results:
            if item is None:
                continue
            tool_call_id, name, result = item
            self.prompt_engine.append_tool_result(
                tool_call_id=tool_call_id,
                name=name,
                content=self._truncate_tool_result(result),
            )

    @staticmethod
    def _decode_arguments(arguments: Any) -> JsonObject:
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid tool arguments JSON: {arguments[:200]}") from exc
            if isinstance(decoded, dict):
                return decoded
        raise ValueError("tool arguments must be a JSON object")

    def _truncate_tool_result(self, result: JsonValue | str) -> JsonValue | str:
        max_chars = max(500, self.config.max_tool_result_chars)
        if isinstance(result, str):
            if len(result) <= max_chars:
                return result
            return f"{result[:max_chars]}\n...[tool output truncated to {max_chars} chars]"

        rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(rendered) <= max_chars:
            return result
        return {
            "truncated": True,
            "max_chars": max_chars,
            "preview": rendered[:max_chars],
        }
