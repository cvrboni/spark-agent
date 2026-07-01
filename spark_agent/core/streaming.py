from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from spark_agent.core.types import JsonObject

type TokenCallback = Callable[[str], None]


def parse_sse_data_line(line: str) -> JsonObject | None:
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if not payload or payload == "[DONE]":
        return None
    decoded = json.loads(payload)
    if isinstance(decoded, dict):
        return decoded
    return None


def merge_tool_call_delta(accumulated: dict[int, JsonObject], delta: Mapping[str, Any]) -> None:
    index = int(delta.get("index", 0))
    entry = accumulated.setdefault(
        index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if delta.get("id"):
        entry["id"] = str(delta["id"])
    function = delta.get("function")
    if isinstance(function, Mapping):
        if function.get("name"):
            entry["function"]["name"] = str(entry["function"]["name"]) + str(function["name"])
        if function.get("arguments"):
            entry["function"]["arguments"] = str(entry["function"]["arguments"]) + str(
                function["arguments"]
            )


async def collect_streamed_completion(
    response: httpx.Response,
    *,
    on_content_token: TokenCallback | None = None,
    on_reasoning_token: TokenCallback | None = None,
) -> JsonObject:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, JsonObject] = {}

    async for line in response.aiter_lines():
        chunk = parse_sse_data_line(line)
        if chunk is None:
            continue
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            continue

        token = delta.get("content")
        if isinstance(token, str) and token:
            content_parts.append(token)
            if on_content_token is not None:
                on_content_token(token)

        reasoning_token = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning_token, str) and reasoning_token:
            reasoning_parts.append(reasoning_token)
            if on_reasoning_token is not None:
                on_reasoning_token(reasoning_token)

        tool_call_deltas = delta.get("tool_calls")
        if isinstance(tool_call_deltas, list):
            for tool_call_delta in tool_call_deltas:
                if isinstance(tool_call_delta, dict):
                    merge_tool_call_delta(tool_calls, tool_call_delta)

    message: JsonObject = {"role": "assistant"}
    if content_parts:
        message["content"] = "".join(content_parts)
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {"choices": [{"message": message}]}
