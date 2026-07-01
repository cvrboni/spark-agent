from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spark_agent.core.types import JsonObject


class ModelAdapterError(ValueError):
    """Raised when a model response cannot be normalized."""


@dataclass(frozen=True, slots=True)
class NormalizedModelMessage:
    content: str | None
    reasoning_content: str | None
    tool_calls: tuple[JsonObject, ...]
    raw_message: JsonObject


class ModelAdapter:
    """Normalize OpenAI-compatible and local-model tool-call responses.

    vLLM can expose structured tool calls for models that have a native parser. Some local
    models are more reliable with stable XML tags, so this adapter also accepts:

    <tool_call>{"name":"repo_grep","arguments":{"query":"PromptEngine"}}</tool_call>

    The fallback is intentionally narrow: arbitrary JSON in normal prose is not treated as a
    tool call.
    """

    _TOOL_CALL_RE = re.compile(
        r"<tool_call(?:\s+name=(?P<quote>[\"'])(?P<name>[^\"']+)(?P=quote))?\s*>(?P<body>.*?)</tool_call>", re.DOTALL)

    def normalize_message(self, message: Mapping[str, Any]) -> NormalizedModelMessage:
        raw_message = dict(message)
        content = _optional_string(message.get("content"))
        reasoning_content = _optional_string(
            message.get("reasoning_content") or message.get("reasoning")
        )
        tool_calls = self._normalize_tool_calls(message.get("tool_calls"))
        if not tool_calls and content:
            tool_calls = self._parse_tagged_tool_calls(content)
            if tool_calls and not _content_has_non_tool_text(content):
                content = None
        return NormalizedModelMessage(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            raw_message=raw_message,
        )

    def _normalize_tool_calls(self, value: Any) -> tuple[JsonObject, ...]:
        if value in (None, ""):
            return ()
        if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
            raise ModelAdapterError("tool_calls must be a sequence")
        normalized = []
        for index, item in enumerate(value):
            normalized.append(self._normalize_tool_call(item, index=index))
        return tuple(normalized)

    def _normalize_tool_call(self, value: Any, *, index: int) -> JsonObject:
        if not isinstance(value, Mapping):
            raise ModelAdapterError(f"tool call {index} must be an object")
        function = value.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments", {})
        else:
            name = value.get("name")
            arguments = value.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ModelAdapterError(f"tool call {index} is missing function name")
        call_id = value.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = _stable_tool_call_id(name, arguments, index)
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }

    def _parse_tagged_tool_calls(self, content: str) -> tuple[JsonObject, ...]:
        calls: list[JsonObject] = []
        for index, match in enumerate(self._TOOL_CALL_RE.finditer(content)):
            name = match.group("name")
            body = match.group("body").strip()
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ModelAdapterError("invalid JSON inside <tool_call> tag") from exc
            if isinstance(payload, Mapping) and name is None:
                calls.append(self._normalize_tool_call(payload, index=index))
                continue
            if not isinstance(payload, Mapping):
                raise ModelAdapterError("<tool_call> body must be a JSON object")
            if name is None:
                raise ModelAdapterError("<tool_call> tag is missing tool name")
            calls.append(
                self._normalize_tool_call(
                    {"name": name, "arguments": dict(payload)},
                    index=index,
                )
            )
        return tuple(calls)


def _stable_tool_call_id(name: str, arguments: Any, index: int) -> str:
    try:
        rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        rendered = repr(arguments)
    digest = hashlib.sha1(f"{index}:{name}:{rendered}".encode()).hexdigest()[:12]
    return f"call_{digest}"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _content_has_non_tool_text(content: str) -> bool:
    stripped = ModelAdapter._TOOL_CALL_RE.sub("", content).strip()
    return bool(stripped)
