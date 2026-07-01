from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Self

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type ChatMessage = dict[str, Any]


STATIC_HEADER = "<spark_static_prefix_v1>"
STATIC_FOOTER = "</spark_static_prefix_v1>"
DYNAMIC_HEADER = "<spark_append_only_tail_v1>"


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class PromptBlock:
    """A named immutable prompt block rendered into the cacheable static prefix."""

    name: str
    content: str
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("PromptBlock.name must be a non-empty stable identifier")
        if "\n" in self.name or "\r" in self.name:
            raise ValueError("PromptBlock.name must not contain newlines")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_jsonable(
        cls,
        name: str,
        value: JsonValue,
        *,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Self:
        return cls(name=name, content=_canonical_json(value), metadata=metadata or {})

    def render(self) -> str:
        metadata = _canonical_json(dict(self.metadata))
        return f"<block name={json.dumps(self.name)} metadata={metadata}>\n{self.content}\n</block>"


class PromptEngine:
    """Deterministic prompt composer optimized for vLLM prefix caching.

    Static blocks are rendered once during construction. Dynamic events are only appended after the
    static prefix and are never inserted before it, which keeps vLLM's prefix cache reusable across
    agent turns.
    """

    def __init__(
        self,
        *,
        static_blocks: Sequence[PromptBlock],
        dynamic_events: Iterable[ChatMessage] = (),
    ) -> None:
        if not static_blocks:
            raise ValueError("At least one static prompt block is required")
        self._static_blocks = tuple(static_blocks)
        self._static_prefix = self._render_static_prefix(self._static_blocks)
        self._static_hash = hashlib.sha256(self._static_prefix.encode("utf-8")).hexdigest()
        self._dynamic_events: list[ChatMessage] = []
        for event in dynamic_events:
            self.append_event(event)

    @property
    def static_hash(self) -> str:
        return self._static_hash

    @property
    def static_prefix(self) -> str:
        return self._static_prefix

    @property
    def dynamic_events(self) -> tuple[ChatMessage, ...]:
        return tuple(dict(event) for event in self._dynamic_events)

    def append_user_message(self, content: str) -> None:
        self.append_event({"role": "user", "content": content})

    def append_assistant_message(
        self,
        *,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        event: ChatMessage = {"role": "assistant"}
        if content is not None:
            event["content"] = content
        if reasoning_content is not None:
            event["reasoning_content"] = reasoning_content
        if tool_calls:
            event["tool_calls"] = [dict(tool_call) for tool_call in tool_calls]
        self.append_event(event)

    def append_tool_result(self, *, tool_call_id: str, name: str, content: JsonValue | str) -> None:
        rendered = content if isinstance(content, str) else _canonical_json(content)
        self.append_event(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": rendered,
            }
        )

    def append_observation(self, content: str) -> None:
        self.append_event({"role": "user", "content": f"Observation:\n{content}"})

    def append_event(self, event: Mapping[str, Any]) -> None:
        role = event.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            raise ValueError(f"Unsupported chat role: {role!r}")
        if role == "system" and self._dynamic_events:
            raise ValueError("Dynamic system events are only allowed before other dynamic events")
        self._dynamic_events.append(dict(event))

    def render_messages(self) -> list[ChatMessage]:
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": f"{self._static_prefix}\n{DYNAMIC_HEADER}",
            }
        ]
        messages.extend(dict(event) for event in self._dynamic_events)
        return messages

    def render_flat_prompt(self) -> str:
        dynamic = "\n".join(_canonical_json(dict(event)) for event in self._dynamic_events)
        return f"{self._static_prefix}\n{DYNAMIC_HEADER}\n{dynamic}"

    def dynamic_tail_bytes(self) -> int:
        if not self._dynamic_events:
            return 0
        dynamic = "\n".join(_canonical_json(dict(event)) for event in self._dynamic_events)
        return len(f"{DYNAMIC_HEADER}\n{dynamic}".encode())

    def remaining_context_budget(self, total_budget: int) -> int:
        return max(0, total_budget - self.dynamic_tail_bytes())

    @staticmethod
    def _render_static_prefix(blocks: Sequence[PromptBlock]) -> str:
        rendered_blocks = "\n".join(block.render() for block in blocks)
        return f"{STATIC_HEADER}\n{rendered_blocks}\n{STATIC_FOOTER}"

