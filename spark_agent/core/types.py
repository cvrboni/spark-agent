from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from spark_agent.core.prompt_engine import JsonValue

type JsonObject = dict[str, Any]
type ToolHandler = Callable[[JsonObject], Awaitable[JsonValue | str]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    definition: JsonObject
    handler: ToolHandler

