"""SparkAgent: cache-stable async agent primitives for local vLLM clusters."""

from __future__ import annotations

from typing import Any

__all__ = ["AgentExecutor", "PromptBlock", "PromptEngine", "ToolSpec", "VLLMClientConfig"]


def __getattr__(name: str) -> Any:
    if name in {"PromptBlock", "PromptEngine"}:
        from spark_agent.core.prompt_engine import PromptBlock, PromptEngine

        return {"PromptBlock": PromptBlock, "PromptEngine": PromptEngine}[name]
    if name in {"AgentExecutor", "VLLMClientConfig"}:
        from spark_agent.core.executor import AgentExecutor, VLLMClientConfig

        return {"AgentExecutor": AgentExecutor, "VLLMClientConfig": VLLMClientConfig}[name]
    if name == "ToolSpec":
        from spark_agent.core.types import ToolSpec

        return ToolSpec
    raise AttributeError(name)
