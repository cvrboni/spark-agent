from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Static provider hints that belong in the cacheable prompt prefix."""

    name: str
    tool_call_contract: str

    @classmethod
    def from_model(cls, *, base_url: str, model: str) -> ProviderProfile:
        model_key = model.lower()
        base_key = base_url.lower()
        if "deepseek" in model_key and "v1" in base_key:
            return cls(
                name="vllm-deepseek",
                tool_call_contract=_native_with_tagged_fallback_contract(),
            )
        if any(key in base_key for key in ("ollama", "llama.cpp", "llamacpp")):
            return cls(
                name="openai-compatible-tagged-fallback",
                tool_call_contract=_tagged_fallback_first_contract(),
            )
        return cls(
            name="openai-compatible-local",
            tool_call_contract=_native_with_tagged_fallback_contract(),
        )


def _native_with_tagged_fallback_contract() -> str:
    return (
        "Tool-call contract:\n"
        "- Prefer the provider's structured tool_calls field when available.\n"
        "- If structured tool_calls are unavailable or unreliable, emit only one or more stable "
        "<tool_call>...</tool_call> blocks and no prose for that assistant turn.\n"
        "- Tagged fallback shape: "
        '<tool_call>{"name":"tool_name","arguments":{"key":"value"}}</tool_call>\n'
        "- arguments must match the provided JSON schema exactly; omit unknown keys.\n"
        "- Final answers must be plain text and must not contain tool-call syntax."
    )


def _tagged_fallback_first_contract() -> str:
    return (
        "Tool-call contract:\n"
        "- Emit tool requests as stable <tool_call>...</tool_call> blocks and no prose for that "
        "assistant turn.\n"
        "- Tagged fallback shape: "
        '<tool_call>{"name":"tool_name","arguments":{"key":"value"}}</tool_call>\n'
        "- arguments must match the provided JSON schema exactly; omit unknown keys.\n"
        "- Final answers must be plain text and must not contain tool-call syntax."
    )
