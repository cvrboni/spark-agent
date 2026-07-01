from __future__ import annotations

from spark_agent.core.provider import ProviderProfile


def test_provider_profile_detects_vllm_deepseek() -> None:
    profile = ProviderProfile.from_model(
        base_url="http://localhost:8000/v1",
        model="deepseek-v4-flash",
    )

    assert profile.name == "vllm-deepseek"
    assert "structured tool_calls" in profile.tool_call_contract
    assert "<tool_call>" in profile.tool_call_contract


def test_provider_profile_detects_tagged_fallback_gateways() -> None:
    profile = ProviderProfile.from_model(
        base_url="http://localhost:11434/ollama/v1",
        model="qwen-coder",
    )

    assert profile.name == "openai-compatible-tagged-fallback"
    assert "Emit tool requests as stable" in profile.tool_call_contract
