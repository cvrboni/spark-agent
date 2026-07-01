from __future__ import annotations

from spark_agent.config import SparkAgentConfig, write_default_config


def test_write_and_read_config(tmp_path) -> None:
    path = tmp_path / "config.toml"

    written = write_default_config(
        path,
        base_url="http://llm.example:8000/v1",
        model="deepseek-v4-flash",
        language="it",
    )
    config = SparkAgentConfig.from_file(written)

    assert config.base_url == "http://llm.example:8000/v1"
    assert config.model == "deepseek-v4-flash"
    assert config.language == "it"
    assert config.max_retries == 1
    assert config.retry_backoff_s == 0.25


def test_env_overrides_config(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.toml"
    write_default_config(path, base_url="http://localhost:8000/v1", model="local")

    monkeypatch.setenv("SPARK_AGENT_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("SPARK_AGENT_MODEL", "override-model")
    monkeypatch.setenv("SPARK_AGENT_MAX_RETRIES", "3")
    monkeypatch.setenv("SPARK_AGENT_RETRY_BACKOFF_S", "0.5")

    config = SparkAgentConfig.from_file(path)

    assert config.base_url == "http://example.test/v1"
    assert config.model == "override-model"
    assert config.max_retries == 3
    assert config.retry_backoff_s == 0.5
