from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Self

APP_DIR_NAME = "spark-agent"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_LANGUAGE = "auto"
ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HEX_SECRET_RE = re.compile(r"^[a-f0-9]{24,}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SparkAgentConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key_env: str = "SPARK_AGENT_API_KEY"
    language: str = DEFAULT_LANGUAGE
    timeout_s: float = 120.0
    max_tokens: int = 2048
    max_retries: int = 1
    retry_backoff_s: float = 0.25
    sandbox_backend: str = "local"
    sandbox_image: str = "python:3.13-slim"
    repo_context_budget: int = 24_000
    prefer_local_tools: bool = True
    approval_policy: str = "interactive"
    stream_responses: bool = True

    @classmethod
    def from_file(cls, path: Path | None = None) -> Self:
        config_path = path or default_config_path()
        if not config_path.exists():
            return cls.from_env(cls())
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
        provider = raw.get("provider", {})
        agent = raw.get("agent", {})
        config = cls(
            base_url=str(provider.get("base_url", DEFAULT_BASE_URL)),
            model=str(provider.get("model", DEFAULT_MODEL)),
            api_key_env=str(provider.get("api_key_env", "SPARK_AGENT_API_KEY")),
            timeout_s=float(provider.get("timeout_s", 120.0)),
            max_tokens=int(provider.get("max_tokens", 2048)),
            max_retries=int(provider.get("max_retries", 1)),
            retry_backoff_s=float(provider.get("retry_backoff_s", 0.25)),
            sandbox_backend=str(provider.get("sandbox_backend", "local")),
            sandbox_image=str(provider.get("sandbox_image", "python:3.13-slim")),
            language=str(agent.get("language", DEFAULT_LANGUAGE)),
            repo_context_budget=int(agent.get("repo_context_budget", 24_000)),
            prefer_local_tools=bool(agent.get("prefer_local_tools", True)),
            approval_policy=str(agent.get("approval_policy", "interactive")),
            stream_responses=bool(agent.get("stream_responses", True)),
        )
        return cls.from_env(config)

    @classmethod
    def from_env(cls, config: Self) -> Self:
        return cls(
            base_url=os.getenv("SPARK_AGENT_BASE_URL", config.base_url),
            model=os.getenv("SPARK_AGENT_MODEL", config.model),
            api_key_env=os.getenv("SPARK_AGENT_API_KEY_ENV", config.api_key_env),
            language=os.getenv("SPARK_AGENT_LANGUAGE", config.language),
            timeout_s=float(os.getenv("SPARK_AGENT_TIMEOUT_S", str(config.timeout_s))),
            max_tokens=int(os.getenv("SPARK_AGENT_MAX_TOKENS", str(config.max_tokens))),
            max_retries=int(os.getenv("SPARK_AGENT_MAX_RETRIES", str(config.max_retries))),
            retry_backoff_s=float(
                os.getenv("SPARK_AGENT_RETRY_BACKOFF_S", str(config.retry_backoff_s))
            ),
            sandbox_backend=os.getenv("SPARK_AGENT_SANDBOX_BACKEND", config.sandbox_backend),
            sandbox_image=os.getenv("SPARK_AGENT_SANDBOX_IMAGE", config.sandbox_image),
            repo_context_budget=int(
                os.getenv("SPARK_AGENT_REPO_CONTEXT_BUDGET", str(config.repo_context_budget))
            ),
            prefer_local_tools=_env_bool(
                os.getenv("SPARK_AGENT_PREFER_LOCAL_TOOLS"),
                default=config.prefer_local_tools,
            ),
            approval_policy=os.getenv("SPARK_AGENT_APPROVAL_POLICY", config.approval_policy),
            stream_responses=_env_bool(
                os.getenv("SPARK_AGENT_STREAM_RESPONSES"),
                default=config.stream_responses,
            ),
        )

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) or os.getenv("OPENAI_API_KEY")

    @property
    def api_key_env_is_valid_name(self) -> bool:
        return bool(ENV_VAR_NAME_RE.fullmatch(self.api_key_env))

    @property
    def api_key_env_looks_like_secret(self) -> bool:
        return bool(HEX_SECRET_RE.fullmatch(self.api_key_env)) or self.api_key_env.startswith(
            ("sk-", "sk_", "Bearer ")
        )

    def to_toml(self) -> str:
        prefer_local_tools = "true" if self.prefer_local_tools else "false"
        stream_responses = "true" if self.stream_responses else "false"
        return (
            "# SparkAgent local configuration\n"
            "# Optimized for OpenAI-compatible local inference servers such as vLLM.\n\n"
            "[provider]\n"
            f"base_url = {_toml_string(self.base_url)}\n"
            f"model = {_toml_string(self.model)}\n"
            f"api_key_env = {_toml_string(self.api_key_env)}\n"
            f"timeout_s = {self.timeout_s:g}\n"
            f"max_tokens = {self.max_tokens}\n"
            f"max_retries = {self.max_retries}\n"
            f"retry_backoff_s = {self.retry_backoff_s:g}\n"
            f"sandbox_backend = {_toml_string(self.sandbox_backend)} # local, docker, podman\n"
            f"sandbox_image = {_toml_string(self.sandbox_image)}\n\n"
            "[agent]\n"
            f"language = {_toml_string(self.language)} # auto, it, en\n"
            f"repo_context_budget = {self.repo_context_budget}\n"
            f"prefer_local_tools = {prefer_local_tools}\n"
            f"approval_policy = {_toml_string(self.approval_policy)} # auto, interactive, never\n"
            f"stream_responses = {stream_responses}\n"
        )

    def to_file(self, path: Path) -> None:
        path.write_text(self.to_toml(), encoding="utf-8")


def default_config_path() -> Path:
    config_home = os.getenv("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / APP_DIR_NAME / "config.toml"
    return Path.home() / ".config" / APP_DIR_NAME / "config.toml"


def write_default_config(
    path: Path | None = None,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    sandbox_backend: str = "local",
    sandbox_image: str = "python:3.13-slim",
    force: bool = False,
) -> Path:
    config_path = path or default_config_path()
    if config_path.exists() and not force:
        return config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    SparkAgentConfig(
        base_url=base_url,
        model=model,
        language=language,
        sandbox_backend=sandbox_backend,
        sandbox_image=sandbox_image,
    ).to_file(config_path)
    return config_path


def _env_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
