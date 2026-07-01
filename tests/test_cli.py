from __future__ import annotations

from spark_agent.cli import main, resolve_retrieval_mode
from spark_agent.config import SparkAgentConfig


def test_cli_init_writes_config(tmp_path, capsys) -> None:
    path = tmp_path / "config.toml"

    code = main(
        [
            "--config",
            str(path),
            "init",
            "--base-url",
            "http://llm.example:8000/v1",
            "--model",
            "deepseek-v4-flash",
            "--language",
            "auto",
        ]
    )

    assert code == 0
    assert path.exists()
    assert "Wrote config" in capsys.readouterr().out
    assert SparkAgentConfig.from_file(path).base_url == "http://llm.example:8000/v1"


def test_cli_init_without_options_does_not_overwrite_existing_config(tmp_path, capsys) -> None:
    path = tmp_path / "config.toml"
    main(["--config", str(path), "init", "--base-url", "http://llm.example:8000/v1"])

    code = main(["--config", str(path), "init"])

    assert code == 0
    assert SparkAgentConfig.from_file(path).base_url == "http://llm.example:8000/v1"
    assert "Config already exists" in capsys.readouterr().out


def test_cli_init_with_explicit_option_updates_existing_config(tmp_path) -> None:
    path = tmp_path / "config.toml"
    main(["--config", str(path), "init", "--base-url", "http://localhost:8000/v1"])

    code = main(["--config", str(path), "init", "--base-url", "http://llm.example:8000/v1"])

    assert code == 0
    assert SparkAgentConfig.from_file(path).base_url == "http://llm.example:8000/v1"


def test_cli_prompt_preview_uses_config(tmp_path, capsys) -> None:
    path = tmp_path / "config.toml"
    main(["--config", str(path), "init", "--language", "it"])

    code = main(["--config", str(path), "prompt-preview"])

    assert code == 0
    output = capsys.readouterr().out
    assert "<spark_static_prefix_v1>" in output
    assert "Rispondi in italiano" in output
    assert "Tool-call contract" in output
    assert "<tool_call>" in output


def test_resolve_retrieval_mode_auto_follows_prefer_local_tools() -> None:
    with_tools = SparkAgentConfig(prefer_local_tools=True)
    without_tools = SparkAgentConfig(prefer_local_tools=False)

    assert resolve_retrieval_mode(with_tools, "auto") == "model"
    assert resolve_retrieval_mode(without_tools, "auto") == "local"
    assert resolve_retrieval_mode(with_tools, "none") == "none"


def test_cli_doctor_rejects_secret_in_api_key_env(tmp_path, capsys) -> None:
    path = tmp_path / "config.toml"
    secret = "e57aa47a3e76800de34f946cd20794a0"
    SparkAgentConfig(api_key_env=secret).to_file(path)

    code = main(["--config", str(path), "doctor"])

    output = capsys.readouterr().out
    assert code == 2
    assert secret not in output
    assert "e57a...94a0" in output
    assert "Provider retries" in output
    assert "Repo index cache" in output
    assert "must be an environment variable name" in output
