from __future__ import annotations

from spark_agent.cli import main


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


def test_cli_prompt_preview_uses_config(tmp_path, capsys) -> None:
    path = tmp_path / "config.toml"
    main(["--config", str(path), "init", "--language", "it"])

    code = main(["--config", str(path), "prompt-preview"])

    assert code == 0
    output = capsys.readouterr().out
    assert "<spark_static_prefix_v1>" in output
    assert "Rispondi in italiano" in output
