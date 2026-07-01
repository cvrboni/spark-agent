from __future__ import annotations

import pytest

from spark_agent.core.sandbox import (
    CommandSandboxPolicy,
    LocalSandbox,
    SandboxViolationError,
    validate_patch,
    validate_read_path,
)


def test_command_sandbox_accepts_allowlisted_prefix() -> None:
    CommandSandboxPolicy.default().validate(["python", "-m", "pytest", "tests"])


def test_command_sandbox_rejects_unknown_command() -> None:
    with pytest.raises(SandboxViolationError, match="not allowlisted"):
        CommandSandboxPolicy.default().validate(["bash", "-lc", "pytest"])


def test_validate_patch_accepts_repo_relative_paths() -> None:
    validate_patch(
        """diff --git a/spark_agent/example.py b/spark_agent/example.py
--- a/spark_agent/example.py
+++ b/spark_agent/example.py
@@ -1 +1 @@
-old
+new
"""
    )


def test_validate_patch_rejects_path_escape() -> None:
    with pytest.raises(SandboxViolationError, match="escapes repository root"):
        validate_patch(
            """diff --git a/../outside.py b/../outside.py
--- a/../outside.py
+++ b/../outside.py
@@ -1 +1 @@
-old
+new
"""
        )


def test_validate_patch_rejects_internal_directory() -> None:
    with pytest.raises(SandboxViolationError, match="internal directory"):
        validate_patch(
            """diff --git a/.spark-agent/session b/.spark-agent/session
--- a/.spark-agent/session
+++ b/.spark-agent/session
@@ -1 +1 @@
-old
+new
"""
        )


@pytest.mark.asyncio
async def test_local_sandbox_runs_allowlisted_command(tmp_path) -> None:
    result = await LocalSandbox(tmp_path).run_command(["git", "status"], timeout_s=5)

    assert result.command == ("git", "status")
    assert isinstance(result.returncode, int)
    assert "stdout" in result.to_json(stdout_chars=10, stderr_chars=10)


@pytest.mark.asyncio
async def test_local_sandbox_applies_patch(tmp_path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("old\n", encoding="utf-8")
    patch = """diff --git a/sample.txt b/sample.txt
--- a/sample.txt
+++ b/sample.txt
@@ -1 +1 @@
-old
+new
"""

    result = await LocalSandbox(tmp_path).apply_patch(patch)

    assert result.returncode == 0
    assert source.read_text(encoding="utf-8") == "new\n"


def test_validate_read_path_rejects_sensitive_file(tmp_path) -> None:
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=secret\n", encoding="utf-8")

    with pytest.raises(SandboxViolationError, match="sensitive file"):
        validate_read_path(secret, tmp_path)


def test_validate_read_path_rejects_internal_directory(tmp_path) -> None:
    path = tmp_path / ".spark-agent" / "session.jsonl"
    path.parent.mkdir()
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(SandboxViolationError, match="internal directory"):
        validate_read_path(path, tmp_path)
