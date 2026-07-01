from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


class SandboxViolationError(PermissionError):
    """Raised when a tool request violates the local sandbox policy."""


IGNORED_PATH_PARTS = frozenset({".git", ".spark-agent", ".venv", "__pycache__", "node_modules"})
SENSITIVE_FILE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
SENSITIVE_SUFFIXES = (".key", ".pem", ".p12", ".pfx")


@dataclass(frozen=True, slots=True)
class CommandSandboxPolicy:
    allowed_prefixes: tuple[tuple[str, ...], ...]

    @classmethod
    def default(cls) -> CommandSandboxPolicy:
        return cls(
            allowed_prefixes=(
                ("git", "status"),
                ("git", "diff"),
                ("python", "-m", "pytest"),
                ("python", "-m", "compileall"),
                ("python3", "-m", "pytest"),
                ("python3", "-m", "compileall"),
                ("python3.13", "-m", "pytest"),
                ("python3.13", "-m", "compileall"),
                ("pytest",),
                ("ruff", "check"),
            )
        )

    def validate(self, command: list[str]) -> None:
        if not command:
            raise SandboxViolationError("command must not be empty")
        if not any(tuple(command[: len(prefix)]) == prefix for prefix in self.allowed_prefixes):
            raise SandboxViolationError(f"command is not allowlisted: {shlex.join(command)}")


@dataclass(frozen=True, slots=True)
class SandboxProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def to_json(self, *, stdout_chars: int, stderr_chars: int) -> dict[str, object]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout[:stdout_chars],
            "stderr": self.stderr[:stderr_chars],
        }


@dataclass(frozen=True, slots=True)
class LocalSandbox:
    """Default local subprocess sandbox.

    This backend intentionally uses exec-style subprocess calls, never a shell. It enforces the
    repository cwd, command allowlist, patch path validation, bounded output, and timeouts.
    Stronger isolation backends can implement the same behavior later.
    """

    root: Path
    command_policy: CommandSandboxPolicy = field(default_factory=CommandSandboxPolicy.default)

    async def run_command(self, command: list[str], *, timeout_s: float) -> SandboxProcessResult:
        self.command_policy.validate(command)
        timeout_s = max(1.0, min(float(timeout_s), 120.0))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.root,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"command timed out after {timeout_s:g}s: {shlex.join(command)}"
            ) from exc
        return SandboxProcessResult(
            command=tuple(command),
            returncode=process.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def apply_patch(self, patch: str, *, timeout_s: float = 30.0) -> SandboxProcessResult:
        validate_patch(patch)
        command = ["git", "apply", "--whitespace=nowarn", "--"]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.root,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(patch.encode("utf-8")),
                timeout=max(1.0, min(float(timeout_s), 120.0)),
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TimeoutError("git apply timed out") from exc
        return SandboxProcessResult(
            command=tuple(command),
            returncode=process.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )


def validate_read_path(path: Path, root: Path) -> None:
    relative = path.resolve().relative_to(root.resolve())
    parts = set(relative.parts)
    if parts & IGNORED_PATH_PARTS:
        raise SandboxViolationError(f"path targets ignored or internal directory: {relative}")
    name = path.name
    if name in SENSITIVE_FILE_NAMES or name.endswith(SENSITIVE_SUFFIXES):
        raise SandboxViolationError(f"refusing to read sensitive file: {relative}")


def validate_patch(patch: str) -> None:
    if not patch.strip():
        raise SandboxViolationError("patch must be non-empty")
    paths = _extract_patch_paths(patch)
    if not paths:
        raise SandboxViolationError("patch must contain at least one file path")
    for path in paths:
        _validate_relative_repo_path(path)


def _extract_patch_paths(patch: str) -> set[str]:
    paths: set[str] = set()
    in_hunk = False
    for raw_line in patch.splitlines():
        if raw_line.startswith("diff --git "):
            in_hunk = False
            continue
        if raw_line.startswith("@@ "):
            in_hunk = True
            continue
        if in_hunk:
            continue
        if raw_line.startswith(("--- ", "+++ ")):
            value = raw_line[4:].strip()
        elif raw_line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            value = raw_line.split(" ", 2)[2].strip()
        else:
            continue
        if value == "/dev/null":
            continue
        path = value.split("\t", 1)[0].strip()
        if path.startswith(("a/", "b/")):
            path = path[2:]
        paths.add(path)
    return paths


def _validate_relative_repo_path(value: str) -> None:
    path = PurePosixPath(value)
    if value.startswith("/") or path.is_absolute():
        raise SandboxViolationError(f"patch path must be relative: {value}")
    if not value or value in {".", ".."}:
        raise SandboxViolationError(f"invalid patch path: {value}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise SandboxViolationError(f"patch path escapes repository root: {value}")
    if any(part in IGNORED_PATH_PARTS for part in path.parts):
        raise SandboxViolationError(f"patch path targets ignored or internal directory: {value}")
