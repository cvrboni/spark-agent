from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath


class SandboxViolationError(PermissionError):
    """Raised when a tool request violates the local sandbox policy."""


class SandboxBackend(StrEnum):
    LOCAL = "local"
    DOCKER = "docker"
    PODMAN = "podman"

    @classmethod
    def from_value(cls, value: str) -> SandboxBackend:
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"sandbox_backend must be one of: {allowed}") from exc


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


@dataclass(frozen=True, slots=True)
class ContainerSandbox:
    """Docker/Podman-backed sandbox for stronger local isolation.

    The active repository is mounted read-write at `/workspace`, the container runs with network
    disabled, a non-root UID/GID, dropped capabilities, and conservative process/memory limits.
    The image must provide the validation commands the agent is allowed to run.
    """

    root: Path
    engine: SandboxBackend
    image: str = "python:3.13-slim"
    command_policy: CommandSandboxPolicy = field(default_factory=CommandSandboxPolicy.default)
    memory: str = "2g"
    cpus: str = "2"
    pids_limit: int = 256

    def __post_init__(self) -> None:
        if self.engine not in {SandboxBackend.DOCKER, SandboxBackend.PODMAN}:
            raise ValueError("ContainerSandbox requires docker or podman backend")

    async def run_command(self, command: list[str], *, timeout_s: float) -> SandboxProcessResult:
        self.command_policy.validate(command)
        container_command = self.build_container_command(command)
        return await _run_process(
            container_command,
            cwd=self.root,
            timeout_s=timeout_s,
            timeout_label=f"{self.engine.value} run",
        )

    async def apply_patch(self, patch: str, *, timeout_s: float = 30.0) -> SandboxProcessResult:
        validate_patch(patch)
        command = ["git", "apply", "--whitespace=nowarn", "--"]
        container_command = self.build_container_command(command)
        return await _run_process(
            container_command,
            cwd=self.root,
            timeout_s=timeout_s,
            stdin=patch.encode("utf-8"),
            timeout_label=f"{self.engine.value} git apply",
        )

    def build_container_command(self, command: list[str]) -> list[str]:
        root = str(self.root.resolve())
        return [
            self.engine.value,
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--user",
            "1000:1000",
            "--workdir",
            "/workspace",
            "--volume",
            f"{root}:/workspace:rw",
            self.image,
            *command,
        ]


def build_sandbox(
    root: Path,
    *,
    backend: SandboxBackend | str = SandboxBackend.LOCAL,
    image: str = "python:3.13-slim",
) -> LocalSandbox | ContainerSandbox:
    resolved = SandboxBackend.from_value(str(backend))
    if resolved is SandboxBackend.LOCAL:
        return LocalSandbox(root)
    return ContainerSandbox(root=root, engine=resolved, image=image)


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


async def _run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    stdin: bytes | None = None,
    timeout_label: str,
) -> SandboxProcessResult:
    timeout_s = max(1.0, min(float(timeout_s), 120.0))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout=timeout_s)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise TimeoutError(f"{timeout_label} timed out after {timeout_s:g}s") from exc
    return SandboxProcessResult(
        command=tuple(command),
        returncode=process.returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )
