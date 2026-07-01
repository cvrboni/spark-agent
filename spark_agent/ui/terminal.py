from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True, slots=True)
class TerminalUI:
    """Small terminal rendering layer with an optional Rich backend."""

    stream: TextIO
    enabled: bool = True
    rich: bool = True

    @classmethod
    def stderr(cls, *, enabled: bool = True) -> TerminalUI:
        return cls(stream=sys.stderr, enabled=enabled)

    @property
    def _console(self) -> Any | None:
        if not self.enabled or not self.rich or not self.stream.isatty():
            return None
        try:
            from rich.console import Console
        except ImportError:
            return None
        return Console(file=self.stream, stderr=self.stream is sys.stderr)

    def info(self, message: str) -> None:
        self._print("info", message, style="cyan")

    def warning(self, message: str) -> None:
        self._print("warn", message, style="yellow")

    def error(self, message: str) -> None:
        self._print("error", message, style="red")

    def success(self, message: str) -> None:
        self._print("ok", message, style="green")

    def metric(self, label: str, value: str) -> None:
        console = self._console
        if console is not None:
            console.print(f"[dim]{label}[/dim] [bold]{value}[/bold]")
            return
        self.info(f"{label} {value}")

    def step(self, current: int, total: int, *, model: str, timeout_s: float, max_tokens: int) -> None:
        self.info(
            f"step {current}/{total}: waiting for {model} "
            f"(timeout={timeout_s:g}s, max_tokens={max_tokens})"
        )

    def turn(self, current: int, total: int, *, model: str, timeout_s: float, max_tokens: int) -> None:
        self.info(
            f"turn {current}/{total}: waiting for {model} "
            f"(timeout={timeout_s:g}s, max_tokens={max_tokens})"
        )

    def tools_ran(self, names: Iterable[str]) -> None:
        names_text = ", ".join(names)
        console = self._console
        if console is not None:
            console.print(f"[magenta]tools[/magenta] {names_text}")
            return
        self.info(f"ran tools: {names_text}")

    def session_header(
        self,
        *,
        session_id: str,
        action: str,
        model: str,
        sandbox_backend: str,
        repo: str,
    ) -> None:
        if not self.enabled:
            return
        console = self._console
        if console is not None:
            from rich.table import Table

            table = Table.grid(padding=(0, 2))
            table.add_column(style="dim")
            table.add_column()
            table.add_row("session", session_id)
            table.add_row("state", action)
            table.add_row("model", model)
            table.add_row("sandbox", sandbox_backend)
            table.add_row("repo", repo)
            console.rule("[bold]SparkAgent[/bold]")
            console.print(table)
            console.print("[dim]Type /help for commands, /exit to leave.[/dim]")
            return
        print(f"SparkAgent {action} session {session_id}", file=self.stream, flush=True)
        print("Type /help for commands, /exit to leave.", file=self.stream, flush=True)

    def approval_request(self, name: str, summary: str) -> None:
        if not self.enabled:
            return
        console = self._console
        if console is not None:
            from rich.panel import Panel
            from rich.syntax import Syntax

            body: Any = summary
            if name == "apply_patch":
                body = Syntax(summary, "diff", theme="ansi_dark", word_wrap=False)
            console.print(Panel(body, title=f"approval required: {name}", border_style="yellow"))
            return
        print(f"\n[spark-agent] approval required:\n{summary}\n", file=self.stream, flush=True)

    def _print(self, label: str, message: str, *, style: str) -> None:
        if not self.enabled:
            return
        console = self._console
        if console is not None:
            console.print(f"[{style}]{label}[/{style}] {message}")
            return
        print(f"[spark-agent] {message}", file=self.stream, flush=True)
