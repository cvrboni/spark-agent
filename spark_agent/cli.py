from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from spark_agent.config import SparkAgentConfig, default_config_path, write_default_config
from spark_agent.core.executor import AgentExecutor, VLLMClientConfig, VLLMRequestError
from spark_agent.core.prompt_engine import PromptBlock, PromptEngine
from spark_agent.core.types import ToolSpec
from spark_agent.session import SessionStore
from spark_agent.tools.codebase import codebase_tool_specs, tool_definitions, view_file_outline
from spark_agent.tools.workspace import workspace_tool_definitions, workspace_tool_specs


def build_prompt(config: SparkAgentConfig, *, repo_root: Path | None = None) -> PromptEngine:
    language_policy = {
        "it": "Rispondi in italiano, salvo nomi di codice o output tecnico.",
        "en": "Answer in English unless code or tool output requires otherwise.",
        "auto": (
            "Match the user's language. Prefer Italian for Italian prompts and English for English "
            "prompts. Keep code identifiers unchanged."
        ),
    }.get(config.language, "Match the user's language.")
    root = repo_root or Path.cwd()
    return PromptEngine(
        static_blocks=[
            PromptBlock(
                name="system",
                content=(
                    "You are SparkAgent, a cache-stable coding agent optimized for local "
                    "OpenAI-compatible inference servers. Use tools for repository inspection, "
                    "targeted edits, and validation. "
                    "Never load large files into context when a grep or outline is sufficient. "
                    "When changing code, inspect first, patch narrowly, then run the most relevant "
                    "allowlisted validation command. Keep working until the task is complete or a "
                    "real blocker is reached.\n"
                    f"{language_policy}"
                ),
            ),
            PromptBlock(
                name="codebase_context",
                content=(
                    f"Repository root: {root.name or '.'}\n"
                    f"Context budget target: {config.repo_context_budget} bytes.\n"
                    "Retrieve code context append-only through repo_grep, view_file_outline, "
                    "list_files, and read_file. Use apply_patch for edits."
                ),
            ),
            PromptBlock.from_jsonable("tool_definitions", all_tool_definitions()),
        ],
    )


def all_tool_definitions() -> list[dict[str, object]]:
    return [*tool_definitions(), *workspace_tool_definitions()]


def all_tool_specs(repo_root: Path | None = None) -> list[ToolSpec]:
    return [*codebase_tool_specs(repo_root), *workspace_tool_specs(repo_root)]


def build_final_prompt(config: SparkAgentConfig, source_prompt: PromptEngine) -> PromptEngine:
    language_policy = {
        "it": "Rispondi in italiano.",
        "en": "Answer in English.",
        "auto": "Match the user's language.",
    }.get(config.language, "Match the user's language.")
    final_prompt = PromptEngine(
        static_blocks=[
            PromptBlock(
                name="system",
                content=(
                    "You are SparkAgent in final-answer mode. Tools are not available. "
                    "Do not emit DSML, XML, JSON tool calls, or function-call syntax. "
                    "Use only the observations already present and answer in plain text.\n"
                    f"{language_policy}"
                ),
            )
        ]
    )
    for event in source_prompt.dynamic_events:
        role = event.get("role")
        if role == "tool":
            final_prompt.append_user_message(
                f"Tool observation from {event.get('name', 'tool')}:\n{event.get('content', '')}"
            )
        elif role == "assistant" and event.get("content"):
            final_prompt.append_assistant_message(content=str(event["content"]))
        elif role == "user":
            final_prompt.append_event(event)
    final_prompt.append_user_message("Provide the final answer now in plain text.")
    return final_prompt


async def build_local_repo_snapshot(root: Path, *, max_files: int = 6, max_chars: int = 8_000) -> str:
    candidates = _collect_python_files(root)
    candidates.sort(key=lambda path: (_repo_file_priority(path), str(path)))
    selected = candidates[:max_files]
    outlines = []
    for path in selected:
        try:
            outline = await view_file_outline(path, max_bytes=24_000)
        except (OSError, SyntaxError, UnicodeError) as exc:
            outlines.append({"path": str(path.relative_to(root)), "error": str(exc)})
            continue
        outline["path"] = str(path.relative_to(root))
        outlines.append(outline)
    top_level = _top_level_entries(root)
    rendered = json.dumps(
        {
            "root": ".",
            "top_level": top_level,
            "python_file_count": len(candidates),
            "outlined_files": outlines,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}\n...[repository snapshot truncated to {max_chars} chars]"


def _collect_python_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if path.is_file()
        and not any(part in {".git", ".venv", "__pycache__", "node_modules"} for part in path.parts)
    ]


def _top_level_entries(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.iterdir()
        if path.name not in {".git", ".venv", "__pycache__"} and not path.name.startswith(".pytest")
    )


def _repo_file_priority(path: Path) -> tuple[int, int]:
    text = str(path)
    if "spark_agent/core" in text:
        return (0, len(path.parts))
    if "spark_agent/cli.py" in text or "spark_agent/config.py" in text:
        return (1, len(path.parts))
    if "spark_agent/tools" in text:
        return (2, len(path.parts))
    if "tests" in path.parts:
        return (3, len(path.parts))
    return (4, len(path.parts))


async def run_local_retrieval_agent(
    args: argparse.Namespace,
    *,
    config: SparkAgentConfig,
    task: str,
    timeout_s: float,
    max_tokens: int,
) -> int:
    if not args.quiet:
        print("[spark-agent] collecting local repository snapshot...", file=sys.stderr, flush=True)
    snapshot = await build_local_repo_snapshot(
        Path.cwd(),
        max_files=args.local_snapshot_files,
        max_chars=args.local_snapshot_chars,
    )
    prompt = PromptEngine(
        static_blocks=[
            PromptBlock(
                name="system",
                content=(
                    "You are SparkAgent, a fast local-first coding agent. Tools are not available "
                    "in this final answer call. Use the provided compact repository snapshot. "
                    "Do not emit tool calls or DSML. Answer concisely with practical next steps. "
                    "The executable is spark-agent, not spark_agent. Known CLI commands are: "
                    "spark-agent init, spark-agent doctor, spark-agent run, spark-agent chat, "
                    "spark-agent continue, spark-agent prompt-preview."
                ),
            )
        ]
    )
    prompt.append_user_message(task)
    prompt.append_user_message(f"Repository snapshot:\n{snapshot}")
    client_config = VLLMClientConfig(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )
    if not args.quiet:
        print(
            f"[spark-agent] waiting for {config.model} "
            f"(timeout={timeout_s:g}s, max_tokens={max_tokens})...",
            file=sys.stderr,
            flush=True,
        )
    try:
        async with AgentExecutor(prompt_engine=prompt, config=client_config, tools=()) as agent:
            turn = await agent.step()
    except VLLMRequestError as exc:
        print(f"Provider request failed: {exc}", file=sys.stderr)
        return 1
    print(turn.content or "")
    return 0


async def run_agent(args: argparse.Namespace) -> int:
    config = SparkAgentConfig.from_file(Path(args.config) if args.config else None)
    prompt = build_prompt(config)
    task = args.task or sys.stdin.read().strip()
    if not task:
        print("No task provided. Pass a prompt or pipe one through stdin.", file=sys.stderr)
        return 2
    timeout_s = args.timeout if args.timeout is not None else min(config.timeout_s, 30.0)
    max_tokens = args.max_tokens if args.max_tokens is not None else min(config.max_tokens, 768)
    if args.retrieval_mode == "local":
        return await run_local_retrieval_agent(
            args,
            config=config,
            task=task,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )

    tool_call_max_tokens = (
        args.tool_call_max_tokens
        if args.tool_call_max_tokens is not None
        else min(max_tokens, 256)
    )
    client_config = VLLMClientConfig(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_s=timeout_s,
        max_tokens=tool_call_max_tokens,
        max_tool_result_chars=args.max_tool_result_chars,
    )
    final_client_config = VLLMClientConfig(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        max_tool_result_chars=args.max_tool_result_chars,
    )
    try:
        answer: str | None = None
        async with AgentExecutor(
            prompt_engine=prompt,
            config=client_config,
            tools=all_tool_specs(Path.cwd()) if args.retrieval_mode == "model" else (),
        ) as agent:
            prompt.append_user_message(task)
            tool_rounds = 0
            for turn_number in range(1, args.max_turns + 1):
                if not args.quiet:
                    print(
                        f"[spark-agent] turn {turn_number}/{args.max_turns}: "
                        f"waiting for {config.model} "
                        f"(timeout={timeout_s:g}s, max_tokens={tool_call_max_tokens})...",
                        file=sys.stderr,
                        flush=True,
                    )
                turn = await agent.step()
                if turn.tool_calls:
                    tool_rounds += 1
                    names = [
                        str(tool_call.get("function", {}).get("name", "unknown"))
                        for tool_call in turn.tool_calls
                    ]
                    if not args.quiet:
                        print(
                            f"[spark-agent] ran tools: {', '.join(names)}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if tool_rounds >= args.max_tool_rounds:
                        if not args.quiet:
                            print(
                                "[spark-agent] tool round budget reached; finalizing.",
                                file=sys.stderr,
                                flush=True,
                            )
                        break
                    continue
                answer = turn.content or ""
                break
            if answer is None:
                if not args.finalize:
                    raise RuntimeError(f"Agent did not finish within {args.max_turns} turns")
                prompt.append_user_message(
                    "Tools are now disabled. Do not call tools. Do not emit DSML, XML, JSON tool "
                    "calls, or function-call syntax. Answer now in plain text using only the "
                    "observations already available."
                )
        if answer is None:
            if not args.quiet:
                print(
                    "[spark-agent] finalizing without tools...",
                    file=sys.stderr,
                    flush=True,
                )
            final_prompt = build_final_prompt(config, prompt)
            async with AgentExecutor(
                prompt_engine=final_prompt,
                config=final_client_config,
                tools=(),
            ) as final_agent:
                final_turn = await final_agent.step()
                answer = final_turn.content or ""
    except VLLMRequestError as exc:
        print(f"Provider request failed: {exc}", file=sys.stderr)
        print("Try a smaller --max-tokens, a shorter task, or check `spark-agent doctor`.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent stopped: {exc}", file=sys.stderr)
        print("Try increasing --max-turns or ask for a narrower repository task.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    print(answer)
    return 0


async def run_session_agent(args: argparse.Namespace) -> int:
    config = SparkAgentConfig.from_file(Path(args.config) if args.config else None)
    store = SessionStore(Path.cwd())
    try:
        if args.session:
            session = store.load(args.session)
            resumed = True
        elif args.resume:
            session = store.load_latest()
            resumed = True
        else:
            session = store.create(title=args.task)
            resumed = False
    except (FileNotFoundError, ValueError) as exc:
        print(f"Session error: {exc}", file=sys.stderr)
        return 2

    prompt = build_prompt(config)
    for event in session.events:
        prompt.append_event(event)

    task = args.task or sys.stdin.read().strip()
    if not task and not resumed:
        print("No task provided. Pass a prompt or pipe one through stdin.", file=sys.stderr)
        return 2
    if task:
        prompt.append_user_message(task)
        session = store.append_events(session, [prompt.dynamic_events[-1]])
    elif resumed:
        prompt.append_user_message("Continue the previous task from the current repository state.")
        session = store.append_events(session, [prompt.dynamic_events[-1]])

    if not args.quiet:
        action = "resuming" if resumed else "created"
        print(f"[spark-agent] {action} session {session.session_id}", file=sys.stderr, flush=True)

    timeout_s = args.timeout if args.timeout is not None else min(config.timeout_s, 60.0)
    max_tokens = args.max_tokens if args.max_tokens is not None else min(config.max_tokens, 1024)
    client_config = VLLMClientConfig(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        max_tool_result_chars=args.max_tool_result_chars,
    )
    saved_events = len(prompt.dynamic_events)
    tool_rounds = 0
    answer: str | None = None
    force_finalize = False

    try:
        async with AgentExecutor(
            prompt_engine=prompt,
            config=client_config,
            tools=all_tool_specs(Path.cwd()),
        ) as agent:
            for turn_number in range(1, args.max_steps + 1):
                if not args.quiet:
                    print(
                        f"[spark-agent] step {turn_number}/{args.max_steps}: waiting for "
                        f"{config.model} (timeout={timeout_s:g}s, max_tokens={max_tokens})...",
                        file=sys.stderr,
                        flush=True,
                    )
                turn = await agent.step()
                new_events = list(prompt.dynamic_events[saved_events:])
                if new_events:
                    session = store.append_events(session, new_events)
                    saved_events += len(new_events)
                if turn.tool_calls:
                    tool_rounds += 1
                    if not args.quiet:
                        names = [
                            str(tool_call.get("function", {}).get("name", "unknown"))
                            for tool_call in turn.tool_calls
                        ]
                        print(
                            f"[spark-agent] ran tools: {', '.join(names)}",
                            file=sys.stderr,
                            flush=True,
                        )
                    if tool_rounds >= args.max_tool_rounds:
                        prompt.append_user_message(
                            "Tool budget reached. Produce a concise final answer with current "
                            "status, files changed, validation run, and remaining blockers."
                        )
                        session = store.append_events(session, [prompt.dynamic_events[-1]])
                        saved_events += 1
                        force_finalize = True
                        break
                    continue
                answer = turn.content or ""
                break
        if answer is None and force_finalize:
            async with AgentExecutor(
                prompt_engine=prompt,
                config=client_config,
                tools=(),
            ) as final_agent:
                turn = await final_agent.step()
                new_events = list(prompt.dynamic_events[saved_events:])
                if new_events:
                    session = store.append_events(session, new_events)
                answer = turn.content or ""
    except VLLMRequestError as exc:
        print(f"Provider request failed: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent stopped: {exc}", file=sys.stderr)
        return 1

    if answer is None:
        print(f"Agent stopped after {args.max_steps} steps. Session: {session.session_id}", file=sys.stderr)
        return 1
    print(answer)
    if not args.quiet:
        print(f"[spark-agent] session: {session.session_id}", file=sys.stderr, flush=True)
    return 0


async def doctor(args: argparse.Namespace) -> int:
    config = SparkAgentConfig.from_file(Path(args.config) if args.config else None)
    client_config = VLLMClientConfig(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_s=min(config.timeout_s, 30.0),
        max_tokens=128,
    )
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "temperature": 0,
        "max_tokens": client_config.max_tokens,
    }
    print(f"Config: {Path(args.config) if args.config else default_config_path()}")
    print(f"Provider: {client_config.chat_completions_url}")
    print(f"Model: {config.model}")
    print(f"Language: {config.language}")
    print(
        f"API key env: {_redact_if_secret_like(config.api_key_env)} "
        f"({'set' if config.api_key else 'not set'})"
    )
    if not config.api_key_env_is_valid_name or config.api_key_env_looks_like_secret:
        print(
            "Config error: provider.api_key_env must be an environment variable name, "
            "for example SPARK_AGENT_API_KEY. Put the token in that environment variable, "
            "not directly in config.toml."
        )
        return 2
    try:
        async with httpx.AsyncClient(timeout=client_config.timeout_s) as client:
            response = await client.post(
                client_config.chat_completions_url,
                json=payload,
                headers=client_config.headers,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        print(f"Provider check: failed ({exc})")
        return 1
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    print(f"Provider check: ok ({content!r})")
    return 0


def _redact_if_secret_like(value: str) -> str:
    if len(value) > 12 and (
        value.startswith(("sk-", "sk_", "Bearer "))
        or all(char.lower() in "0123456789abcdef" for char in value)
    ):
        return f"{value[:4]}...{value[-4:]}"
    if len(value) <= 12 or value.isidentifier():
        return value
    return f"{value[:4]}...{value[-4:]}"


def init_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else default_config_path()
    existing = SparkAgentConfig.from_file(config_path) if config_path.exists() else SparkAgentConfig()
    requested_config = SparkAgentConfig(
        base_url=args.base_url or existing.base_url,
        model=args.model or existing.model,
        language=args.language or existing.language,
        api_key_env=existing.api_key_env,
        timeout_s=existing.timeout_s,
        max_tokens=existing.max_tokens,
        repo_context_budget=existing.repo_context_budget,
        prefer_local_tools=existing.prefer_local_tools,
    )
    has_explicit_updates = args.base_url is not None or args.model is not None or args.language is not None
    should_write = args.force or has_explicit_updates or not config_path.exists()
    if not should_write:
        print(f"Config already exists: {config_path}")
        print("Use --force or pass --base-url/--model/--language to update it.")
        return 0

    path = write_default_config(
        config_path,
        base_url=requested_config.base_url,
        model=requested_config.model,
        language=requested_config.language,
        force=True,
    )
    print(f"Wrote config: {path}")
    return 0


def prompt_preview(args: argparse.Namespace) -> int:
    config = SparkAgentConfig.from_file(Path(args.config) if args.config else None)
    prompt = build_prompt(config)
    print(prompt.render_messages()[0]["content"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spark-agent",
        description="Cache-stable coding agent for local OpenAI-compatible LLM servers.",
    )
    parser.add_argument("--config", help="Path to config.toml. Defaults to XDG config path.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    init = subparsers.add_parser("init", help="Create a local config file.")
    init.add_argument("--base-url")
    init.add_argument("--model")
    init.add_argument("--language", choices=["auto", "it", "en"])
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=init_config)

    doctor_parser = subparsers.add_parser("doctor", help="Check config and provider connectivity.")
    doctor_parser.set_defaults(async_func=doctor)

    run_parser = subparsers.add_parser("run", help="Run an agent task.")
    run_parser.add_argument("task", nargs="?", help="Task prompt. Reads stdin if omitted.")
    run_parser.add_argument(
        "--retrieval-mode",
        choices=["local", "model", "none"],
        default="local",
        help="Context retrieval strategy. local is fastest and is the default.",
    )
    run_parser.add_argument("--max-turns", type=int, default=8)
    run_parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=1,
        help="Maximum tool-calling rounds before forcing a final answer. Defaults to 1.",
    )
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-turn provider timeout in seconds. Defaults to min(config timeout, 30).",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-turn max output tokens. Defaults to min(config max_tokens, 768).",
    )
    run_parser.add_argument(
        "--tool-call-max-tokens",
        type=int,
        default=None,
        help="Max output tokens while tools are enabled. Defaults to min(max_tokens, 256).",
    )
    run_parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=6000,
        help="Maximum characters from each tool result appended back into the model context.",
    )
    run_parser.add_argument(
        "--finalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After max tool turns, make one final no-tool call to force an answer.",
    )
    run_parser.add_argument("--quiet", action="store_true", help="Hide progress messages.")
    run_parser.add_argument(
        "--local-snapshot-files",
        type=int,
        default=6,
        help="Number of local Python files to outline in local retrieval mode.",
    )
    run_parser.add_argument(
        "--local-snapshot-chars",
        type=int,
        default=8000,
        help="Maximum characters of local repository snapshot sent to the model.",
    )
    run_parser.set_defaults(async_func=run_agent)

    chat_parser = subparsers.add_parser("chat", help="Start a persistent local coding session.")
    chat_parser.add_argument("task", nargs="?", help="Initial task prompt. Reads stdin if omitted.")
    chat_parser.add_argument("--session", help="Resume a specific session id instead of creating one.")
    chat_parser.add_argument("--resume", action="store_true", help="Resume the latest session.")
    chat_parser.add_argument("--max-steps", type=int, default=16)
    chat_parser.add_argument("--max-tool-rounds", type=int, default=12)
    chat_parser.add_argument("--timeout", type=float, default=None)
    chat_parser.add_argument("--max-tokens", type=int, default=None)
    chat_parser.add_argument("--max-tool-result-chars", type=int, default=6000)
    chat_parser.add_argument("--quiet", action="store_true")
    chat_parser.set_defaults(async_func=run_session_agent)

    continue_parser = subparsers.add_parser("continue", help="Continue the latest persistent session.")
    continue_parser.add_argument("task", nargs="?", help="Optional follow-up prompt.")
    continue_parser.add_argument("--session", help="Session id to resume.")
    continue_parser.add_argument("--max-steps", type=int, default=16)
    continue_parser.add_argument("--max-tool-rounds", type=int, default=12)
    continue_parser.add_argument("--timeout", type=float, default=None)
    continue_parser.add_argument("--max-tokens", type=int, default=None)
    continue_parser.add_argument("--max-tool-result-chars", type=int, default=6000)
    continue_parser.add_argument("--quiet", action="store_true")
    continue_parser.set_defaults(async_func=run_session_agent, resume=True)

    preview = subparsers.add_parser("prompt-preview", help="Print the static prompt prefix.")
    preview.set_defaults(func=prompt_preview)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        if hasattr(args, "async_func"):
            return asyncio.run(args.async_func(args))
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
