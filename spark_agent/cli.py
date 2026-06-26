from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from spark_agent.config import SparkAgentConfig, default_config_path, write_default_config
from spark_agent.core.executor import AgentExecutor, VLLMClientConfig, VLLMRequestError
from spark_agent.core.prompt_engine import PromptBlock, PromptEngine
from spark_agent.tools.codebase import codebase_tool_specs, tool_definitions


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
                    "OpenAI-compatible inference servers. Use tools for repository inspection. "
                    "Never load large files into context when a grep or outline is sufficient.\n"
                    f"{language_policy}"
                ),
            ),
            PromptBlock(
                name="codebase_context",
                content=(
                    f"Repository root: {root.resolve()}\n"
                    f"Context budget target: {config.repo_context_budget} bytes.\n"
                    "Retrieve code context append-only through repo_grep and view_file_outline."
                ),
            ),
            PromptBlock.from_jsonable("tool_definitions", tool_definitions()),
        ],
    )


async def run_agent(args: argparse.Namespace) -> int:
    config = SparkAgentConfig.from_file(Path(args.config) if args.config else None)
    prompt = build_prompt(config)
    task = args.task or sys.stdin.read().strip()
    if not task:
        print("No task provided. Pass a prompt or pipe one through stdin.", file=sys.stderr)
        return 2
    client_config = VLLMClientConfig(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_s=config.timeout_s,
        max_tokens=config.max_tokens,
    )
    try:
        async with AgentExecutor(
            prompt_engine=prompt,
            config=client_config,
            tools=codebase_tool_specs(),
        ) as agent:
            answer = await agent.run(task, max_turns=args.max_turns)
    except VLLMRequestError as exc:
        print(f"Provider request failed: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Agent stopped: {exc}", file=sys.stderr)
        print("Try increasing --max-turns or ask for a narrower repository task.", file=sys.stderr)
        return 1
    print(answer)
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
    print(f"API key env: {config.api_key_env} ({'set' if config.api_key else 'not set'})")
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
    run_parser.add_argument("--max-turns", type=int, default=8)
    run_parser.set_defaults(async_func=run_agent)

    preview = subparsers.add_parser("prompt-preview", help="Print the static prompt prefix.")
    preview.set_defaults(func=prompt_preview)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if hasattr(args, "async_func"):
        return asyncio.run(args.async_func(args))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
