from __future__ import annotations

import asyncio
import os

from spark_agent.core.executor import AgentExecutor, VLLMClientConfig
from spark_agent.core.prompt_engine import PromptBlock, PromptEngine
from spark_agent.tools.codebase import codebase_tool_specs, tool_definitions


async def main() -> None:
    prompt = PromptEngine(
        static_blocks=[
            PromptBlock("system", "You are SparkAgent. Use repository tools before answering."),
            PromptBlock("codebase_context", "Retrieve repository context selectively."),
            PromptBlock.from_jsonable("tool_definitions", tool_definitions()),
        ],
    )
    config = VLLMClientConfig(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
        model=os.getenv("VLLM_MODEL", "deepseek-ai/DeepSeek-V4"),
        api_key=os.getenv("SPARK_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    async with AgentExecutor(
        prompt_engine=prompt,
        config=config,
        tools=codebase_tool_specs(),
    ) as agent:
        answer = await agent.run("Find the PromptEngine class and summarize its cache policy.")
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
