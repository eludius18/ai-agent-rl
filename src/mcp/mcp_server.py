import asyncio
import time
import os
from dotenv import load_dotenv

from mcp_agent.app import MCPApp
from mcp_agent.config import Settings, OpenAISettings, LoggerSettings
from mcp_agent.agents.agent import Agent
from mcp_agent.tools.crewai_tool import from_crewai_tool
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from crewai_tools import SerperDevTool, FileWriterTool

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY") or "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"

settings = Settings(
    logger=LoggerSettings(level="info"),
    openai=OpenAISettings(
        api_key=openai_api_key,
        default_model="gpt-3.5-turbo",
    ),
)

app = MCPApp(name="ai_agent_rl_mcp_server", settings=settings)

async def main():
    async with app.run() as agent_app:
        logger = agent_app.logger

        search_tool = SerperDevTool()
        file_tool = FileWriterTool()
        agent = Agent(
            name="search_mcp_agent",
            instruction="You are a helpful AI trading assistant.",
            server_names=[],
            functions=[
                from_crewai_tool(search_tool),
                from_crewai_tool(file_tool),
            ],
        )

        async with agent:
            llm = await agent.attach_llm(OpenAIAugmentedLLM)

            result = await llm.generate_str(
                message="What is Singapore's favorite dish? Write a haiku about it in ./haiku.md"
            )
            logger.info(f"✅ LLM Result: {result}")

if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    print(f"Total run time: {time.time() - start:.2f}s")
