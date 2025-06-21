import os
from crewai import Agent
from langchain_core.language_models import BaseLanguageModel
from typing import Tuple

def get_web_search_tool():
    if os.getenv("OPENAI_API_KEY"):
        print("🔑 Using WebsiteSearchTool with OpenAI API Key for RAG")
        from crewai_tools.tools.website_search.website_search_tool import WebsiteSearchTool
        return [WebsiteSearchTool()]
    else:
        print("🦆 Using DuckDuckGoSearchRun for RAG (no OpenAI API Key required)")
        from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
        from langchain.tools import Tool
        search = DuckDuckGoSearchRun()
        return [
            Tool(
                name="Web Search",
                func=search.run,
                description="Useful to search recent news and current events from the web"
            )
        ]

def build_agents(llm: BaseLanguageModel) -> Tuple[Agent, Agent, Agent, Agent]:
    tools = get_web_search_tool()

    market_analyst = Agent(
        role="Market Analyst",
        goal="Analyze market trends using current news and data",
        backstory="An expert in financial markets who combines on-chain, off-chain, and live news to assess trends.",
        tools=tools,
        llm=llm
    )

    news_checker = Agent(
        role="News Checker",
        goal="Find relevant news or events that might affect cryptocurrency prices",
        backstory="A fast, precise agent trained to search the web for impactful financial or crypto events.",
        tools=tools,
        llm=llm
    )

    rl_decision_agent = Agent(
        role="RL Trading Agent",
        goal="Take trading actions based on the reinforcement learning policy",
        backstory="A decision-making agent that uses signals from the model and other analysts.",
        tools=tools,
        llm=llm
    )

    risk_manager = Agent(
        role="Risk Manager",
        goal="Evaluate risk and capital exposure before making any trade",
        backstory="Protects the system from making high-risk trades that violate constraints.",
        tools=tools,
        llm=llm
    )

    return market_analyst, news_checker, rl_decision_agent, risk_manager