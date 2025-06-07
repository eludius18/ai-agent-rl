from crewai import Agent
from langchain_ollama import OllamaLLM

def build_agents(llm: OllamaLLM):
    market_analyst = Agent(
        role="Market Analyst",
        goal="Analyze crypto market trends and predict short-term price direction.",
        backstory="You are a seasoned financial analyst specialized in cryptocurrencies.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    rl_decision_agent = Agent(
        role="RL Decision Maker",
        goal="Decide whether to execute a trade based on expected reward and policy confidence.",
        backstory="You are an AI agent trained in reinforcement learning to detect optimal trading points.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    risk_manager = Agent(
        role="Risk Manager",
        goal="Evaluate risk level of the proposed trade and approve only if within acceptable limits.",
        backstory="You are an expert in portfolio and risk management for algorithmic trading systems.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    return market_analyst, rl_decision_agent, risk_manager