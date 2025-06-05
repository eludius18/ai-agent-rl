from crewai import Agent
from crew_integration.llm import get_llm

llm = get_llm()

market_analyst = Agent(
    role="Market Analyst",
    goal="Analyze market data and identify trading signals.",
    backstory="Expert in trend detection.",
    verbose=True,
    llm=llm
)

rl_decision_agent = Agent(
    role="RL Decision Agent",
    goal="Use the RL output and analysis to decide an action.",
    backstory="Reinforcement Learning driven decision-maker.",
    verbose=True,
    llm=llm
)

risk_manager = Agent(
    role="Risk Manager",
    goal="Validate the proposed action before execution.",
    backstory="Risk-averse supervisor.",
    verbose=True,
    llm=llm
)
