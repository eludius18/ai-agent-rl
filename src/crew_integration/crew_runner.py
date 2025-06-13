# src/crew_integration/crew_runner.py

import os
import importlib

from crewai import Crew
from crew_integration.agents import build_agents
from crew_integration.tasks import build_tasks
from dotenv import load_dotenv


load_dotenv()

def run_crew_flow(state, action):
    # Dynamically load the LLM provider from environment
    provider = os.getenv("CREWAI_LLM_PROVIDER")
    class_name = os.getenv("CREWAI_LLM_CLASS")
    model = os.getenv("CREWAI_LLM_MODEL")

    module = importlib.import_module(provider)
    llm_class = getattr(module, class_name)
    llm = llm_class(model=model)

    market_analyst, rl_decision_agent, risk_manager, news_checker = build_agents(llm)

    analyze_market_task, decision_task, risk_task, news_task = build_tasks(
        market_analyst, rl_decision_agent, risk_manager, news_checker, state, action
    )

    crew = Crew(
        agents=[market_analyst, rl_decision_agent, risk_manager, news_checker],
        tasks=[analyze_market_task, decision_task, risk_task, news_task],
        verbose=True
    )

    return crew.kickoff()
