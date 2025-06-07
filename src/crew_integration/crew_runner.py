from crewai import Crew
from crew_integration.agents import build_agents
from crew_integration.tasks import build_tasks
from langchain_ollama import OllamaLLM

def run_crew_flow(state, action):
    # Use local model with correct format to avoid LiteLLM errors
    llm = OllamaLLM(model="ollama/mistral")

    # Create agents
    market_analyst, rl_decision_agent, risk_manager = build_agents(llm)

    # Pass all required agents and data to build_tasks
    analyze_market_task, decision_task, risk_task = build_tasks(
        market_analyst, rl_decision_agent, risk_manager, state, action
    )

    # Assemble the crew
    crew = Crew(
        agents=[market_analyst, rl_decision_agent, risk_manager],
        tasks=[analyze_market_task, decision_task, risk_task],
        verbose=True
    )

    result = crew.kickoff()
    return result