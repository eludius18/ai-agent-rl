from crewai import Task
from crew_integration.agents import market_analyst, rl_decision_agent, risk_manager
from crew_integration.utils import get_market_state, get_rl_action

market_data = get_market_state()
rl_output = get_rl_action(market_data)

analyze_task = Task(
    description=f"Analyze market data: {market_data}. Identify trends and possible actions.",
    agent=market_analyst,
    expected_output="Market analysis summary"
)

rl_task = Task(
    description=f"Based on RL output: '{rl_output}', propose a final action for trading.",
    agent=rl_decision_agent,
    expected_output="Final proposed trading action"
)

validate_task = Task(
    description="Review the proposed action. Is it safe to execute? Justify your decision.",
    agent=risk_manager,
    expected_output="Execution approval or rejection"
)
