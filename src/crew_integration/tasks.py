from crewai import Task

def build_tasks(market_analyst, rl_decision_agent, risk_manager, state, action):
    market_data = {
        "price": state[0],
        "volume": state[1],
        "volatility": state[2]
    }

    analyze_task = Task(
        description=f"Analyze the following market data: {market_data}. Identify trends or risks.",
        agent=market_analyst,
        expected_output="A summary of the market with potential trading signals."
    )

    rl_task = Task(
        description=f"The RL agent proposes the action: {action}. Justify whether it's appropriate.",
        agent=rl_decision_agent,
        expected_output="Final recommended action based on RL input."
    )

    validate_task = Task(
        description="Evaluate the recommended action. Should it be executed or rejected? Justify.",
        agent=risk_manager,
        expected_output="Approval or rejection with explanation."
    )

    return analyze_task, rl_task, validate_task
