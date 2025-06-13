from crewai import Task

def build_tasks(market_analyst, rl_decision_agent, risk_manager, news_checker, state, action):
    market_data = {
        "price": state[0],
        "volume": state[1],
        "volatility": state[2]
    }

    analyze_task = Task(
        description=f"Analyze market data: {market_data}. Identify trends or risks.",
        agent=market_analyst,
        expected_output="Market summary with trading signals."
    )

    news_task = Task(
        description="Search recent crypto news to identify any breaking events or market-moving alerts.",
        agent=news_checker,
        expected_output="Summary of relevant crypto news."
    )

    rl_task = Task(
        description=f"The RL agent suggests: {action}. Is it valid?",
        agent=rl_decision_agent,
        expected_output="Recommendation on action's validity."
    )

    validate_task = Task(
        description="Evaluate risk of the action and approve/reject.",
        agent=risk_manager,
        expected_output="Approval or rejection with explanation."
    )

    return analyze_task, news_task, rl_task, validate_task
