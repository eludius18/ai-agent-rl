import numpy as np
from agent.ddpg import Agent

def get_market_state():
    return {
        "price": 105.2,
        "volume": 30000,
        "volatility": 0.03
    }

def get_rl_action(state: dict) -> str:
    agent = Agent(state_dim=3, action_dim=1, max_action=1)
    input_array = np.array([state["price"], state["volume"], state["volatility"]])
    action = agent.select_action(input_array)
    return f"RL suggests action: {action}"
