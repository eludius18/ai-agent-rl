import os
import time
import requests
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import CryptoTradingEnv

# Load environment variables
load_dotenv()
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE"))
REWARD_THRESHOLD_PERCENT = float(os.getenv("REWARD_THRESHOLD_PERCENT"))

def evaluate_model():
    """
    Evaluates the trained model by running it in the environment.
    Returns a float estimated reward.
    """
    print("🔍 Evaluating model...")

    env = DummyVecEnv([lambda: CryptoTradingEnv()])
    model = PPO.load("trading_agent")
    
    obs = env.reset()
    total_reward = 0
    for _ in range(10):  # Run for 10 steps
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    total_reward = float(total_reward.item())  # Convert NumPy array to Python float

    print(f"📊 Estimated Reward: {total_reward}")
    return total_reward


def continue_training():
    """
    Loads the model, evaluates it, and decides whether to continue training.
    Retrains if performance is below a set threshold (15% of initial investment).
    """
    try:
        model = PPO.load("trading_agent")
        print("✅ Loaded existing model.")

        # Evaluate the model before training
        initial_reward = evaluate_model()
        reward_threshold = (REWARD_THRESHOLD_PERCENT / 100) * INITIAL_BALANCE

        print(f"📊 Initial Estimated Reward: {initial_reward}")
        print(f"🔹 Retraining Threshold: {reward_threshold} (15% of initial investment)")

        if initial_reward < reward_threshold:
            print("⚠️ Reward too low, retraining the model...")
            env = DummyVecEnv([lambda: CryptoTradingEnv()])
            model.set_env(env)
            model.learn(total_timesteps=5000)
            model.save("trading_agent")
            print("✅ Model updated and saved.")
        else:
            print("🚀 Model performance is good, no need for retraining.")

    except Exception as e:
        print(f"⚠️ No model found. Training from scratch... ({e})")
        model = PPO("MlpPolicy", DummyVecEnv([lambda: CryptoTradingEnv()]), verbose=1)
        model.learn(total_timesteps=5000)
        model.save("trading_agent")
        print("✅ Model trained and saved.")

# Export functions for use in other scripts
__all__ = ["continue_training", "evaluate_model"]
