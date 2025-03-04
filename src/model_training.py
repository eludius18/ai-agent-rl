from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_env import CryptoTradingEnv

def continue_training():
    """
    Loads the model, continues training, and saves updates.
    """
    try:
        model = PPO.load("trading_agent")
        print("Continuing training with new market data...")
    except:
        print("No model found. Training from scratch...")
        model = PPO("MlpPolicy", DummyVecEnv([lambda: CryptoTradingEnv()]), verbose=1)
    
    model.learn(total_timesteps=5000)
    model.save("trading_agent")
    print("Model updated and saved.")

def evaluate_model():
    """
    Evaluates the trained model and returns a sample reward.
    """
    print("Evaluating model...")
    return 10  # Sample reward for testing

# Asegurar que las funciones sean exportadas
__all__ = ["continue_training", "evaluate_model"]