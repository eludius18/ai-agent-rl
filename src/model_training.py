import os
import numpy as np
import logging
import time
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import CryptoTradingEnv

# Load environment variables
load_dotenv()

INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE"))
MODEL_PATH = os.getenv("MODEL_PATH")

# Model Health Thresholds
POLICY_LOSS_THRESHOLD = float(os.getenv("POLICY_LOSS_THRESHOLD"))
VALUE_LOSS_THRESHOLD = float(os.getenv("VALUE_LOSS_THRESHOLD"))
ENTROPY_LOSS_THRESHOLD = float(os.getenv("ENTROPY_LOSS_THRESHOLD"))

class TrainingMetricsCallback(BaseCallback):
    """ Extracts training loss metrics """
    def __init__(self):
        super().__init__()
        self.policy_loss = []
        self.value_loss = []
        self.entropy_loss = []

    def _on_step(self) -> bool:
        logs = self.model.logger.name_to_value
        if "train/policy_loss" in logs:
            self.policy_loss.append(logs["train/policy_loss"])
        if "train/value_loss" in logs:
            self.value_loss.append(logs["train/value_loss"])
        if "train/entropy_loss" in logs:
            self.entropy_loss.append(logs["train/entropy_loss"])
        return True

def evaluate_model():
    """ Runs the model and returns estimated reward """
    env = DummyVecEnv([lambda: CryptoTradingEnv()])
    model = PPO.load(MODEL_PATH)
    
    obs = env.reset()
    total_reward = 0
    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return float(total_reward.item())

def is_model_optimal():
    """
    Evaluates if the model meets the required thresholds.
    Returns a tuple: (bool, policy_loss, value_loss, entropy_loss)
    """
    if not os.path.exists(MODEL_PATH):
        return False, 0, 0, 0  # ✅ Model does not exist, force retraining

    model = PPO.load(MODEL_PATH)
    env = DummyVecEnv([lambda: CryptoTradingEnv()])
    model.set_env(env)

    callback = TrainingMetricsCallback()
    model.learn(total_timesteps=200, callback=callback)

    avg_policy_loss = np.mean(callback.policy_loss) if callback.policy_loss else 0
    avg_value_loss = np.mean(callback.value_loss) if callback.value_loss else 0
    avg_entropy_loss = np.mean(callback.entropy_loss) if callback.entropy_loss else 0

    needs_retraining = (
        (abs(avg_policy_loss) > POLICY_LOSS_THRESHOLD and avg_policy_loss > 0) or
        (abs(avg_value_loss) > VALUE_LOSS_THRESHOLD and avg_value_loss > 0) or
        (avg_entropy_loss < ENTROPY_LOSS_THRESHOLD and avg_entropy_loss > 0)
    )

    logging.info(f"🔎 Model Evaluation -> Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy Loss: {avg_entropy_loss:.4f}")
    logging.info(f"🔎 Thresholds -> Policy: {POLICY_LOSS_THRESHOLD}, Value: {VALUE_LOSS_THRESHOLD}, Entropy: {ENTROPY_LOSS_THRESHOLD}")
    logging.info(f"🔎 Needs retraining? {'Yes' if needs_retraining else 'No'}")

    return not needs_retraining, avg_policy_loss, avg_value_loss, avg_entropy_loss

def continue_training():
    """
    Retrains the model only if necessary, based on technical indicators.
    """
    model_exists = os.path.exists(MODEL_PATH)
    logging.info(f"🔍 Model file exists: {model_exists}")

    if model_exists:
        optimal, policy_loss, value_loss, entropy_loss = is_model_optimal()
        logging.info(f"🔍 Model optimal? {optimal} (Policy Loss: {policy_loss}, Value Loss: {value_loss}, Entropy Loss: {entropy_loss})")
        
        if optimal:
            logging.info("✅ Model is performing optimally. No retraining needed.")
            return
        
        logging.warning("🔄 Model Retraining Triggered!")
    else:
        logging.warning("⚠️ No existing model found. Training a new one from scratch.")

    # Log before training
    if model_exists:
        before_training_time = os.path.getmtime(MODEL_PATH)
        logging.info(f"🕒 Model file timestamp before training: {before_training_time}")

    # Initialize environment
    env = DummyVecEnv([lambda: CryptoTradingEnv()])
    
    if model_exists:
        logging.info("📂 Loading existing model...")
        model = PPO.load(MODEL_PATH)
        model.set_env(env)
    else:
        logging.info("📢 Initializing new PPO model...")
        model = PPO("MlpPolicy", env, verbose=1)

    logging.info("⏳ Training the model for 5000 timesteps...")
    model.learn(total_timesteps=5000)

    logging.info("💾 Deleting old model and saving new one...")

    # Ensure we delete the old model first
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        time.sleep(1)

    # Save new model
    model.save(MODEL_PATH)
    time.sleep(1)

    # Force update file timestamp
    os.utime(MODEL_PATH, (time.time(), time.time()))

    # Log after training
    after_training_time = os.path.getmtime(MODEL_PATH)
    logging.info(f"🕒 Model file timestamp after saving: {after_training_time}")

    logging.info("✅ Model updated and saved.")