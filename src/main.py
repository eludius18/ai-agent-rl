import os
import time
import numpy as np
from dotenv import load_dotenv
from telegram_bot import TelegramBot
from model_training import continue_training, evaluate_model
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import CryptoTradingEnv

# Load environment variables
load_dotenv()

# Fetch environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL"))
TRADE_ALERT_THRESHOLD = float(os.getenv("TRADE_ALERT_THRESHOLD"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE"))
MODEL_PATH = os.getenv("MODEL_PATH")

# Initialize Telegram Bot
telegram_bot = TelegramBot(TELEGRAM_TOKEN, CHAT_ID)

def model_exists():
    """Checks if a pre-trained model exists."""
    return os.path.exists(MODEL_PATH)

class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to track training loss metrics.
    Extracts policy_loss and value_loss after each training iteration.
    """
    def __init__(self):
        super().__init__()
        self.policy_loss = []
        self.value_loss = []

    def _on_step(self) -> bool:
        logs = self.model.logger.name_to_value
        if "train/policy_loss" in logs:
            self.policy_loss.append(logs["train/policy_loss"])
        if "train/value_loss" in logs:
            self.value_loss.append(logs["train/value_loss"])
        return True  # Continue training

def should_retrain(model):
    """
    Determines if the model should be retrained based on training performance.
    Uses a mini-training session to track `policy_loss` and `value_loss`.
    """
    print("📊 Checking model training metrics...")

    # Ensure model has an environment set
    env = DummyVecEnv([lambda: CryptoTradingEnv()])
    model.set_env(env)

    # Run a small training session to gather loss metrics
    callback = TrainingMetricsCallback()
    model.learn(total_timesteps=200, callback=callback)

        # Print ALL logs captured by the model
    print(f"📜 Logged values: {model.logger.name_to_value}")

    if not callback.policy_loss or not callback.value_loss:
        print("⚠️ No training loss data available. Skipping retraining.")
        return False

    avg_policy_loss = np.mean(callback.policy_loss)
    avg_value_loss = np.mean(callback.value_loss)

    print(f"📉 Avg Policy Loss: {avg_policy_loss:.4f}, Avg Value Loss: {avg_value_loss:.2f}")

    # Retrain if policy loss is too high or value loss exceeds threshold
    return abs(avg_policy_loss) > 0.1 or abs(avg_value_loss) > 1000

def check_for_opportunity():
    """
    Evaluates the model and sends a Telegram alert if a trading opportunity meets the threshold.
    """
    print("🔍 Checking for trading opportunities...")

    if not model_exists():
        print("⚠️ No model found. Training from scratch...")
        continue_training()
        return

    estimated_reward = evaluate_model()

    if estimated_reward > (TRADE_ALERT_THRESHOLD / 100) * INITIAL_BALANCE:
        message = f"🚀 Trading Opportunity! Estimated Profit: ${estimated_reward:.2f}."
        telegram_bot.send_message(message)
        print("📩 Alert Sent to Telegram.")
    else:
        print("⚠️ No profitable trade detected yet.")

def main():
    """
    Runs the bot continuously, checking for opportunities and retraining if needed.
    """
    while True:
        check_for_opportunity()

        # Load existing model for evaluation
        model = PPO.load(MODEL_PATH)

        # Check model performance before deciding on retraining
        if should_retrain(model):
            print("🔄 Retraining the model due to learning degradation...")
            telegram_bot.send_message("🔄 Retraining model due to learning degradation...")
            continue_training()
            telegram_bot.send_message("✅ Model retrained successfully!")

        print(f"⏳ Waiting {CHECK_INTERVAL} seconds before next check...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()