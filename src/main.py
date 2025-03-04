import os
import time
import numpy as np
from dotenv import load_dotenv
from telegram_bot import TelegramBot
from model_training import continue_training, evaluate_model

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

def should_retrain(model):
    """
    Determines whether the model should be retrained based on performance metrics.
    - Uses a mini-training step to check `policy_loss` and `value_loss`.
    - Retrains only if learning performance degrades.
    """
    train_info = model.learn(total_timesteps=100, log_interval=10)
    policy_loss = train_info.get("policy_loss", 0)
    value_loss = train_info.get("value_loss", 0)

    print(f"📉 Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.2f}")

    # Retrain only if the model is struggling to learn effectively
    return abs(policy_loss) > 0.1 or abs(value_loss) > 1000

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
