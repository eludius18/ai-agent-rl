import os
import time
import threading
import logging
import numpy as np
from dotenv import load_dotenv
from telegram_bot import TelegramBot
from model_training import continue_training, evaluate_model, is_model_optimal
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import CryptoTradingEnv

# Load environment variables
load_dotenv()

# Fetch environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL"))
MODEL_CHECK_IMPROVEMENT_INTERVAL = int(os.getenv("MODEL_CHECK_IMPROVEMENT_INTERVAL"))
TRADE_ALERT_THRESHOLD = float(os.getenv("TRADE_ALERT_THRESHOLD"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE"))
MODEL_PATH = os.getenv("MODEL_PATH")
TELEGRAM_ENABLE = int(os.getenv("TELEGRAM_ENABLE"))

# Model Thresholds
POLICY_LOSS_THRESHOLD = float(os.getenv("POLICY_LOSS_THRESHOLD"))
VALUE_LOSS_THRESHOLD = float(os.getenv("VALUE_LOSS_THRESHOLD"))
ENTROPY_LOSS_THRESHOLD = float(os.getenv("ENTROPY_LOSS_THRESHOLD"))

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Telegram Bot only if Telegram is enabled
if TELEGRAM_ENABLE:
    telegram_bot = TelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    bot_thread = threading.Thread(target=telegram_bot.run, daemon=True)
    bot_thread.start()
else:
    telegram_bot = None


def check_for_opportunity():
    """
    Evaluates the model and sends a Telegram alert if a trading opportunity meets the threshold.
    """
    logging.info("🔍 Checking for trading opportunities...")

    estimated_reward = evaluate_model()
    profit_percentage = (estimated_reward / INITIAL_BALANCE) * 100

    if estimated_reward > (TRADE_ALERT_THRESHOLD / 100) * INITIAL_BALANCE:
        message = (
            "📊 **Trading Opportunity Detected!**\n"
            f"💰 **Estimated Profit:** ${estimated_reward:.2f} ({profit_percentage:.2f}%)\n"
            "🚀 **Market conditions are favorable!**"
        )
        logging.info(message)
        if TELEGRAM_ENABLE:
            telegram_bot.send_message(message)


def main():
    """
    Runs the bot continuously, checking for opportunities and retraining if needed.
    """
    last_model_check = 0

    while True:
        check_for_opportunity()

        current_time = time.time()
        if current_time - last_model_check >= MODEL_CHECK_IMPROVEMENT_INTERVAL:
            logging.info("🔍 Checking model health...")
            
            # Log values before making a retraining decision
            logging.info("🔎 Checking if model needs retraining...")
            optimal, policy_loss, value_loss, entropy_loss = is_model_optimal()


            logging.info(f"🔎 Evaluated Model -> Policy Loss: {policy_loss:.4f}, "
                         f"Value Loss: {value_loss:.4f}, Entropy Loss: {entropy_loss:.4f}")
            logging.info(f"🔎 Model optimal? {optimal}")

            if not optimal:
                logging.info(f"🔄 Retraining the model due to suboptimal performance!")
                logging.info(f"📊 Policy Loss: {policy_loss:.4f} (Threshold: {POLICY_LOSS_THRESHOLD})")
                logging.info(f"📊 Value Loss: {value_loss:.4f} (Threshold: {VALUE_LOSS_THRESHOLD})")
                logging.info(f"📊 Entropy Loss: {entropy_loss:.4f} (Min Required: {ENTROPY_LOSS_THRESHOLD})")
                
                continue_training()
                if TELEGRAM_ENABLE:
                    telegram_bot.send_message(
                        f"🔄 **Retraining Triggered!**\n"
                        f"📊 **Policy Loss:** {policy_loss:.4f} (Threshold: {POLICY_LOSS_THRESHOLD})\n"
                        f"📊 **Value Loss:** {value_loss:.4f} (Threshold: {VALUE_LOSS_THRESHOLD})\n"
                        f"📊 **Entropy Loss:** {entropy_loss:.4f} (Min Required: {ENTROPY_LOSS_THRESHOLD})\n"
                        "⚠️ **Model performance dropped below acceptable levels. Retraining now!**"
                    )

            last_model_check = time.time()

        logging.info(f"⏳ Waiting {CHECK_INTERVAL} seconds before next check...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
