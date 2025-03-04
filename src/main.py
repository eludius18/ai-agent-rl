import os
from dotenv import load_dotenv
from telegram_bot import TelegramBot
from model_training import continue_training

# Load environment variables from .env file
load_dotenv()

# Initialize Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

telegram_bot = TelegramBot(TELEGRAM_TOKEN, CHAT_ID)

def main():
    """
    Ensures the model is trained and starts the Telegram bot.
    """
    continue_training()  # Make sure the model improves continuously
    telegram_bot.start()  # Start the Telegram bot

if __name__ == "__main__":
    main()
