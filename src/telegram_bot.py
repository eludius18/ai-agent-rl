import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
import os
from dotenv import load_dotenv
from model_training import evaluate_model, continue_training

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Configure logging for debugging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class TelegramBot:
    """
    A Telegram bot for trading alerts and model retraining.
    """

    def __init__(self, token, chat_id):
        """
        Initializes the Telegram bot and command handlers.
        """
        self.chat_id = chat_id
        self.application = Application.builder().token(token).build()

        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("check", self.check))
        self.application.add_handler(CommandHandler("retrain", self.retrain))

    async def start(self, update: Update, context: CallbackContext) -> None:
        """
        Responds to the /start command.
        """
        await update.message.reply_text("🤖 AI Trading Bot is running! Use /check to get a trading alert.")

    async def check(self, update: Update, context: CallbackContext) -> None:
        """
        Responds to the /check command by evaluating the model and sending an alert.
        """
        reward = evaluate_model()
        await update.message.reply_text(f"📊 Estimated Trading Reward: {reward:.2f}")

    async def retrain(self, update: Update, context: CallbackContext) -> None:
        """
        Responds to the /retrain command by retraining the model.
        """
        await update.message.reply_text("🔄 Retraining model...")
        continue_training()
        await update.message.reply_text("✅ Model retrained successfully!")

    def run(self):
        """
        Starts the Telegram bot.
        """
        self.application.run_polling()