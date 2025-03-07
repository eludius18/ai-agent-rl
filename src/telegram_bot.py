import logging
import asyncio
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
import os
from dotenv import load_dotenv
from model_training import evaluate_model, continue_training

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

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
        Initializes the Telegram bot and registers command handlers.
        """
        self.token = token
        self.chat_id = chat_id
        self.application = Application.builder().token(token).build()

        # Register bot commands
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("check", self.check))
        self.application.add_handler(CommandHandler("retrain", self.retrain))
        self.application.add_handler(CommandHandler("help", self.help))  # Ahora la función existe

    async def start(self, update: Update, context: CallbackContext) -> None:
        """
        Responds to the /start command and provides usage instructions.
        """
        logging.info("Command /start received")
        message = (
            "🤖 **AI Trading Bot is running!**\n"
            "Use the following commands:\n"
            "🔹 `/check` - Check for trading opportunities\n"
            "🔹 `/retrain` - Retrain the model if needed\n"
            "🔹 `/help` - Show this help menu"
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    async def check(self, update: Update, context: CallbackContext) -> None:
        """
        Evaluates the model and sends an alert if a profitable trade is detected.
        """
        logging.info("Command /check received")
        reward = evaluate_model()
        message = f"📊 **Trading Evaluation**\n✅ Estimated Profit: **${reward:.2f}**" if reward > 0 else "⚠️ No profitable trade detected."
        await update.message.reply_text(message, parse_mode="Markdown")

    async def retrain(self, update: Update, context: CallbackContext) -> None:
        """
        Retrains the model asynchronously.
        """
        logging.info("Command /retrain received")
        await update.message.reply_text("🔄 Retraining model... Please wait.", parse_mode="Markdown")
        await asyncio.to_thread(continue_training)
        await update.message.reply_text("✅ Model retrained successfully!", parse_mode="Markdown")

    async def help(self, update: Update, context: CallbackContext) -> None:
        """
        Provides a help message listing available commands.
        """
        logging.info("Command /help received")
        message = (
            "📌 **Available Commands:**\n"
            "🔹 `/start` - Start the bot\n"
            "🔹 `/check` - Check for trading opportunities\n"
            "🔹 `/retrain` - Retrain the AI model\n"
            "🔹 `/help` - Show this menu"
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    async def send_message_async(self, message: str) -> None:
        """
        Sends a message to the Telegram chat asynchronously.
        """
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logging.error(f"⚠️ Telegram Error: {await response.text()}")

    def send_message(self, message: str) -> None:
        """
        Ensures that messages are sent safely from different threads.
        """
        asyncio.run(self.send_message_async(message))

    async def start_polling(application):
        await application.run_polling()

    def run(self):
        """
        Starts the Telegram bot in an asyncio event loop.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start_polling(self.application))