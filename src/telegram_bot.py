import logging
import asyncio
import aiohttp
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from dotenv import load_dotenv
from model_training import evaluate_model, continue_training

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class TelegramBot:
    """
    Telegram bot for triggering model evaluation and retraining.
    """

    def __init__(self, token: str, chat_id: str):
        """
        Initialize the bot, application, and command handlers.
        """
        self.token = token
        self.chat_id = chat_id
        self.application = Application.builder().token(token).build()

        # Register bot commands
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("check", self.check))
        self.application.add_handler(CommandHandler("retrain", self.retrain))
        self.application.add_handler(CommandHandler("help", self.help))

    async def start(self, update: Update, context: CallbackContext) -> None:
        """
        Respond to /start command with instructions.
        """
        logging.info("Received /start command")
        message = (
            "🤖 **AI Trading Bot is online!**\n"
            "Available commands:\n"
            "🔹 `/check` - Check for trading opportunities\n"
            "🔹 `/retrain` - Retrain the model\n"
            "🔹 `/help` - Show help menu"
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    async def check(self, update: Update, context: CallbackContext) -> None:
        """
        Evaluate the trading model and respond with the result.
        """
        logging.info("Received /check command")
        reward = evaluate_model()
        message = (
            f"📊 **Trading Evaluation**\n✅ Estimated Profit: **${reward:.2f}**"
            if reward > 0 else "⚠️ No profitable trade detected."
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    async def retrain(self, update: Update, context: CallbackContext) -> None:
        """
        Trigger model retraining asynchronously.
        """
        logging.info("Received /retrain command")
        await update.message.reply_text("🔄 Retraining model... Please wait.", parse_mode="Markdown")
        await asyncio.to_thread(continue_training)
        await update.message.reply_text("✅ Model retrained successfully!", parse_mode="Markdown")

    async def help(self, update: Update, context: CallbackContext) -> None:
        """
        Provide a help menu with available commands.
        """
        logging.info("Received /help command")
        message = (
            "📌 **Available Commands:**\n"
            "🔹 `/start` - Start the bot\n"
            "🔹 `/check` - Check for trading opportunities\n"
            "🔹 `/retrain` - Retrain the model\n"
            "🔹 `/help` - Show this help menu"
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    async def send_message_async(self, message: str) -> None:
        """
        Send a message asynchronously via Telegram Bot API.
        """
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    logging.error(f"⚠️ Telegram Error: {await response.text()}")

    def send_message(self, message: str) -> None:
        """
        Safe wrapper for sending a message from a non-async context.
        """
        asyncio.run(self.send_message_async(message))

    def run(self):
        """
        Run the Telegram bot in a background thread without signal handling.
        Avoids `set_wakeup_fd` issues on macOS and other non-main threads.
        """
        import threading

        def start_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def runner():
                await self.application.initialize()
                await self.application.start()
                await self.application.bot.delete_webhook()
                await self.application.update_queue.join()
                await self.application.stop()
                
            loop.run_until_complete(runner())

        bot_thread = threading.Thread(target=start_bot, name="TelegramBotThread", daemon=True)
        bot_thread.start()