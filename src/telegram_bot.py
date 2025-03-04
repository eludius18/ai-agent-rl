import telegram
from telegram.ext import Updater, CommandHandler
from model_training import evaluate_model, continue_training

class TelegramBot:
    """
    Telegram bot for trading alerts and retraining.
    """
    def __init__(self, token, chat_id):
        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id
        self.updater = Updater(token=token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self.dispatcher.add_handler(CommandHandler("retrain", self.retrain))

    def retrain(self, update, context):
        continue_training()
        update.message.reply_text("📊 Model retrained and updated!")
