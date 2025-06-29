import unittest
from unittest.mock import AsyncMock, patch
from src.telegram_bot import TelegramBot

class TestTelegramBot(unittest.TestCase):
    """
    Tests for the Telegram bot functionality.
    """

    @patch("src.telegram_bot.Application")
    def setUp(self, mock_application):
        """Initialize the bot with test token and chat_id."""
        self.bot = TelegramBot("TEST_TOKEN", "TEST_CHAT_ID")

    @patch("src.telegram_bot.TelegramBot.send_message_async", new_callable=AsyncMock)
    def test_send_message(self, mock_send_message):
        """Ensure the bot sends messages correctly."""
        message = "Test message"
        self.bot.send_message(message)
        mock_send_message.assert_called_once_with(message)

    @patch("src.telegram_bot.evaluate_model", return_value=5.0)
    async def test_check_command(self, mock_evaluate, mock_update, mock_context):
        """Ensure the /check command works correctly."""
        await self.bot.check(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_with("📊 **Trading Evaluation**\n✅ Estimated Profit: **$5.00**", parse_mode="Markdown")

    @patch("src.telegram_bot.continue_training")
    async def test_retrain_command(self, mock_retrain, mock_update, mock_context):
        """Ensure the /retrain command triggers training."""
        await self.bot.retrain(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_with("✅ Model retrained successfully!", parse_mode="Markdown")

if __name__ == '__main__':
    unittest.main()
