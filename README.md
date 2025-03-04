# AI Agent RL - Crypto Trading Bot with Reinforcement Learning

## 📌 Project Overview
AI Agent RL is a cryptocurrency trading bot powered by **Reinforcement Learning (RL)**. The bot learns trading strategies using **PPO (Proximal Policy Optimization)** and improves over time. It integrates with **Telegram** for real-time alerts and manual retraining.

## 🚀 Features
✅ **Continuous Learning:** The AI model improves with every execution.  
✅ **Trading Automation:** Uses Binance API for real-time price data.  
✅ **Telegram Bot Integration:** Sends alerts and allows retraining.  
✅ **Modular Architecture:** Separate components for easy maintenance.  
✅ **Automated Testing:** Unit tests included for local execution.  

## 📂 Directory Structure
```
AI-Agent-RL/
│── src/
│   ├── main.py                # Main execution script
│   ├── trading_env.py         # Gym-based custom trading environment
│   ├── model_training.py      # RL model training & evaluation
│   ├── telegram_bot.py        # Telegram bot for alerts
│── tests/
│   ├── test_trading_env.py    # Unit tests for trading environment
│   ├── test_model_training.py # Unit tests for model training
│── .env                       # Environment variables (not committed)
│── requirements.txt           # Dependencies
│── README.md                  # Documentation
```

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/eludius18/ai-agent-rl.git
cd ai-agent-rl
```

### 2️⃣ Create a Virtual Environment (Recommended)
```sh
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file in the project root:
```
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
```
**API Keys:**  
- **Telegram:** Create a bot using [BotFather](https://t.me/BotFather).  
- **Binance:** Get API keys from [Binance API Management](https://www.binance.com/en/my/settings/api-management).  

## 🚀 Running the Bot
```sh
python src/main.py
```

## 🛠️ Running Tests (Local Execution)
To ensure the bot is working correctly, run the unit tests:
```sh
PYTHONPATH=src python -m unittest discover tests/
```

## 🛠️ How It Works
1. **Retrieves crypto price data** from Binance.
2. **Trains the AI model** using past price data.
3. **Predicts Buy, Hold, or Sell actions.**
4. **Sends trading alerts** to Telegram.
5. **Can retrain itself** over time for better accuracy.

## ⚡ Telegram Commands
| Command  | Description  |
|----------|-------------|
| `/start` | Start the bot |
| `/check` | Get a trading alert |
| `/retrain` | Retrain the model |

## 📊 Future Enhancements
✅ Add support for multiple cryptocurrencies.  
✅ Improve trading strategy with technical indicators.  
✅ Build a web dashboard for monitoring trades.  

## 📜 License
MIT License - Free to use and modify!

🚀 **Start your AI-driven crypto trading today!**