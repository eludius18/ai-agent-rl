# AI Agent RL - Crypto Trading Bot with Reinforcement Learning

## 📌 Project Overview
AI Agent RL is an **AI-powered cryptocurrency trading bot** utilizing **Reinforcement Learning (RL)** with **Proximal Policy Optimization (PPO)**. This bot learns trading strategies over time, optimizing buy/sell decisions based on historical data and live market conditions. It also integrates with **Telegram** for real-time alerts and **automatic retraining** when performance degrades.

## 🚀 Features

- 🚀 **Self-Optimizing AI**: The model continuously improves by evaluating its own learning performance.  
- 🐿 **Automated Trading Alerts**: Uses real-time market data to detect trading opportunities.  
- 🔄 **Smart Retraining**: Retrains **only** when performance metrics indicate degradation.  
- 📊 **Synthetic Data Fallback**: If live data is unavailable, it generates realistic price patterns.  
- 💬 **Telegram Integration**: Sends alerts and allows model retraining via Telegram commands.  
- 🛠 **Fully Modular**: Components are separate for easy maintenance and upgrades.  
- ✅ **Unit Testing**: Automated tests included for stability.

## 📂 Directory Structure
```
AI-Agent-RL/
│── src/
│   ├── main.py                # Main execution script
│   ├── trading_env.py         # Custom Gym-based trading environment
│   ├── model_training.py      # AI model training & evaluation logic
│   ├── telegram_bot.py        # Telegram bot for alerts & retraining
│── tests/
│   ├── test_trading_env.py    # Unit tests for trading environment
│   └── test_model_training.py # Unit tests for model training
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
# Telegram API
TELEGRAM_TOKEN=your_telegram_bot_token
CHAT_ID=your_telegram_chat_id

# Crypto API (CoinGecko, Binance, or another provider)
COINGECKO_API=https://api.coingecko.com/api/v3

# AI Training Config
INITIAL_BALANCE=your_initial_balance
REWARD_THRESHOLD_PERCENT=your_reward_threshold_percent
TRADE_ALERT_THRESHOLD=your_trade_alert_threshold
CHECK_INTERVAL=your_check_interval
MODEL_PATH=your_model_path
```
**API Keys:**  
- **Telegram:** Create a bot using [BotFather](https://t.me/BotFather).  

## 🚀 Running the Bot
```sh
python src/main.py
```

## 🛠️ Running Tests
To ensure stability, run unit tests:
```sh
PYTHONPATH=src python -m unittest discover tests/
```

## 💡 How It Works
### 🌐 PPO Algorithm Explained
PPO (**Proximal Policy Optimization**) is an advanced RL algorithm that helps the AI improve its **trading strategies**. The model learns from past price movements, adjusting **buy/sell/hold** decisions based on its reward function.

### 🌍 Model Learning Workflow
1. **Fetch Market Data:** Uses real-time prices from CoinGecko.
2. **Predict Actions:** Chooses whether to Buy, Hold, or Sell.
3. **Evaluate Performance:** Tracks the AI's success over time.
4. **Retraining (if needed):** If performance degrades, the model retrains itself automatically.
5. **Send Alerts:** Telegram notifications for potential trade opportunities.

### ⚡ Telegram Commands
| Command  | Description  |
|----------|-------------|
| `/start` | Start the bot |
| `/check` | Get a trading alert (model evaluation) |
| `/retrain` | Force retraining the model |

## 📈 When Does the AI Retrain Itself?
Unlike basic trading bots, this AI **only retrains when needed**. It evaluates its **own learning efficiency** and **policy loss**:
- **If the AI is struggling to learn**, it retrains itself automatically.
- **If no valid trade opportunities arise, but learning is stable, it doesn’t retrain.**
- **Retraining isn’t affected by short-term market dips.**

## 💚 License
MIT License - Free to use and modify!

🚀 **Start your AI-driven crypto trading today!**