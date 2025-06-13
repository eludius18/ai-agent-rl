# AI-Agent-RL

A self-evolving reinforcement learning (RL) AI Agent trading bot enhanced with CrewAI agents for intelligent decision-making and validation. The bot monitors crypto markets, evaluates trading opportunities using PPO, and verifies decisions using a dynamic team of agents powered by local or cloud-based LLMs. Telegram alerts keep you updated in real-time.

 
## 🚀 Key Features

- 🤖 **Self-Evolving AI**: Learns and refines trading strategies using PPO reinforcement learning.
- 📉 **Market-Aware Decision Engine**: Detects patterns and evaluates risks in real-time.
- 🔁 **Smart Retraining Logic**: Triggers model retraining only when performance metrics fall below thresholds.
- 🧠 **Multi-Agent CrewAI Reasoning**:  
  Uses **CrewAI** agents to enhance decision-making:
  - `Market Analyst Agent`: Interprets market indicators and flags volatility trends.
  - `RL Decision Agent`: Validates the action proposed by the reinforcement learning model.
  - `Risk Manager Agent`: Approves or rejects trades with rationale based on AI and market logic.
- 🔗 **Flexible LLM Backend**: Easily switch between `Ollama`, `OpenAI`, `Anthropic`, or others via `.env`.
- 💬 **Telegram Alerts**: Sends smart alerts and retraining notifications to your device.
- ⚙️ **Fully Modular Design**: Swap out LLMs, trading logic, or comms layer without changing the code.
- ✅ **Test Coverage**: Unit tests ensure training, evaluation, and environment stability.

## 📐 Architecture

```text
+------------------+     +----------------+     +---------------------+
|   Trading Env    | --> |  PPO Model     | --> |   Estimated Reward   |
+------------------+     +----------------+     +---------------------+
                                                      |
                                                      v
                                               +-------------+
                                               | CrewAI Flow |
                                               +-------------+
                                                      |
             +--------------------+--------------------+---------------------+
             |                    |                    |                     |
             v                    v                    v                     |
  +------------------+  +---------------------+  +---------------------+     |
  | Market Analyst    |  | RL Decision Agent   |  |  Risk Manager Agent |     |
  +------------------+  +---------------------+  +---------------------+     |
             \_________________________Validated Action_____________________/
                                                      |
                                                      v
                                           +------------------------+
                                           |  Telegram Notification |
                                           +------------------------+
                                                      |
                                               +--------------+
                                               |  Retraining? |
                                               +--------------+
                                                      |
                                 Yes (if performance drops) |
                                                      v
                                             +------------------+
                                             | Continue Training |
                                             +------------------+
```

## 🧰 Tech Stack

| Layer            | Technology                                                                 |
|------------------|----------------------------------------------------------------------------|
| **LLM Runtime**   | Ollama (local), OpenAI, Anthropic *(dynamic via `.env`)*                  |
| **LLM Orchestration** | [CrewAI](https://github.com/joaomdmoura/crewAI), [LangChain](https://www.langchain.com/) |
| **Agents**        | Multi-agent team (Market Analyst, RL Validator, Risk Manager)             |
| **RL Model**      | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO      |
| **Training Env**  | Custom OpenAI Gym-style `CryptoTradingEnv`                                |
| **Market Data**   | Real-time prices via [CoinGecko API](https://www.coingecko.com/)          |
| **Comms**         | Telegram Bot API                                                          |
| **Config**        | `.env` + `python-dotenv`                                                  |
| **Testing**       | `unittest`                                                                |
| **Runtime**       | Python 3.11 + virtualenv                                                  |

## 📦 Installation

```bash
git clone https://github.com/eludius18/ai-agent-rl.git
cd ai-agent-rl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the root directory with the following environment variables:

```env
# Telegram Bot Configuration
TELEGRAM_TOKEN=        # Telegram Bot Token
CHAT_ID=               # Your Telegram Chat ID
TELEGRAM_ENABLE=       # Set to 1 to enable Telegram alerts

# Crypto API Configuration
BINANCE_API_KEY=       # Your Binance API Key
BINANCE_API_SECRET=    # Your Binance API Secret
COINGECKO_API=         # CoinGecko API URL (e.g., https://api.coingecko.com/api/v3)

# AI Training Parameters
INITIAL_BALANCE=       # Starting balance for simulations
TRADE_ALERT_THRESHOLD= # Minimum profit threshold to trigger an alert
CHECK_INTERVAL=        # Interval in seconds to check for trade opportunities
MODEL_PATH=            # Path to save/load the RL model (e.g., trading_agent.zip)

# CrewAI LLM Configuration
CREWAI_LLM_PROVIDER=   # Python module providing the LLM (e.g., langchain_ollama)
CREWAI_LLM_CLASS=      # Class name to use from the provider (e.g., OllamaLLM)
CREWAI_LLM_MODEL=      # Model to use (e.g., ollama/mistral)

# Model Health Check and Retraining
MODEL_CHECK_IMPROVEMENT_INTERVAL=  # Time in seconds between health checks
POLICY_LOSS_THRESHOLD=             # Maximum policy loss before triggering retraining
VALUE_LOSS_THRESHOLD=              # Maximum value loss before triggering retraining
ENTROPY_LOSS_THRESHOLD=            # Minimum acceptable entropy loss to avoid overfitting

TELEGRAM_TOKEN=your_token
CHAT_ID=your_chat_id
MODEL_PATH=trading_agent.zip
CREWAI_LLM_PROVIDER=langchain_ollama
CREWAI_LLM_CLASS=OllamaLLM
CREWAI_LLM_MODEL=mistral
```

## 🏁 Usage

Start the agent loop:

```bash
python src/main.py
```

The script will:
- Evaluate the model
- Trigger CrewAI agents for decision validation
- Notify via Telegram if action is approved and defined
- Retrain if model performance degrades

## 🧪 Testing

```bash
pytest
```

## 🧠 Credits

This project integrates state-of-the-art reinforcement learning with AI-driven validation agents to build a smarter, self-adaptive trading bot.