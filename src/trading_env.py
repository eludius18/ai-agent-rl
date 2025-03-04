import os
import time
import gym
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
COINGECKO_API = os.getenv("COINGECKO_API")

class CryptoTradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for cryptocurrency trading.
    The agent learns to Buy, Hold, or Sell based on past prices.
    """

    def __init__(self):
        super(CryptoTradingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 0: Buy, 1: Hold, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.data = self.get_crypto_data()
        self.current_step = 0
        self.balance = float(os.getenv("INITIAL_BALANCE", 1000))  # Initial balance
        self.crypto_held = 0

    def get_crypto_data(self):
        """
        Fetches BTC/USDT price data from CoinGecko with exponential backoff.
        If API fails, falls back to synthetic price data.
        """
        url = f"{COINGECKO_API}/coins/bitcoin/market_chart?vs_currency=usd&days=7"
        max_retries = 5
        retry_delay = 2  # Start with 2 seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()

                prices = response.json()["prices"]
                prices = np.array([p[1] for p in prices])  # Extract closing prices
                print("✅ Loaded real price data from CoinGecko (Hourly Data)")
                return prices

            except requests.exceptions.RequestException as e:
                print(f"⚠️ API Error (Attempt {attempt + 1}/{max_retries}): {e}")
                if response.status_code == 429:  # Too Many Requests
                    print(f"⏳ Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    break  # Exit loop if it's another kind of failure

        print("⚙️ Using synthetic price data instead.")
        np.random.seed(42)
        return np.linspace(30000, 35000, num=500) + np.random.randn(500) * 500  # Synthetic data

    def step(self, action):
        """
        Executes a trade and updates balance.
        """
        current_price = self.data[self.current_step]
        reward = 0

        if action == 0:  # Buy
            if self.balance > 0:
                self.crypto_held += self.balance / current_price
                self.balance = 0
        elif action == 2 and self.crypto_held > 0:  # Sell
            self.balance += self.crypto_held * current_price
            self.crypto_held = 0
            reward = self.balance - float(os.getenv("INITIAL_BALANCE", 1000))  # Profit/Loss

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = np.array([self.data[max(0, self.current_step - i)] for i in range(10)])

        return obs, reward, done, {}

    def reset(self):
        """
        Resets the environment.
        """
        self.current_step = 0
        self.balance = float(os.getenv("INITIAL_BALANCE", 1000))
        self.crypto_held = 0
        return np.array([self.data[self.current_step - i] for i in range(10)])