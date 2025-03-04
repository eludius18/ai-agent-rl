import gym
import numpy as np
import os
from binance.client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Initialize Binance Client
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=True)

class CryptoTradingEnv(gym.Env):
    """
    Custom OpenAI Gym environment for cryptocurrency trading.
    The agent will learn to Buy, Hold, or Sell based on past prices.
    """
    def __init__(self):
        super(CryptoTradingEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 0: Buy, 1: Hold, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.data = self.get_crypto_data()
        self.current_step = 0
        self.balance = 1000  # Starting balance
        self.crypto_held = 0

    def get_crypto_data(self):
        """
        Fetches BTC/USDT price data from Binance.
        """
        klines = binance_client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_15MINUTE, limit=500)
        prices = np.array([float(k[4]) for k in klines])  # Closing prices
        return prices

    def step(self, action):
        """
        Executes a trade and updates balance.
        """
        current_price = self.data[self.current_step]
        reward = 0

        if action == 0:  # Buy
            self.crypto_held += self.balance / current_price
            self.balance = 0
        elif action == 2 and self.crypto_held > 0:  # Sell
            self.balance += self.crypto_held * current_price
            self.crypto_held = 0
            reward = self.balance - 1000  # Profit/Loss

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = np.array([self.data[self.current_step - i] for i in range(10)])

        return obs, reward, done, {}

    def reset(self):
        """
        Resets the environment.
        """
        self.current_step = 0
        self.balance = 1000
        self.crypto_held = 0
        return np.array([self.data[self.current_step - i] for i in range(10)])
