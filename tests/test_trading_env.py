import unittest
import numpy as np
from src.trading_env import CryptoTradingEnv

class TestCryptoTradingEnv(unittest.TestCase):
    """Unit tests for the custom trading environment."""

    def setUp(self):
        """Initializes the environment before each test."""
        self.env = CryptoTradingEnv()
        self.env.reset()

    def test_initial_state(self):
        """Ensures the initial state has the correct shape and balance."""
        obs = self.env.reset()

        # Validate observation space shape
        self.assertEqual(obs.shape, (10,), "Observation space is incorrect")

        # Validate initial balance and crypto holdings
        self.assertEqual(self.env.balance, 1000, "Balance should start at 1000 USD")
        self.assertEqual(self.env.crypto_held, 0, "Crypto holdings should start at 0")

    def test_step_function(self):
        """Tests if the step function updates the environment correctly."""
        action = 0  # Buy action
        obs, reward, done, _ = self.env.step(action)

        # Validate observation shape after step
        self.assertEqual(obs.shape, (10,), "Observation shape after step is incorrect")

        # Ensure reward is numeric
        self.assertIsInstance(float(reward), float, "Reward should be numeric")

        # Ensure 'done' flag is boolean
        self.assertIsInstance(done, bool, "Done flag should be boolean")

if __name__ == '__main__':
    unittest.main()