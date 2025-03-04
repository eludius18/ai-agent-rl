import unittest
import numpy as np
from src.trading_env import CryptoTradingEnv

class TestCryptoTradingEnv(unittest.TestCase):
    def setUp(self):
        """Initialize the environment before each test."""
        self.env = CryptoTradingEnv()
        self.env.reset()

    def test_initial_state(self):
        """Ensure the initial state has the correct shape."""
        obs = self.env.reset()
        self.assertEqual(obs.shape, (10,), "Observation space is incorrect")

    def test_step_function(self):
        """Test if the step function updates the environment correctly."""
        action = 0  # Buy
        obs, reward, done, _ = self.env.step(action)
        self.assertEqual(obs.shape, (10,), "Observation shape after step is incorrect")
        self.assertIsInstance(reward, (int, float), "Reward should be numeric")
        self.assertIsInstance(done, bool, "Done flag should be boolean")

    def test_environment_reset(self):
        """Check if reset restores initial conditions."""
        self.env.step(0)  # Perform an action
        self.env.reset()  # Reset environment
        self.assertEqual(self.env.balance, 1000, "Balance should reset to 1000 USD")
        self.assertEqual(self.env.crypto_held, 0, "Crypto held should reset to 0")

if __name__ == '__main__':
    unittest.main()