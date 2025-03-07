import unittest
import os
import time
from unittest.mock import patch
from src.model_training import continue_training, is_model_optimal, evaluate_model

class TestModelTraining(unittest.TestCase):
    """
    Unit tests for the model training and evaluation functions.
    """

    def setUp(self):
        """Ensure the model file is accessible before testing."""
        self.model_file = "trading_agent.zip"

        # Ensure model file exists before testing
        if not os.path.exists(self.model_file):
            with open(self.model_file, "w") as f:
                f.write("Initial Model")

    @patch("src.model_training.is_model_optimal", return_value=(False, 0.1, 0.2, 0.3))
    def test_model_training(self, mock_is_model_optimal):
        """Tests if the model retrains successfully and updates the model file."""

        before_training = os.path.getmtime(self.model_file)

        try:
            time.sleep(1)  # Ensure filesystem timestamp updates
            continue_training()
            success = True
        except Exception as e:
            print(f"❌ Training function failed: {e}")
            success = False

        self.assertTrue(success, "Training function failed.")

        after_training = os.path.getmtime(self.model_file)

        self.assertGreater(after_training, before_training, "Model file was not updated!")

    def test_evaluate_model(self):
        """Tests if evaluate_model() returns a valid reward value."""
        try:
            reward = evaluate_model()
            self.assertIsInstance(reward, float, "Reward should be a float")
            self.assertGreaterEqual(reward, 0, "Reward should not be negative")
        except Exception as e:
            self.fail(f"❌ evaluate_model() failed: {e}")

    def test_model_optimal(self):
        """Tests is_model_optimal() with different loss values."""

        with patch("src.model_training.TrainingMetricsCallback", autospec=True) as mock_callback:
            # Mock training metrics
            mock_callback.return_value.policy_loss = [0.00001]
            mock_callback.return_value.value_loss = [0.00001]
            mock_callback.return_value.entropy_loss = [0.1]

            optimal, policy_loss, value_loss, entropy_loss = is_model_optimal()

            self.assertIsInstance(optimal, bool, "Optimal should be a boolean")
            self.assertIsInstance(policy_loss, float, "Policy loss should be a float")
            self.assertIsInstance(value_loss, float, "Value loss should be a float")
            self.assertIsInstance(entropy_loss, float, "Entropy loss should be a float")

if __name__ == '__main__':
    unittest.main()