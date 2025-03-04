import unittest
import os
import numpy as np
from src.model_training import continue_training, evaluate_model

class TestModelTraining(unittest.TestCase):
    """
    Unit tests for the model training and evaluation functions.
    """

    def test_model_training(self):
        """Tests if the model trains successfully without errors and updates the model file."""
        model_file = "trading_agent.zip"

        # Get last modified time before training
        before_training = os.path.getmtime(model_file) if os.path.exists(model_file) else None

        try:
            continue_training()
            success = True
        except Exception as e:
            print(f"❌ Training function failed: {e}")
            success = False

        # Ensure training ran without errors
        self.assertTrue(success, "Training function failed.")

        # Check if model file was updated
        if before_training is not None:
            after_training = os.path.getmtime(model_file)
            self.assertGreater(after_training, before_training, "Model file was not updated!")

    def test_model_evaluation(self):
        """Tests if the model can be evaluated correctly and returns a valid numeric reward."""
        try:
            reward = evaluate_model()
            
            # Ensure it's a numeric float, handling NumPy types
            self.assertIsInstance(float(reward), float, "Reward should be a Python float")
            
        except Exception as e:
            self.fail(f"❌ Evaluation function failed: {e}")

if __name__ == '__main__':
    unittest.main()