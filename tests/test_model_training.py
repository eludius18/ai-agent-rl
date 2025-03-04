import unittest
from src.model_training import continue_training, evaluate_model

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        """Test if the model trains successfully without errors."""
        try:
            continue_training()
            success = True
        except Exception as e:
            print(f"Error during training: {e}")
            success = False
        self.assertTrue(success, "Training function failed.")

    def test_model_evaluation(self):
        """Test if the model can be evaluated correctly."""
        try:
            reward = evaluate_model()
            self.assertIsInstance(reward, (int, float), "Reward should be numeric")
        except Exception as e:
            self.fail(f"Evaluation function failed: {e}")

if __name__ == '__main__':
    unittest.main()