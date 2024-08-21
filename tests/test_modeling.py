import unittest
from src.modeling.price_predictor import PricePredictor
from src.config import PROCESSED_DATA_PATH


class TestModeling(unittest.TestCase):
    """
    Unit tests for the modeling class.

    Attributes:
        FIXTURES (str): Directory path for fixtures.
    """

    FIXTURES = PROCESSED_DATA_PATH

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Initialize and train the model
        self.predictor = PricePredictor(data=self.FIXTURES)
        self.predictor.train_models()

    def test_preprocess(self):
        """ Test Model Training. """
        self.predictions = self.predictor.predict(future_steps=10)
        self.assertIsNotNone(self.predictions, msg="training did not complete")


if __name__ == "__main__":
    unittest.main()
