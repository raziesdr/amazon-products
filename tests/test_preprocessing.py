import unittest
from src.preprocessing.data_processor import DataProcessor
from src.config import RAW_DATA_PATH


class TestDataProcessor(unittest.TestCase):
    """
    Unit tests for the Preprocess class.

    Attributes:
        FIXTURES (str): Directory path for fixtures.
    """

    FIXTURES = RAW_DATA_PATH

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.processor = DataProcessor(file_path=self.FIXTURES)
        self.data = None

    def test_preprocess(self):
        """Test Preprocessing"""
        self.data = self.processor.load_and_clean_data()
        self.assertIsNotNone(self.data, msg="data not ready.")


if __name__ == "__main__":
    unittest.main()
