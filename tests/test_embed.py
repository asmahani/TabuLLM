import unittest
import pandas as pd
from TabuLLM.embed import TextColumnTransformer

class TestTextColumnTransformer(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        print("Setting up the test environment...")
        self.sample_text = ["This is a test.", "Another test sentence."]
        self.sample_df = pd.DataFrame({'text': self.sample_text})

    def test_initialization(self):
        # Test default initialization
        transformer = TextColumnTransformer()
        self.assertEqual(transformer.type, 'doc2vec')
        self.assertEqual(transformer.google_location, 'us-central1')

if __name__ == '__main__':
    unittest.main()
