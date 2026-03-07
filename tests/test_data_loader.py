import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader()
    
    def test_initialization(self):
        self.assertIsNotNone(self.loader)
    
    def test_load_yahoo(self):
        data = self.loader.load_from_yahoo('AAPL', '2023-01-01', '2023-12-31')
        if data is not None:
            self.assertGreater(len(data), 0)

if __name__ == '__main__':
    unittest.main()
