import unittest
from functions.model_runner import create_model
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker

class TestCreateModel(unittest.TestCase):
    def test_create_model_with_valid_params(self):
        model = create_model("lightgbm", "LGBMClassifier", {})
        self.assertIsInstance(model, LGBMClassifier)

        model = create_model("lightgbm", "LGBMRegressor", {})
        self.assertIsInstance(model, LGBMRegressor)

        model = create_model("lightgbm", "LGBMRanker", {})
        self.assertIsInstance(model, LGBMRanker)

    def test_create_model_with_invalid_library_name(self):
        with self.assertRaises(ValueError):
            create_model("invalid_library", "LGBMClassifier", {})

    def test_create_model_with_invalid_model_name(self):
        with self.assertRaises(ValueError):
            create_model("lightgbm", "invalid_model", {})

if __name__ == '__main__':
    unittest.main()