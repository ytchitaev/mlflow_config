import unittest
from functions.model_runner import create_model
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker

class TestCreateModel(unittest.TestCase):
    def test_create_model_with_valid_params(self):
        model = create_model({'library_name': 'lightgbm', 'model_name': 'LGBMClassifier'}, {})
        self.assertIsInstance(model, LGBMClassifier)

        model = create_model({'library_name': 'lightgbm', 'model_name': 'LGBMRegressor'}, {})
        self.assertIsInstance(model, LGBMRegressor)

        model = create_model({'library_name': 'lightgbm', 'model_name': 'LGBMRanker'}, {})
        self.assertIsInstance(model, LGBMRanker)

    def test_create_model_with_invalid_library_name(self):
        with self.assertRaises(ValueError):
            create_model({'library_name': 'invalid_library', 'model_name': 'LGBMClassifier'}, {})

    def test_create_model_with_invalid_model_name(self):
        with self.assertRaises(ValueError):
            create_model({'library_name': 'lightgbm', 'model_name': 'invalid_model'}, {})

if __name__ == '__main__':
    unittest.main()
