import unittest
import lightgbm as lgb
from functions.model_creator import create_model, MODELS

class TestModelCreation(unittest.TestCase):
    def test_create_model_with_supported_library_and_model(self):
        cfg_model = {'library_name': 'lightgbm', 'model_name': 'LGBMClassifier'}
        params = {'param1': 'value1', 'param2': 'value2'}
        model = create_model(cfg_model, params)
        self.assertIsInstance(model, lgb.LGBMClassifier)
        self.assertEqual(model.param1, 'value1')
        self.assertEqual(model.param2, 'value2')

    def test_create_model_with_unsupported_library_name(self):
        cfg_model = {'library_name': 'invalid_library', 'model_name': 'LGBMClassifier'}
        params = {'param1': 'value1', 'param2': 'value2'}
        with self.assertRaises(ValueError):
            create_model(cfg_model, params)

    def test_create_model_with_unsupported_model_name(self):
        cfg_model = {'library_name': 'lightgbm', 'model_name': 'InvalidModel'}
        params = {'param1': 'value1', 'param2': 'value2'}
        with self.assertRaises(ValueError):
            create_model(cfg_model, params)

    def test_model_creation_in_MODELS_dictionary(self):
        self.assertIn(('lightgbm', 'LGBMClassifier'), MODELS)
        self.assertIn(('lightgbm', 'LGBMRegressor'), MODELS)
        self.assertIn(('lightgbm', 'LGBMRanker'), MODELS)
        # Add more assertions for other models in the dictionary...

if __name__ == '__main__':
    unittest.main()
