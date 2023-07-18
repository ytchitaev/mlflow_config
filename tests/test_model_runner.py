import unittest
from unittest.mock import MagicMock
from functions.model_runner import fit_model, log_model, LightGBMLibraryImplementer
import lightgbm


class TestModelRunner(unittest.TestCase):
    def test_fit_model_with_lightgbm_library(self):
        cfg_model = {'library_name': 'lightgbm', 'callbacks': {}}  # Add 'callbacks' key with an empty dictionary
        model = MagicMock(spec=lightgbm.Booster)
        X_train = MagicMock()
        y_train = MagicMock()

        # Test that the LightGBM model implementer's fit_model method is called
        lightgbm_model_implementer = LightGBMLibraryImplementer()
        lightgbm_model_implementer.fit_model = MagicMock()
        fit_model(cfg_model, model, X_train, y_train)
        lightgbm_model_implementer.fit_model.assert_called_once_with(cfg_model, model, X_train, y_train)

    def test_log_model_with_lightgbm_library(self):
        cfg_model = {'library_name': 'lightgbm'}
        model = MagicMock()
        artifact_path = "path/to/artifact"

        # Test that the LightGBM model implementer's log_model method is called
        lightgbm_model_implementer = LightGBMLibraryImplementer()
        lightgbm_model_implementer.log_model = MagicMock(return_value="logged_model")
        logged_model = log_model(cfg_model, model, artifact_path)
        lightgbm_model_implementer.log_model.assert_called_once_with(model, artifact_path)
        self.assertEqual(logged_model, "logged_model")

    # Add more test cases for other scenarios and libraries...


if __name__ == '__main__':
    unittest.main()
