import unittest
from unittest.mock import MagicMock
from functions.model_runner import LightGBMLibraryImplementer, fit_model, log_model


class TestModelRunner(unittest.TestCase):
    def test_lightgbm_library_implementer_fit_model(self):
        # Create a mock lightgbm.Booster object and data
        model = MagicMock()
        X_train = MagicMock()
        y_train = MagicMock()

        # Create a mock cfg_model with the 'callbacks' key
        cfg_model = {"library_name": "lightgbm", "callbacks": {}}

        # Create an instance of LightGBMLibraryImplementer
        implementer = LightGBMLibraryImplementer()

        # Mock the handle_callbacks method
        implementer.handle_callbacks = MagicMock(return_value=[])

        # Assert that the necessary methods are called with the correct arguments
        model.fit = MagicMock()
        implementer.fit_model(cfg_model, model, X_train, y_train)
        model.fit.assert_called_with(X_train, y_train, callbacks=[])

if __name__ == '__main__':
    unittest.main()
