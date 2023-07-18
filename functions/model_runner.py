from abc import ABC, abstractmethod
from typing import Any, List
import lightgbm
import mlflow.lightgbm


class LibraryImplementer(ABC):
    "abstract base class definition for model fit and model logging"
    @abstractmethod
    def log_model(self, model: Any, artifact_path: str) -> Any:
        pass

    @abstractmethod
    def fit_model(self, cfg_model: dict, model: Any, X_train: Any, y_train: Any) -> Any:
        pass


class LightGBMLibraryImplementer(LibraryImplementer):
    def log_model(self, model: Any, artifact_path: str) -> Any:
        "lightgbm implementation of model logging"
        return mlflow.lightgbm.log_model(model, artifact_path)

    def handle_callbacks(self, cfg_model: dict) -> List:
        "dynamically handle callbacks e.g. lightgbm.log_evaluation(period=100, show_stdv=True)"
        callbacks = []
        callbacks_config = cfg_model["callbacks"]
        for callback_name, callback_args in callbacks_config.items():
            callback_module = __import__("lightgbm", fromlist=[callback_name])
            callback = getattr(callback_module, callback_name)
            callback_obj = callback(**callback_args)
            callbacks.append(callback_obj)
        return callbacks

    def fit_model(self, cfg_model: dict, model: Any, X_train: Any, y_train: Any) -> Any:
        "lightgbm implementation of model fit"
        callbacks = self.handle_callbacks(cfg_model)
        if isinstance(model, lightgbm.Booster):
            callbacks.append(mlflow.lightgbm.callbacks.LGBMLogger())
        model.fit(X_train, y_train, callbacks=callbacks)


def fit_model(cfg_model: dict, model, X_train, y_train):
    "generic interface for fitting a model"
    library_name = cfg_model['library_name']
    if library_name == 'lightgbm':
        lightgbm_model_implementer = LightGBMLibraryImplementer()
        lightgbm_model_implementer.fit_model(cfg_model, model, X_train, y_train)
    else:
        # Add implementation for other libraries
        pass


def log_model(cfg_model: dict, model, artifact_path: str):
    "generic interface for logging model"
    library_name = cfg_model['library_name']
    if library_name == 'lightgbm':
        lightgbm_model_implementer = LightGBMLibraryImplementer()
        return lightgbm_model_implementer.log_model(model, artifact_path)
    else:
        # Add implementation for other libraries
        pass
