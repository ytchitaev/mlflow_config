from abc import ABC, abstractmethod
from typing import Any, List
import lightgbm
import mlflow.lightgbm


class LibraryImplementer(ABC):
    "abstract base class definition for model fit and model logging"

    @abstractmethod
    def fit_model(self, cfg_model: dict, model: Any, X_train: Any, y_train: Any) -> Any:
        pass

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str) -> Any:
        pass


class LightGBMLibraryImplementer(LibraryImplementer):

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

    def log_model(self, model: Any, artifact_path: str) -> Any:
        "lightgbm implementation of model logging"
        return mlflow.lightgbm.log_model(model, artifact_path)


class ModelManager:
    @staticmethod
    def fit_model(cfg_model: dict, model, X_train, y_train):
        library_name = cfg_model['library_name']
        library_implementer = ModelManager._create_library_implementer(
            library_name)
        library_implementer.fit_model(cfg_model, model, X_train, y_train)

    @staticmethod
    def log_model(cfg_model: dict, model, artifact_path: str):
        library_name = cfg_model['library_name']
        library_implementer = ModelManager._create_library_implementer(
            library_name)
        return library_implementer.log_model(model, artifact_path)

    @staticmethod
    def _create_library_implementer(library_name: str) -> LibraryImplementer:
        if library_name == 'lightgbm':
            return LightGBMLibraryImplementer()
        else:
            # Add implementation for other libraries
            pass
