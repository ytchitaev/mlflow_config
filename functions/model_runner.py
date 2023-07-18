from abc import ABC, abstractmethod
from typing import Any
import lightgbm
import mlflow.lightgbm


class LibraryImplementer(ABC):
    @abstractmethod
    def log_model(self, model: Any, artifact_path: str) -> Any:
        pass

    @abstractmethod
    def fit_model(self, model: Any, X_train: Any, y_train: Any) -> Any:
        pass


class LightGBMLibraryImplementer(LibraryImplementer):
    def log_model(self, model: Any, artifact_path: str) -> Any:
        return mlflow.lightgbm.log_model(model, artifact_path)

    def fit_model(self, model: Any, X_train: Any, y_train: Any) -> Any:
        callbacks = [lightgbm.log_evaluation(period=100, show_stdv=True)]
        if isinstance(model, lightgbm.Booster):
            callbacks.append(mlflow.lightgbm.callbacks.LGBMLogger())
        model.fit(X_train, y_train, callbacks=callbacks)


def fit_model(cfg_model: dict, model, X_train, y_train):
    library_name = cfg_model['library_name']
    if library_name == 'lightgbm':
        lightgbm_model_implementer = LightGBMLibraryImplementer()
        lightgbm_model_implementer.fit_model(model, X_train, y_train)
    else:
        # Add implementation for other libraries
        pass


def log_model(cfg_model: dict, model, artifact_path: str):
    library_name = cfg_model['library_name']
    if library_name == 'lightgbm':
        lightgbm_model_implementer = LightGBMLibraryImplementer()
        return lightgbm_model_implementer.log_model(model, artifact_path)
    else:
        # Add implementation for other libraries
        pass
