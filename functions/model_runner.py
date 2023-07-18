from abc import ABC, abstractmethod
from typing import Dict, Any
import lightgbm as lgb
import mlflow.lightgbm

##### Implement model creation

class ModelImplementer(ABC):
    @abstractmethod
    def create(self, params: Dict[str, Any]) -> Any:
        pass


class LGBMClassifierModel(ModelImplementer):
    def create(self, params: Dict[str, Any]) -> Any:
        return lgb.LGBMClassifier(**params)


class LGBMRegressorModel(ModelImplementer):
    def create(self, params: Dict[str, Any]) -> Any:
        return lgb.LGBMRegressor(**params)


class LGBMRankerModel(ModelImplementer):
    def create(self, params: Dict[str, Any]) -> Any:
        return lgb.LGBMRanker(**params)


MODELS = {
    ("lightgbm", "LGBMClassifier"): LGBMClassifierModel(),
    ("lightgbm", "LGBMRegressor"): LGBMRegressorModel(),
    ("lightgbm", "LGBMRanker"): LGBMRankerModel(),
    # Add more models as needed...
}

# interface

def create_model(cfg_model: dict, params: Dict[str, Any]) -> Any:
    library_name, model_name = cfg_model['library_name'], cfg_model['model_name']
    model_creator = MODELS.get((library_name, model_name))
    if model_creator is None:
        raise ValueError(
            f"Unsupported library name: {library_name} - model name: {model_name}")
    return model_creator.create(params)


##### Implement model fit and logging

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
        callbacks = [lgb.log_evaluation(period=100, show_stdv=True)]
        if isinstance(model, lgb.Booster):
            callbacks.append(mlflow.lightgbm.callbacks.LGBMLogger())
        model.fit(X_train, y_train, callbacks=callbacks)

# interface

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