from abc import ABC, abstractmethod
from typing import Dict, Any
import lightgbm as lgb


class ModelCreator(ABC):
    @abstractmethod
    def create(self, params: Dict[str, Any]) -> Any:
        pass


class LGBMClassifierModel(ModelCreator):
    def create(self, params: Dict[str, Any]) -> Any:
        return lgb.LGBMClassifier(**params)


class LGBMRegressorModel(ModelCreator):
    def create(self, params: Dict[str, Any]) -> Any:
        return lgb.LGBMRegressor(**params)


class LGBMRankerModel(ModelCreator):
    def create(self, params: Dict[str, Any]) -> Any:
        return lgb.LGBMRanker(**params)


MODELS = {
    ("lightgbm", "LGBMClassifier"): LGBMClassifierModel(),
    ("lightgbm", "LGBMRegressor"): LGBMRegressorModel(),
    ("lightgbm", "LGBMRanker"): LGBMRankerModel(),
    # Add more models as needed...
}


def create_model(cfg_model: dict, params: Dict[str, Any]) -> Any:
    library_name, model_name = cfg_model['library_name'], cfg_model['model_name']
    model_creator = MODELS.get((library_name, model_name))
    if model_creator is None:
        raise ValueError(
            f"Unsupported library name: {library_name} - model name: {model_name}")
    return model_creator.create(params)
