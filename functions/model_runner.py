from abc import ABC, abstractmethod
from typing import Dict, Any
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker

class Model(ABC):
    @abstractmethod
    def create(self, params: Dict[str, Any]) -> Any:
        pass

class LGBMClassifierModel(Model):
    def create(self, params: Dict[str, Any]) -> Any:
        return LGBMClassifier(**params)

class LGBMRegressorModel(Model):
    def create(self, params: Dict[str, Any]) -> Any:
        return LGBMRegressor(**params)

class LGBMRankerModel(Model):
    def create(self, params: Dict[str, Any]) -> Any:
        return LGBMRanker(**params)

MODELS = {
    ("lightgbm", "LGBMClassifier"): LGBMClassifierModel(),
    ("lightgbm", "LGBMRegressor"): LGBMRegressorModel(),
    ("lightgbm", "LGBMRanker"): LGBMRankerModel(),
    # Add more models as needed...
}

def create_model(library_name: str, model_name: str, params: Dict[str, Any]) -> Any:
    model_creator = MODELS.get((library_name, model_name))
    if model_creator is None:
        raise ValueError(f"Unsupported library name: {library_name} - model name: {model_name}")
    return model_creator.create(params)

## usage example
#def main():
#    factory = ModelFactory()
#    factory.register_model('LGBMClassifier', LGBMClassifierModel())
#    factory.register_model('LGBMRegressor', LGBMRegressorModel())
#    factory.register_model('LGBMRanker', LGBMRankerModel())
#
#if __name__ == "__main__":
#    main()