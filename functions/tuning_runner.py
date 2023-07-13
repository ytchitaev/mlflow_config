import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class TuningMethod(ABC):
    @abstractmethod
    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass


class GridSearchMethod(TuningMethod):
    def __init__(self, param_grid: Dict[str, Any], cv: int):
        self.param_grid = param_grid
        self.cv = cv

    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        grid_search = GridSearchCV(model, self.param_grid, cv=self.cv)
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_, grid_search.cv_results_


class RandomSearchMethod(TuningMethod):
    def __init__(self, param_dist: Dict[str, Any], n_iter: int, cv: int):
        self.param_dist = param_dist
        self.n_iter = n_iter
        self.cv = cv

    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        random_search = RandomizedSearchCV(model, self.param_dist, n_iter=self.n_iter, cv=self.cv)
        random_search.fit(X_train, y_train)
        return random_search.best_params_, random_search.cv_results_


class TuningRunner:
    def __init__(self, tuning_name: str, tuning_params: Dict[str, Any]):
        self.tuning_name = tuning_name
        self.tuning_params = tuning_params

    def run_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tuning_method = self.get_tuning_method()
        return tuning_method.perform_tuning(model, X_train, y_train)

    def get_tuning_method(self) -> TuningMethod:
        if self.tuning_name == 'grid_search':
            return GridSearchMethod(**self.tuning_params)
        elif self.tuning_name == 'random_search':
            return RandomSearchMethod(**self.tuning_params)
        else:
            raise ValueError(f"Invalid tuning method: {self.tuning_name}")
