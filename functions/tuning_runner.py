import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class TuningMethod(ABC):
    @abstractmethod
    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass


class GridSearchMethod(TuningMethod):
    def __init__(self, param_grid: Dict[str, Any], cv: int, scoring: str, n_jobs: int, refit: bool, return_train_score: bool, verbose: bool):
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.return_train_score = return_train_score
        self.verbose = verbose

    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        grid_search = GridSearchCV(model, self.param_grid, cv=self.cv, scoring=self.scoring,
                                   n_jobs=self.n_jobs, refit=self.refit, return_train_score=self.return_train_score, verbose=self.verbose)
        grid_search.fit(X=X_train, y=y_train, eval_set=(
            X_validation, y_validation), eval_names='validation')
        best_estimator = grid_search.best_estimator_ if self.refit else None # return conditional on refit = True
        return grid_search.best_params_, grid_search.cv_results_, best_estimator


class RandomSearchMethod(TuningMethod):
    def __init__(self, param_dist: Dict[str, Any], n_iter: int, cv: int, refit: bool):
        self.param_dist = param_dist
        self.n_iter = n_iter
        self.cv = cv
        self.refit = refit

    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        random_search = RandomizedSearchCV(
            model, self.param_dist, n_iter=self.n_iter, cv=self.cv, refit=self.refit)
        random_search.fit(X=X_train, y=y_train, eval_set=(
            X_validation, y_validation), eval_names='validation')
        best_estimator = random_search.best_estimator_ if self.refit else None  # return conditional on refit = True
        return random_search.best_params_, random_search.cv_results_, best_estimator


class TuningRunner:
    def __init__(self, tuning_name: str, tuning_params: Dict[str, Any]):
        self.tuning_name = tuning_name
        self.tuning_params = tuning_params

    def run_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tuning_method = self.get_tuning_method()
        return tuning_method.perform_tuning(model, X_train, y_train, X_validation, y_validation)

    def get_tuning_method(self) -> TuningMethod:
        if self.tuning_name == 'grid_search':
            return GridSearchMethod(**self.tuning_params)
        elif self.tuning_name == 'random_search':
            return RandomSearchMethod(**self.tuning_params)
        else:
            raise ValueError(f"Invalid tuning method: {self.tuning_name}")
