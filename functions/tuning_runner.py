import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from loggers.mlflow_artifact_logger import mlflow_log_artifact_dict_to_csv, mlflow_log_artifact_dict_to_json


@dataclass
class TuningParams:
    """Class for model tuning parameter outputs"""
    best_params: Dict[str, Any]


@dataclass
class TuningArtifacts:
    """Class for model tuning parameter artifacts"""
    cv_results: Dict[str, Any]
    best_estimator_evals_result: Any

    def log_tuning_artifacts(self, cfg, logger):
        """Log tuning artifacts in mlflow, python logger, and config"""
        if self.cv_results:
            logger.info("Logging cv_results artifact...")
            mlflow_log_artifact_dict_to_csv(cfg, "cv_results.csv", self.cv_results)
            cfg = self.add_artifact_to_config(cfg, "cv_results.csv")
        if self.best_estimator_evals_result:
            logger.info("Logging best_estimator_evals_result artifact...")
            mlflow_log_artifact_dict_to_json(cfg, "best_estimator_evals_result.json", self.best_estimator_evals_result)
            cfg = self.add_artifact_to_config(cfg, "best_estimator_evals_result.json")
        return cfg

    def add_artifact_to_config(self, cfg, artifact_name):
        """Append to artifacts:{} config for use by extensions"""
        cfg.setdefault("artifacts", []).append(artifact_name)
        return cfg


class TuningMethod(ABC):
    """Abstract base class definition for tuning methods"""
    @abstractmethod
    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> Tuple[TuningParams, TuningArtifacts]:
        pass


# Implementations of different tuning methods
@dataclass
class GridSearchMethod(TuningMethod):
    param_grid: Dict[str, Any]
    cv: int
    scoring: str
    n_jobs: int
    refit: bool
    return_train_score: bool
    verbose: bool

    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        grid_search = GridSearchCV(
            model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            return_train_score=self.return_train_score,
            verbose=self.verbose
        )
        grid_search.fit(X=X_train, y=y_train, eval_set=(X_validation, y_validation), eval_names='validation')
        # Return conditional on refit = True
        best_estimator_evals_result = grid_search.best_estimator_.evals_result_ if self.refit else None
        return TuningParams(grid_search.best_params_), TuningArtifacts(grid_search.cv_results_, best_estimator_evals_result)


@dataclass
class RandomSearchMethod(TuningMethod):
    param_dist: Dict[str, Any]
    n_iter: int
    cv: int
    refit: bool
    
    def perform_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        random_search = RandomizedSearchCV(
            model,
            self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            refit=self.refit
        )
        random_search.fit(X=X_train, y=y_train, eval_set=(X_validation, y_validation), eval_names='validation')
        # Return conditional on refit = True
        best_estimator_evals_result = random_search.best_estimator_.evals_result_ if self.refit else None
        return TuningParams(random_search.best_params_), TuningArtifacts(random_search.cv_results_, best_estimator_evals_result)


class TuningRunner:
    """Interface class for calling tuning method from config"""

    def __init__(self, tuning_cfg: dict):
        self.tuning_name = tuning_cfg['name']
        self.tuning_params = tuning_cfg['params']

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
