from enum import Enum, auto
from sklearn import metrics
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from math import sqrt
from typing import Any, List
import pandas as pd
import numpy as np


class AverageTypes(Enum):
    MICRO = 'micro'
    MACRO = 'macro'
    WEIGHTED = 'weighted'
    NONE = auto()


@dataclass
class MetricCalculator:
    "class for defining metric calculations"
    model: any  # Type hint for the model object

    def calculate_metric(self, X, y, metric, params=None, per_data_point=False):
        metric_func_map = {
            'accuracy': metrics.accuracy_score,
            'precision': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, **params),
            'recall': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, **params),
            'f1': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, **params),
            'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred))
        }
        metric_func = metric_func_map.get(metric)
        if metric_func is None:
            raise ValueError(f"Unsupported metric: {metric}")

        if per_data_point:
            return [metric_func([y_true], [y_pred]) for y_true, y_pred in zip(y, self.model.predict(X))]
        else:
            return metric_func(y, self.model.predict(X))


@dataclass
class MetricFactory:
    "class for implementing metric calculations to validation and test datasets"
    model: Any
    X_data: Any
    y_data: Any
    subset: str

    def create_metrics(self, config):
        metrics_dict = {}
        data_point_metrics_dict = {}
        calculator = MetricCalculator(self.model)

        # Check if config is empty, if yes, return empty dictionaries
        if not config:
            return metrics_dict, data_point_metrics_dict

        for metric, metric_config in config.items():
            params = metric_config.get('params', {})
            
            # Summary metric
            metric_name = f"{self.subset}_{metric}"
            metric_value = calculator.calculate_metric(self.X_data, self.y_data, metric, params)
            metrics_dict[metric_name] = metric_value

            # Individual data point metrics
            data_point_metrics = {
                f"{metric}_{i}": metric_value
                for i, metric_value in enumerate(
                    calculator.calculate_metric(self.X_data, self.y_data, metric, params, per_data_point=True))
            }
            data_point_metrics_dict.update(data_point_metrics)

        return metrics_dict, data_point_metrics_dict

    def create_prediction_dataframe(self):
        # Assuming X_data and y_data have the same columns
        columns = list(self.X_data.columns)
        # Create the dataframe
        df = pd.concat([self.X_data, pd.DataFrame(
            {'output_actual': self.y_data, 'output_pred': self.model.predict(self.X_data)})], axis=1)
        return df
