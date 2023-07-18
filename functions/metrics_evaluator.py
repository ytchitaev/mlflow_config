from sklearn import metrics
from enum import Enum, auto
from sklearn.metrics import mean_squared_error
from math import sqrt


class AverageTypes(Enum):
    MICRO = 'micro'
    MACRO = 'macro'
    WEIGHTED = 'weighted'
    NONE = auto()


class MetricCalculator:
    "class for defining metric calculatinos"

    def __init__(self, model):
        self.model = model

    def calculate_metric(self, X, y, metric, additional=None):
        metric_func_map = {
            'accuracy': metrics.accuracy_score,
            'precision': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, **additional),
            'recall': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, **additional),
            'f1': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, **additional),
            'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred))
        }
        metric_func = metric_func_map.get(metric)
        if metric_func is None:
            raise ValueError(f"Unsupported metric: {metric}")
        metric_value = metric_func(y, self.model.predict(X))
        return metric_value

class MetricFactory:
    "class for implementing metric calculations to validation and test datasets"

    def __init__(self, model):
        self.model = model

    def create_metrics(self, config, X_validation, X_test, y_validation, y_test):
        metrics_dict = {}

        calculator = MetricCalculator(self.model)

        for metric, metric_config in config.items():
            additional = metric_config.get('additional', {})

            metric_name_validation = f"validation_{metric}"
            metric_value_validation = calculator.calculate_metric(X_validation, y_validation, metric, additional)
            metrics_dict[metric_name_validation] = metric_value_validation

            metric_name_test = f"test_{metric}"
            metric_value_test = calculator.calculate_metric(X_test, y_test, metric, additional)
            metrics_dict[metric_name_test] = metric_value_test

        return metrics_dict