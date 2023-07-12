from sklearn import metrics
from enum import Enum, auto
from sklearn.metrics import mean_squared_error
from math import sqrt


class AverageTypes(Enum):
    MICRO = 'micro'
    MACRO = 'macro'
    WEIGHTED = 'weighted'
    NONE = auto()


def calculate_metric(model, X_data, y_data, metric_name, average: AverageTypes = AverageTypes.MACRO):
    metric_func_map = {
        'accuracy': metrics.accuracy_score,
        'precision': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average=average.value),
        'recall': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average=average.value),
        'f1': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average=average.value),
        'rmse': lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred))
    }
    metric_func = metric_func_map.get(metric_name)
    if metric_func is None:
        raise ValueError(f"Unsupported metric: {metric_name}")
    metric = metric_func(y_data, model.predict(X_data))
    return metric
