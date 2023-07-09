from sklearn import metrics
from enum import Enum, auto
from sklearn.metrics import mean_squared_error
from math import sqrt

class AverageTypes(Enum):
    MICRO = 'micro'
    MACRO = 'macro'
    WEIGHTED = 'weighted'
    NONE = auto()


def calculate_metric(model, X_validation, y_validation, X_test, y_test, metric_name, average: AverageTypes = 'macro'):
    if metric_name == 'accuracy':
        metric_func = metrics.accuracy_score
    elif metric_name == 'precision':
        def metric_func(y_true, y_pred): return metrics.precision_score(
            y_true, y_pred, average=average)
    elif metric_name == 'recall':
        def metric_func(y_true, y_pred): return metrics.recall_score(
            y_true, y_pred, average=average)
    elif metric_name == 'f1':
        def metric_func(y_true, y_pred): return metrics.f1_score(
            y_true, y_pred, average=average)
    elif metric_name == 'rmse':
        def metric_func(y_true, y_pred): return sqrt(mean_squared_error(
            y_true, y_pred))  
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    validation_metric = metric_func(y_validation, model.predict(X_validation))
    test_metric = metric_func(y_test, model.predict(X_test))

    return validation_metric, test_metric
