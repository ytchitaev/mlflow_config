
import mlflow

def log_evaluation_metrics(configs):
    for metric in configs['evaluate']:
        for split in configs['evaluate'][metric]['split']:
            metric_value = globals().get(f"{split}_{metric}")
            if metric_value is not None:
                mlflow.log_metric(f"{split}_{metric}", metric_value)
            else:
                raise ValueError(f"Variable '{split}_{metric}' does not exist")
