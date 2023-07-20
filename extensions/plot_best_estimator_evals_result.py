import matplotlib.pyplot as plt
import pandas as pd
from typing import Any

from utils.file_processor import load_json, get_relative_path
from utils.config_loader import get_config, is_list_item_in_dict
from functions.extension_runner import ExtensionImplementation


class ExtensionBestEstimatorEvalResult(ExtensionImplementation):
    """Extension implementation for plotting best estimator evaluation results"""

    def check_extension_viability(self):
        return is_list_item_in_dict('best_estimator_evals_result.json', self.cfg['artifacts'])

    def load_extension(self) -> pd.DataFrame:
        return load_json(get_relative_path(get_config(self.cfg, 'execution.artifact_path'), "best_estimator_evals_result.json"))

    def plot_extension(self, data: Any):
        plt.figure(figsize=(10, 6))
        for metric_name, scores in data["validation"].items():
            x = range(len(scores))
            plt.plot(x, scores, label=metric_name)
        plt.xlabel('Hyperparameter Combination')
        plt.ylabel('Metric Score')
        plt.title('Evaluation Results')
        plt.legend()
        return plt
