import matplotlib.pyplot as plt
import pandas as pd
from typing import Any

from utils.file_processor import load_csv, get_relative_path
from utils.config_loader import get_config, is_list_item_in_dict
from functions.extension_runner import ExtensionImplementation


class ExtensionCVResults(ExtensionImplementation):
    """Extension implementation for plotting cross-validation results"""

    def check_extension_viability(self):
        return is_list_item_in_dict('cv_results.csv', self.cfg['artifacts'])

    def load_extension(self) -> pd.DataFrame:
        return load_csv(get_relative_path(get_config(self.cfg, 'execution.artifact_path'), "cv_results.csv"))

    def plot_extension(self, data: Any):
        mean_train_scores = data['mean_train_score']
        mean_validation_scores = data['mean_test_score']
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(mean_train_scores)),
                 mean_train_scores, label='Train')
        plt.plot(range(len(mean_validation_scores)),
                 mean_validation_scores, label='Validation')
        plt.xlabel('Hyperparameter Combination')
        plt.ylabel('Metric Score')
        plt.title('Evaluation Results')
        plt.legend()
        return plt
