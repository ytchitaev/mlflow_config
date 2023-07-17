
import matplotlib.pyplot as plt
from utils.config_loader import get_config
from utils.file_processor import get_relative_path, load_json

class ExtensionFactory:
    @staticmethod
    def create_extension(cfg: dict, ext: str, debug: bool):
        if ext == 'plot_lgbm_tree':
            return ExtensionType.plot_lgbm_tree(cfg, debug)
        elif ext == 'plot_best_estimator_evals_result':
            return ExtensionType.plot_best_estimator_evals_result(cfg, debug)
        elif ext == 'plot_cv_results':
            return ExtensionType.plot_cv_results(cfg, debug)
        else:
            raise ValueError(f"Unknown extension: {ext}")


class ExtensionType:

    @staticmethod
    def plot_lgbm_tree(cfg: dict, debug: bool):
        # Implement the logic for plot_lgbm_tree extension
        pass

    @staticmethod
    def plot_best_estimator_evals_result(cfg: dict):
        best_estimator_evals_result = load_json(get_relative_path(get_config(cfg, 'execution.artifact_path'), "best_estimator_evals_result.json"))
        plt.figure(figsize=(10, 6))
        for metric_name, scores in best_estimator_evals_result["validation"].items():
            x = range(len(scores))
            plt.plot(x, scores, label=metric_name)
        plt.xlabel('Hyperparameter Combination')
        plt.ylabel('Metric Score')
        plt.title('Evaluation Results')
        plt.legend()
        return plt

    @staticmethod
    def plot_cv_results(cfg: dict, debug: bool):
        # Implement the logic for plot_cv_results extension
        pass
