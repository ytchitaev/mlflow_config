import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
from typing import Any

from utils.config_loader import get_config
from functions.extension_runner import ExtensionImplementation


class ExtensionLGBMTree(ExtensionImplementation):
    """Extension implementation for plotting LightGBM tree"""

    def check_extension_viability(self) -> bool:
        return get_config(self.cfg, 'model.library_name') == 'lightgbm'

    def load_extension(self) -> Any:
        model_path = get_config(self.cfg, 'execution.model_path')
        return joblib.load(model_path)

    def plot_extension(self, data: Any) -> plt.figure:
        ax = lgb.plot_tree(data, tree_index=0, figsize=(
            15, 10), show_info=['split_gain'])
        return plt.gcf()
