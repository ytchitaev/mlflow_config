import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib


def plot_lightgbm_tree(model_path, tree_index=0, figsize=(15, 10)):
    model = joblib.load(model_path)
    ax = lgb.plot_tree(model, tree_index=tree_index,
                       figsize=figsize, show_info=['split_gain'])
    return plt
