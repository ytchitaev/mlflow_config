import matplotlib.pyplot as plt
import pandas as pd

def plot_cv_results(cv_results: pd.DataFrame):
    mean_train_scores = cv_results['mean_train_score']
    mean_validation_scores = cv_results['mean_test_score']
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mean_train_scores)), mean_train_scores, label='Train')
    plt.plot(range(len(mean_validation_scores)), mean_validation_scores, label='Validation')
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Metric Score')
    plt.title('Evaluation Results')
    plt.legend()
    return plt