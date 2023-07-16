import matplotlib.pyplot as plt

def plot_best_estimator_evals_result(data):
    plt.figure(figsize=(10, 6))
    for metric_name, scores in data["validation"].items():
        x = range(len(scores))
        plt.plot(x, scores, label=metric_name)
    plt.xlabel('Hyperparameter Combination')
    plt.ylabel('Metric Score')
    plt.title('Evaluation Results')
    plt.legend()
    return plt