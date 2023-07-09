# Objectives
- Separation of code from parameterisation of model, all arguments, paths etc from files.
- Parameterisation should be flexible enough to accept unanticipated arguments where possible.
- Parameterisation from file allows all parameters to be tracked by mlflow
- Show example of how to integrate with databricks, not just run it locally

# To do
- verbose_fit_flag is not used
- grid search is only one example of hyperparamter tuning, maybe move in tuning_runner 
- data_loader doesn't have handling for how to select features columns and prediction column for non sklearn
- evaluate model component could be driven by metadata json, mlflow.log_metric could be dynamic based on evaluate model variables created (or also from metadata listing of evaluate for validate and test)
- Could potentially convert to classes, ask cgpt for suggestions on this

# Notes
- mlflow ui to launch

# Suggestions

If all test and validation accuracies are 1 (or close to 1) in MLflow when you run your code, it suggests that there might be an issue with your evaluation metrics or data splitting process. Here are a few potential reasons for this behavior:

1. Data Leakage: Data leakage occurs when information from the test/validation set is unintentionally used during the model training process. It can lead to over-optimistic performance metrics. Make sure that your data splitting process is properly isolating the test/validation sets from the training set.
2. Incorrect Metric Calculation: Verify that your evaluation metrics, such as accuracy, precision, recall, or F1-score, are calculated correctly. Ensure that you're using the appropriate metrics and passing the correct arguments to the calculation functions.
3. Overfitting: If your model is overfitting, it can perform extremely well on the training data but generalize poorly to unseen data. Check if your model is exhibiting signs of overfitting, such as high training accuracy and low validation/test accuracy.
4. Incorrect Model Usage: Double-check that you're using the trained model correctly when evaluating on the validation and test datasets. Ensure that the input data is preprocessed consistently and matches the expectations of the model.
5. Small or Imbalanced Dataset: If your dataset is small or imbalanced, it can result in inflated accuracy values. Consider the size and balance of your dataset and explore other evaluation metrics, such as precision, recall, or F1-score, that provide a more comprehensive view of model performance.
6. Bug in Code: Review your code for any potential bugs or logical errors. Make sure that you're using the correct data for evaluation and that the correct predictions are being compared to the true labels.
By carefully reviewing these aspects and debugging your code, you should be able to identify and resolve the issue causing all test and validation accuracies to be 1 in MLflow.

