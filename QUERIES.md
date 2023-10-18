Provide an example lightgbm regression implementation with real data using mlflow. The solution should split train, validation, test as 70, 15,15 and use grid search hyperparameter tuning. Model tuning, evaluation and final model should be tracked using mlflow autolog.

Provide an example pytorch lightning regression implementation with real data using mlflow. The solution should split train, validation, test as 70, 15,15 and use grid search hyperparameter tuning. Model tuning, evaluation and final model should be tracked using mlflow autolog.


Provide an example pytorch lightning regression implementation with real data using mlflow with autolog. The solution should split train, validation, test as 70, 15,15 and use grid search hyperparameter tuning.

For both lightgbm and pytorch lightning examples, is there any reason to scope the mlflow run to include the grid search step?

Considerations:
    - How much to store in json cfg vs code directly in python?
    -