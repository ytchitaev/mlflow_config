import os
import argparse
import traceback

import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from functions.config_mapping import MAPPING_SPECIFICATION

from functions.config_loader import load_configurations
from functions.mlflow_logger import instantiate_python_logger, mlflow_log_artifact_cv_results
from functions.data_loader import load_data
from functions.data_splitter import split_dataset
from functions.model_runner import create_model
from functions.evaluation_runner import perform_grid_search
from functions.evaluation_metrics import calculate_metric


def main(config_file):

    # Parse CLI args
    configs = load_configurations(config_file, MAPPING_SPECIFICATION)
    # Define experiment name
    mlflow.set_experiment(configs['setup.experiment_name'])
    # Start an MLflow run

    with mlflow.start_run() as run:

        try:
            # Get experiment / run details and instantiate python logger
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            experiment_run_path = os.path.join("mlruns", experiment_id, run_id)
            model_path = os.path.join("mlruns", experiment_id, run_id, "artifacts/model/model.pkl")
            logger, file_handler_path = instantiate_python_logger(
                experiment_run_path,
                configs['setup.paths.run_temp_folder'],
                configs['setup.artefacts.python_logging.file_name']
            )
            logger.info("Running mlflow...")
            logger.info(f"Started run: {experiment_run_path}")

            # Log the config parameters
            mlflow.log_params(configs['setup'])
            mlflow.log_params(configs['grid_search'])

            # Load data
            logger.info("Loading data...")
            X_input, y_input = load_data(
                data_source=configs['setup.data_source'], 
                dataset_name=configs['setup.dataset_name'],
                input_columns=configs['setup.input_columns'],
                output_columns=configs['setup.output_columns']
            )
            # Split data
            logger.info("Splitting data...")
            X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(
                X_input, y_input,
                configs['setup.split.train_percentage'],
                configs['setup.split.validation_percentage'],
                configs['setup.split.test_percentage'])

            # Instantiate model
            logger.info("Loading initial model...")
            model = create_model(
                configs['setup.library_name'], configs['setup.model_name'], params={})

            # Run grid search
            logger.info("Running grid search...")
            grid_search = perform_grid_search(
                model, 
                X_train, 
                y_train, 
                configs['grid_search.param_grid'], 
                configs['grid_search.cv']
            )

            # Get best model values from grid search
            best_params = grid_search.best_params_
            cv_results = grid_search.cv_results_

            # Train the model with the best parameters
            logger.info("Training model with best parameters from grid search...")
            best_model = create_model(
                configs['setup.library_name'], 
                configs['setup.model_name'], 
                best_params
            )
            
            best_model.fit(
                X_train,
                y_train,
                callbacks=[lgb.log_evaluation(period=100, show_stdv=True)]
            )

            # Evaluate model
            logger.info("Evaluating model...")
            #validation_accuracy, test_accuracy = calculate_metric(best_model, X_validation, y_validation, X_test, y_test, 'accuracy')
            #validation_precision, test_precision = calculate_metric(best_model, X_validation, y_validation, X_test, y_test, 'precision', average='macro')
            #validation_recall, test_recall = calculate_metric(best_model, X_validation, y_validation, X_test, y_test, 'recall', average='macro')
            #validation_f1, test_f1 = calculate_metric(best_model, X_validation, y_validation, X_test, y_test, 'f1', average='macro')
            validation_rmse, test_rmse = calculate_metric(best_model, X_validation, y_validation, X_test, y_test, 'rmse')

            # Log model, column names, parameters and artifacts
            logger.info("Logging model, parameters and artifacts...")
            mlflow.lightgbm.log_model(best_model, "model")
            mlflow.log_param("input_columns", configs['setup.input_columns'])
            mlflow.log_param("output_columns", configs['setup.output_columns'])
            mlflow.log_params(best_params)
            mlflow_log_artifact_cv_results(
                experiment_id, 
                run_id,
                configs['setup.paths.run_temp_folder'],
                configs['setup.artefacts.cv_results.file_name'],
                cv_results
            )

            # Log evaluations
            logger.info("Logging evaluations...")
            #mlflow.log_metric("validation_accuracy", validation_accuracy)
            #mlflow.log_metric("test_accuracy", test_accuracy)
            #mlflow.log_metric("validation_precision", validation_precision)
            #mlflow.log_metric("test_precision", test_precision)
            #mlflow.log_metric("validation_recall", validation_recall)
            #mlflow.log_metric("test_recall", test_recall)
            #mlflow.log_metric("validation_f1", validation_f1)
            #mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("validation_rmse", validation_rmse)
            mlflow.log_metric("test_rmse", test_rmse)

        except Exception as e:
            logger.error(f"Exception occurred during model training: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            mlflow.log_param("status", "FAILED")
            mlflow.log_artifact(file_handler_path)
            logger.info(
                f"Finished run: {experiment_run_path}")

        else:
            mlflow.log_param("status", "SUCCESS")
            mlflow.log_artifact(file_handler_path)
            logger.info("Model training completed successfully.")
            logger.info(f"Finished run: {experiment_run_path}")
            logger.info(f"Input columns: {', '.join(configs['setup.input_columns'])}")
            logger.info(f"Output columns: {', '.join(configs['setup.output_columns'])}")
            logger.info(f"Model - full path:{model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specify the configuration file.")
    parser.add_argument('-c', '--config', type=str,
                        default='config_iris.json', help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)
