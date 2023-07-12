import argparse
import traceback

import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from functions.config_mapping import GLOBAL_MAPPING, EXPERIMENT_MAPPING

from functions.run_manager import setup_run
from functions.config_loader import load_configurations
from functions.mlflow_logger import instantiate_python_logger, mlflow_log_artifact_dict_to_csv, mlflow_log_artifact_dict_to_json
from functions.data_loader import load_data
from functions.data_splitter import split_dataset
from functions.model_runner import create_model
from functions.evaluation_runner import perform_grid_search
from functions.evaluation_metrics import AverageTypes, calculate_metric
from functions.evaluation_logger import log_evaluation_metrics

def main(config_file):

    global_configs = load_configurations(config_file, GLOBAL_MAPPING)
    configs = load_configurations(config_file, EXPERIMENT_MAPPING)
    mlflow.set_experiment(configs['setup.experiment_name'])

    with mlflow.start_run() as run:

        try:
            # Instantiate logger and run
            run_id, experiment_id, experiment_run_path, model_path = setup_run(run)
            logger, file_handler_path = instantiate_python_logger(experiment_run_path, global_configs['run_temp_subdir'], global_configs['py_log_file_name'])
            logger.info(f"Started run: {experiment_run_path}")

            # Log the config as artifact
            mlflow_log_artifact_dict_to_json(experiment_id, run_id, global_configs['run_temp_subdir'], "config.json", configs[''])

            # Load data
            logger.info("Loading data...")
            X_input, y_input = load_data(**configs['data'])

            # Split data
            logger.info("Splitting data...")
            X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(X_input, y_input, **configs['split'])

            # Instantiate model
            logger.info("Loading initial model...")
            model = create_model(configs['model.library_name'], configs['model.model_name'], params={})

            # Run grid search to get best model values and grid search CV run logs
            logger.info("Running grid search...")
            best_params, cv_results = perform_grid_search(model, X_train, y_train,  **configs['tuning.grid_search'])

            # Train the model with the best parameters
            logger.info("Training model with best parameters from grid search...")
            best_model = create_model(configs['model.library_name'], configs['model.model_name'], best_params)
            best_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=100, show_stdv=True)])

            # Evaluate model
            logger.info("Evaluating model...")

            validation_accuracy = calculate_metric(best_model, X_validation, y_validation, 'accuracy')
            test_accuracy = calculate_metric(best_model, X_test, y_test, 'accuracy')

            validation_precision = calculate_metric(best_model, X_validation, y_validation, 'precision', average = AverageTypes.MACRO)
            test_precision = calculate_metric(best_model, X_test, y_test, 'precision', average = AverageTypes.MACRO)

            validation_recall = calculate_metric(best_model, X_validation, y_validation, 'recall', average = AverageTypes.MACRO)
            test_recall = calculate_metric(best_model, X_test, y_test, 'recall', average = AverageTypes.MACRO)

            validation_f1 = calculate_metric(best_model, X_validation, y_validation, 'f1', average = AverageTypes.MACRO)
            test_f1 = calculate_metric(best_model, X_test, y_test, 'f1', average = AverageTypes.MACRO)

            validation_rmse = calculate_metric(best_model, X_validation, y_validation, 'rmse')
            test_rmse = calculate_metric(best_model, X_test, y_test, 'rmse')

            # Log model, best parameters and artifacts
            logger.info("Logging model, parameters and artifacts...")
            mlflow.lightgbm.log_model(best_model, "model")
            mlflow.log_params(best_params)
            mlflow_log_artifact_dict_to_csv(experiment_id, run_id, global_configs['run_temp_subdir'], configs['artefacts.cv_results.file_name'], cv_results)

            # Log evaluations
            logger.info("Logging evaluations...")
            #log_evaluation_metrics(configs)
            mlflow.log_metric("validation_accuracy", validation_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("validation_precision", validation_precision)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("validation_recall", validation_recall)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("validation_f1", validation_f1)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("validation_rmse", validation_rmse)
            mlflow.log_metric("test_rmse", test_rmse)

        except Exception as e:
            logger.error(f"Exception occurred during model training: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            mlflow.log_param("status", "FAILED")

        else:
            logger.info(f"Input columns: {', '.join(configs['data.input_columns'])}")
            logger.info(f"Output columns: {', '.join(configs['data.output_columns'])}")
            mlflow.log_param("status", "SUCCESS")
            logger.info(f"Model training completed - full path:{model_path}")
        
        finally:
            mlflow.log_artifact(file_handler_path)
            logger.info(f"Finished run: {experiment_run_path}")            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the configuration file.")
    parser.add_argument('-c', '--config', type=str, default='config_iris.json', help='Path to the configuration file')
    args = parser.parse_args()
    main(args.config)
