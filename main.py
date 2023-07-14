import argparse
import traceback
import copy 
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
from functions.tuning_runner import TuningRunner
from functions.metrics_evaluator import MetricFactory

def main(global_config_file_name, config_file_name):

    global_configs = load_configurations(global_config_file_name, GLOBAL_MAPPING)
    configs = load_configurations(config_file_name, EXPERIMENT_MAPPING)
    mlflow.set_experiment(configs['setup.experiment_name'])

    with mlflow.start_run() as run:

        try:

            # Create logger, define run identifiers and log config and global config
            run_id, experiment_id, experiment_run_path, model_path = setup_run(run)
            logger, file_handler_path = instantiate_python_logger(experiment_run_path, global_configs['run_temp_subdir'], global_configs['py_log_file_name'])
            mlflow_log_artifact_dict_to_json(experiment_id, run_id, global_configs['run_temp_subdir'], "global_config.json", global_configs[''])
            mlflow_log_artifact_dict_to_json(experiment_id, run_id, global_configs['run_temp_subdir'], "config.json", configs[''])
            logger.info(f"Started run: {experiment_run_path}")

            # Load data
            logger.info("Loading data...")
            X_input, y_input = load_data(**configs['data'])

            # Split data
            logger.info("Splitting data...")
            X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(X_input, y_input, **configs['split'])

            # Create model with initial params and create final params
            logger.info("Loading initial model...")
            initial_params = configs['model.params']
            final_params = copy.deepcopy(initial_params)
            model = create_model(configs['model.library_name'], configs['model.model_name'], params=initial_params)

            # Run tuning if enabled, update final params with tuned values and log final params
            if configs['tuning']:
                logger.info("Running tuning...")
                tunning_runner = TuningRunner(configs['tuning.name'], configs['tuning.params'])
                cv_params, cv_results = tunning_runner.run_tuning(model, X_train, y_train)
                final_params.update(cv_params)
                mlflow.log_params(final_params)
                mlflow_log_artifact_dict_to_csv(experiment_id, run_id, global_configs['run_temp_subdir'], configs['artefacts.cv_results.file_name'], cv_results)

            # Train the model with final params and log model
            logger.info("Training model...")
            best_model = create_model(configs['model.library_name'], configs['model.model_name'], params=final_params)
            best_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=100, show_stdv=True)])
            mlflow.lightgbm.log_model(best_model, "model")

            # Evaluate model
            logger.info("Evaluating model...")
            metrics_factory = MetricFactory(best_model)
            metrics = metrics_factory.create_metrics(configs['evaluate'], X_validation, X_test, y_validation, y_test)
            [mlflow.log_metric(metric_name, metric_value) for metric_name, metric_value in metrics.items()]

        except Exception as e:
            logger.error(f"Exception occurred during model training: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            mlflow.log_param("status", "FAILED")

        else:
            # debugging
            logger.info(f"Input columns: {', '.join(configs['data.input_columns'])}")
            logger.info(f"Output columns: {', '.join(configs['data.output_columns'])}")
            logger.info(f"Final model parameters: { {**final_params} }")
            # success
            mlflow.log_param("status", "SUCCESS")
            logger.info(f"Model training completed - full path:{model_path}")
        
        finally:
            mlflow.log_artifact(file_handler_path)
            logger.info(f"Finished run: {experiment_run_path}")            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gc', '--global_config', type=str, default='global_config.json', help='Global JSON config in ./configs')
    parser.add_argument('-c', '--config', type=str, default='config_iris_tuning.json', help='Experiment JSON config in ./configs')
    args = parser.parse_args()
    main(args.global_config, args.config)
