import argparse
import traceback
import copy
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from functions.config_loader import setup_run, load_json, get_path, combine_configs, get_config, output_last_exec_json
from functions.mlflow_logger import init_python_logger, mlflow_log_artifact_dict_to_csv, mlflow_log_artifact_dict_to_json
from functions.data_loader import load_data
from functions.data_splitter import split_dataset
from functions.model_runner import create_model
from functions.tuning_runner import TuningRunner
from functions.metrics_evaluator import MetricFactory


def main(cfg):

    mlflow.set_experiment(get_config(cfg, 'setup.experiment_name'))
    with mlflow.start_run() as run:

        try:

            # Create logger, define run identifiers and log config and global config
            run_id, experiment_id, experiment_run_path, model_path = setup_run(run)
            logger, file_handler_path = init_python_logger(experiment_run_path, get_config(cfg, 'global.run_temp_subdir'), get_config(cfg, 'global.python_log_file_name'))
            mlflow_log_artifact_dict_to_json(experiment_run_path, get_config(cfg, 'global.run_temp_subdir'), "config.json", get_config(cfg))
            output_last_exec_json(run_id, experiment_id)
            logger.info(f"Started run: {experiment_run_path}")

            # Load data
            logger.info("Loading data...")
            X_input, y_input = load_data(**get_config(cfg, 'data'))

            # Split data
            logger.info("Splitting data...")
            X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(X_input, y_input, **get_config(cfg, 'split'))

            # Create model with initial params and create final params
            logger.info("Loading initial model...")
            initial_params = get_config(cfg, 'model.params')
            final_params = copy.deepcopy(initial_params)
            model = create_model(get_config(cfg, 'model.library_name'), get_config(cfg, 'model.model_name'), params=initial_params)

            # Run tuning if enabled, update final params with tuned values and log final params
            if get_config(cfg, 'tuning'):
                logger.info("Running tuning...")
                tunning_runner = TuningRunner(get_config(cfg, 'tuning.name'), get_config(cfg, 'tuning.params'))
                cv_params, cv_results = tunning_runner.run_tuning(model, X_train, y_train)
                final_params.update(cv_params)
                mlflow.log_params(final_params)
                mlflow_log_artifact_dict_to_csv(experiment_run_path, get_config(cfg, 'global.run_temp_subdir'), get_config(cfg, 'artefacts.cv_results.file_name'), cv_results)

            # Train the model with final params and log model
            logger.info("Training model...")
            best_model = create_model(get_config(cfg, 'model.library_name'), get_config(cfg, 'model.model_name'), params=final_params)
            best_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=100, show_stdv=True)])
            mlflow.lightgbm.log_model(best_model, "model")

            # Evaluate model and log evaluations
            logger.info("Evaluating model...")
            metrics_factory = MetricFactory(best_model)
            metrics = metrics_factory.create_metrics(get_config(cfg, 'evaluate'), X_validation, X_test, y_validation, y_test)
            [mlflow.log_metric(metric_name, metric_value) for metric_name, metric_value in metrics.items()]

        except Exception as e:
            logger.error(f"Exception occurred during model training: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            mlflow.log_param("status", "FAILED")

        else:
            # print log of result
            logger.info(f"Input columns: {', '.join(get_config(cfg,'data.input_columns'))}")
            logger.info(f"Output columns: {', '.join(get_config(cfg,'data.output_columns'))}")
            logger.info(f"Final model parameters: { {**final_params} }")
            # success
            mlflow.log_param("status", "SUCCESS")
            logger.info(f"Model training completed - full path:{model_path}")

        finally:
            mlflow.log_artifact(file_handler_path)
            logger.info(f"Finished run: {experiment_run_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config_path', type=str, default='configs', help='Configs project path')
    parser.add_argument('-g', '--config_global_file_name', type=str, default='global.json', help='Global JSON config file name')
    parser.add_argument('-d', '--config_default_file_name', type=str, default='default.json', help='Default experiment JSON config file name')
    parser.add_argument('-e', '--config_experiment_file_name', type=str, default='iris_tuning.json', help='Experiment JSON config file name')
    args = parser.parse_args()

    config_global = load_json(get_path(args.config_path, args.config_global_file_name))
    config_default = load_json(get_path(args.config_path, args.config_default_file_name))
    config_experiment = load_json(get_path(args.config_path, args.config_experiment_file_name))
    config_combined = combine_configs(config_global, config_default, config_experiment)

    main(cfg=config_combined)
