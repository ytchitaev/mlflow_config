import argparse
import traceback
import copy
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from utils.config_loader import combine_configs, get_config
from utils.file_processor import load_json, get_full_path
from functions.run_manager import setup_run, output_last_exec_json
from functions.mlflow_logger import init_python_logger, mlflow_log_artifact_dict_to_csv, mlflow_log_artifact_dict_to_json
from functions.data_loader import load_data
from functions.data_splitter import split_dataset
from functions.model_runner import create_model
from functions.tuning_runner import TuningRunner
from functions.metrics_evaluator import MetricFactory


def main(cfg: dict):

    mlflow.set_experiment(get_config(cfg, 'setup.experiment_name'))
    with mlflow.start_run() as run:

        try:

            # Create logger, define run identifiers and log config and global config
            exec = setup_run(run, cfg)
            logger, file_handler_path = init_python_logger(exec.experiment_run_path, get_config(cfg, 'global.temp_dir'), get_config(cfg, 'global.python_log_file_name'))
            mlflow_log_artifact_dict_to_json(exec.experiment_run_path, get_config(cfg, 'global.temp_dir'), "config.json", get_config(cfg))
            output_last_exec_json(exec, cfg)
            logger.info(f"Started run: {exec.experiment_run_path}")

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
                cv_params, cv_results, best_estimator = tunning_runner.run_tuning(model, X_train, y_train, X_validation, y_validation)
                final_params.update(cv_params)
                mlflow_log_artifact_dict_to_csv(exec.experiment_run_path, get_config(cfg, 'global.temp_dir'), get_config(cfg, 'artefacts.cv_results.file_name'), cv_results)
                if best_estimator:
                    mlflow_log_artifact_dict_to_json(exec.experiment_run_path, get_config(cfg, 'global.temp_dir'), get_config(cfg, 'artefacts.best_estimator.file_name'), best_estimator.evals_result_)

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
            mlflow.log_param("status", "SUCCESS")
            logger.info(f"Model run completed.")

        finally:
            logger.info(f"{'Input columns:' : <25} {get_config(cfg,'data.input_columns')}")
            logger.info(f"{'Output columns:' : <25} {get_config(cfg,'data.output_columns')}")
            logger.info(f"{'Model full path:' : <25} {exec.model_path}")
            logger.info(f"{'Final model parameters:' : <25} { {**final_params} }")
            logger.info(f"{'Finished run:' : <25} {exec.experiment_run_path}")
            mlflow.log_params(final_params)
            mlflow.log_artifact(file_handler_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config_path', type=str, default='configs', help='Configs project path')
    parser.add_argument('-g', '--config_global_file_name', type=str, default='global.json', help='Global JSON config file name')
    parser.add_argument('-d', '--config_default_file_name', type=str, default='default.json', help='Default experiment JSON config file name')
    parser.add_argument('-e', '--config_experiment_file_name', type=str, default='iris_tuning.json', help='Experiment JSON config file name')
    args = parser.parse_args()

    config_global = load_json(get_full_path(args.config_path, args.config_global_file_name))
    config_default = load_json(get_full_path(args.config_path, args.config_default_file_name))
    config_experiment = load_json(get_full_path(args.config_path, args.config_experiment_file_name))
    config_combined = combine_configs(config_global, config_default, config_experiment)

    main(cfg=config_combined)
