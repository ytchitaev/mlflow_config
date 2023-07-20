import copy
import mlflow

from utils.config_loader import get_config, combine_configs

from loggers.mlflow_artifact_logger import mlflow_log_artifact_dict_to_json
from loggers.python_logger import LoggingTypes, init_python_logger

from functions.run_manager import build_execution_config 
from functions.data_loader import load_data
from functions.data_splitter import split_dataset
from functions.model_creator import create_model
from functions.model_runner import fit_model, log_model
from functions.tuning_runner import TuningRunner
from functions.metrics_evaluator import MetricFactory


# initiate run, add execution configs, initiate python logger
def run_stage_initiate_run(cfg: dict, run):
    exec_cfg = build_execution_config(run, get_config(cfg, 'global'))
    cfg = combine_configs(cfg, exec_cfg)
    logger, file_handler_path = init_python_logger(cfg, LoggingTypes.MLFLOW)
    [mlflow.set_tag(key, value) for key, value in get_config(cfg, 'setup.tags').items()]
    mlflow_log_artifact_dict_to_json(cfg, get_config(cfg, 'global.config_file_name'), get_config(cfg))
    logger.info(f"Started run: {get_config(cfg, 'execution.experiment_run_path')}")
    return cfg, logger, file_handler_path


# load x, y data objects from data source based on config
def run_stage_load_data(logger, cfg):
    X_input, y_input = load_data(logger, **get_config(cfg, 'data'))
    return X_input, y_input


# split data into train, validation, test based on config
def run_stage_split_dataset(logger, cfg, X_input, y_input):
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(
                logger, X_input, y_input, **get_config(cfg, 'split'))
    return X_train, X_validation, X_test, y_train, y_validation, y_test


# Create model with initial params and create mutable final params for fine tuning
def run_stage_initial_model(logger, cfg):
    logger.info("Loading initial model and config params...")
    initial_params = get_config(cfg, 'model.params')
    final_params = copy.deepcopy(initial_params)
    model = create_model(cfg_model=get_config(
        cfg, 'model'), params=initial_params)
    return model, initial_params, final_params

# Run tuning if enabled, update final params with tuned values and log final params
def run_stage_tuning(logger, cfg, model, final_params, X_train, y_train, X_validation, y_validation):
    if get_config(cfg, 'tuning'):
        logger.info("Tuning model...")
        tuning_runner = TuningRunner(get_config(cfg, 'tuning'))
        tuning_params, tuning_artifacts = tuning_runner.run_tuning(
            model, X_train, y_train, X_validation, y_validation)
        final_params.update(tuning_params.best_params)
        cfg = tuning_artifacts.log_tuning_artifacts(cfg, logger)
    return final_params, cfg


# Train the model with final params and log model
def run_stage_fit_model(logger, cfg, final_params, X_train, y_train):
    logger.info("Train model...")
    best_model = create_model(cfg_model=get_config(
        cfg, 'model'), params=final_params)
    fit_model(get_config(cfg, 'model'), best_model, X_train, y_train)
    log_model(get_config(cfg, 'model'), best_model, "model")
    return best_model


# Evaluate model and log evaluations
def run_stage_evaluate_model(logger, cfg, best_model, X_validation, X_test, y_validation, y_test):
    logger.info("Evaluating model...")
    metrics_factory = MetricFactory(best_model)
    metrics = metrics_factory.create_metrics(get_config(
    cfg, 'evaluate'), X_validation, X_test, y_validation, y_test)
    mlflow.log_metrics(metrics)