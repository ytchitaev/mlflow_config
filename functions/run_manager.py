import mlflow
import traceback

from utils.config_loader import get_config
from utils.file_processor import get_relative_path, check_and_create_path, write_json


def build_execution_config(run, cfg_global: dict) -> dict:
    experiment_run_path = get_relative_path([cfg_global['mlruns_dir'], run.info.experiment_id, run.info.run_id])
    artifact_path = get_relative_path([experiment_run_path, cfg_global['artifacts_dir']])
    model_path = get_relative_path([artifact_path, cfg_global['artifacts_model_subdir_path']])
    return {
        "execution": {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_run_path": experiment_run_path,
            "artifact_path": artifact_path,
            "model_path": model_path,
        }
    }


def write_last_config(cfg: dict):
    output_path = get_relative_path(get_config(cfg, 'global.outputs_dir'))
    check_and_create_path(output_path)
    last_exec_config_file = get_relative_path([output_path, get_config(cfg, 'global.last_execution_config')])
    write_json(cfg, last_exec_config_file)
    pass


def setup_experiment(cfg):
    mlflow.set_experiment(get_config(cfg, 'setup.experiment_name'))
    pass


def log_run_outcome(status, logger, exception:Exception=None, traceback:traceback=None):
    mlflow.log_param("status", status)
    if exception:
        logger.error(f"Exception occurred during model training: {str(exception)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        logger.info(f"Model run completed.")
    pass


def finalise_run(cfg, logger, final_params, file_handler_path):
    logger.info(f"{'Final model parameters:' : <25} { {**final_params} }")
    logger.info(f"{'Input columns:' : <25} {get_config(cfg,'data.input_columns')}")
    logger.info(f"{'Output columns:' : <25} {get_config(cfg,'data.output_columns')}")
    logger.info(f"{'Model full path:' : <25} {get_config(cfg, 'execution.model_path')}")
    logger.info(f"{'Finished run:' : <25} {get_config(cfg, 'execution.experiment_run_path')}")
    write_last_config(cfg)
    mlflow.log_params(final_params)
    mlflow.log_artifact(file_handler_path)
    pass