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
