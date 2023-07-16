import os
import json
from typing import Union, List

MLRUNS_DIR = "mlruns"
OUTPUTS_DIR = "outputs"
CONFIGS_DIR = "configs"
ARTIFACTS_DIR = "artifacts"
MODEL_RUN_SUBDIR = "artifacts/model/model.pkl"
LAST_EXEC_FILE_NAME = "last_exec.json"


def write_json(last_run, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(last_run, json_file)


def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_full_path(path_values: Union[str, List[Union[str, int]]], file_name: Union[str, int] = None):
    """Get the full path by joining the path values list or single string and an optional file name."""
    path_values = [str(path_values)] if isinstance(
        path_values, (str, int)) else list(map(str, path_values))
    path = os.path.join(os.getcwd(), *[p.strip('/') for p in path_values])
    if file_name is not None:
        path = os.path.join(path, str(file_name))
    return path


def define_last_exec(run_id, experiment_id, experiment_run_path, artifacts_path, model_path):
    last_run = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_run_path": experiment_run_path,
        "artifacts_path": artifacts_path,
        "model_path": model_path
    }
    return last_run


def setup_run(run):
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    experiment_run_path = get_full_path([MLRUNS_DIR, experiment_id, run_id])
    artifacts_path = get_full_path(
        [MLRUNS_DIR, experiment_id, run_id, ARTIFACTS_DIR])
    model_path = get_full_path(
        [MLRUNS_DIR, experiment_id, run_id, MODEL_RUN_SUBDIR])
    return run_id, experiment_id, experiment_run_path, artifacts_path, model_path


def output_last_exec_json(run_id, experiment_id, experiment_run_path, artifacts_path, model_path):
    check_and_create_path(get_full_path(OUTPUTS_DIR))
    last_exec_path_file = get_full_path(OUTPUTS_DIR, LAST_EXEC_FILE_NAME)
    last_exec_data = define_last_exec(
        run_id, experiment_id, experiment_run_path, artifacts_path, model_path)
    write_json(last_exec_data, last_exec_path_file)
