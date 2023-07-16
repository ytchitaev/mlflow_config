from dataclasses import dataclass, asdict

from utils.file_processor import write_json, check_and_create_path, get_full_path
from utils.config_loader import get_config


@dataclass
class ExecInstance:
    run_id: str
    experiment_id: str
    experiment_run_path: str
    artifacts_path: str
    model_path: str

def setup_run(run, cfg:dict):
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    experiment_run_path = get_full_path([get_config(cfg, 'global.mlruns_dir'), experiment_id, run_id])
    artifacts_path = get_full_path([get_config(cfg, 'global.mlruns_dir'), experiment_id, run_id, get_config(cfg, 'global.artifacts_dir')])
    model_path = get_full_path([get_config(cfg, 'global.mlruns_dir'), experiment_id, run_id, get_config(cfg, 'global.artifacts_dir'), get_config(cfg, 'global.artifacts_model_subdir_path')])
    return ExecInstance(run_id, experiment_id, experiment_run_path, artifacts_path, model_path)


def output_last_exec_json(last_exec: ExecInstance, cfg:dict):
    check_and_create_path(get_full_path(get_config(cfg, 'global.outputs_dir')))
    last_exec_path_file = get_full_path(get_config(cfg, 'global.outputs_dir'), get_config(cfg, 'global.last_exec_file_name'))
    last_exec_data = asdict(last_exec)
    write_json(last_exec_data, last_exec_path_file)
