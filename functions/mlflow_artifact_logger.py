import os
import pandas as pd
import mlflow

from utils.config_loader import get_config
from utils.file_processor import write_json

def mlflow_log_artifact_dict_to_csv(exec, cfg, file_name, dictionary):
    dictionary_df = pd.DataFrame(dictionary)
    temp_dir = os.path.join(exec.experiment_run_path,
                            get_config(cfg, 'global.temp_dir'))
    os.makedirs(temp_dir, exist_ok=True)
    csv_path = os.path.join(temp_dir, file_name)
    dictionary_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    os.remove(csv_path)


def mlflow_log_artifact_dict_to_json(exec, cfg, file_name, dictionary):
    temp_dir = os.path.join(exec.experiment_run_path,
                            get_config(cfg, 'global.temp_dir'))
    os.makedirs(temp_dir, exist_ok=True)
    json_path = os.path.join(temp_dir, file_name)
    write_json(dictionary, json_path)
    mlflow.log_artifact(json_path)
    os.remove(json_path)
