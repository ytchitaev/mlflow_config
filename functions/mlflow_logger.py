import os
import sys
import pandas as pd
import mlflow
import logging

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def instantiate_python_logger(experiment_run_path, run_temp_folder, python_logging_file_name):
    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Redirect stdout to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)

    # Create a console handler for logging output to the CLI/IDE
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a file handler for logging output to a file
    temp_dir = os.path.join(experiment_run_path, run_temp_folder)
    os.makedirs(temp_dir, exist_ok=True)
    file_handler_path = os.path.join(temp_dir, python_logging_file_name)
    file_handler = logging.FileHandler(file_handler_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, file_handler_path


def mlflow_log_artifact_cv_results(experiment_id, run_id, run_temp_folder, cv_results_file_name, cv_results):
    cv_results_df = pd.DataFrame(cv_results)
    temp_dir = os.path.join("mlruns", experiment_id, run_id, run_temp_folder)
    os.makedirs(temp_dir, exist_ok=True)
    csv_path = os.path.join(temp_dir, cv_results_file_name)
    cv_results_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    os.remove(csv_path)
