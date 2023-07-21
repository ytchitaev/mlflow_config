import os
import sys
import logging
from enum import Enum

from utils.config_loader import get_config


class LoggingTypes(Enum):
    MLFLOW = 'mlflow'
    EXTENSION = 'extension'


class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance."""

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def init_python_logger(cfg: dict, component: LoggingTypes):
    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Redirect stdout to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)

    # Create a console handler for logging output to the CLI/IDE
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a file handler for logging output to a file
    temp_dir = os.path.join(get_config(
        cfg, 'execution.experiment_run_path'), get_config(cfg, 'global.temp_dir'))
    os.makedirs(temp_dir, exist_ok=True)
    if component == LoggingTypes.MLFLOW:
        file_handler_path = os.path.join(temp_dir, get_config(
            cfg, 'global.python_mlflow_log_file_name'))
    elif component == LoggingTypes.EXTENSION:
        file_handler_path = os.path.join(temp_dir, get_config(
            cfg, 'global.python_extension_log_file_name'))
    file_handler = logging.FileHandler(file_handler_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, file_handler_path
