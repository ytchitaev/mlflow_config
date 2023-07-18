import argparse
import traceback
import mlflow

from utils.config_loader import combine_configs
from utils.file_processor import load_json, get_relative_path
from functions.run_manager import setup_experiment, finalise_run, log_run_outcome
from stages.experiment_stages import *


def main(cfg: dict):

    setup_experiment(cfg)
    with mlflow.start_run() as run:

        try:
            cfg, logger, file_handler_path = run_stage_initiate_run(cfg, run)
            X_input, y_input = run_stage_load_data(logger, cfg)
            X_train, X_validation, X_test, y_train, y_validation, y_test = run_stage_split_dataset(logger, cfg, X_input, y_input)
            model, initial_params, final_params = run_stage_initial_model(logger, cfg)
            final_params = run_stage_tuning(logger, cfg, model, final_params, X_train, y_train, X_validation, y_validation)
            best_model = run_stage_fit_model(logger, cfg, final_params, X_train, y_train)
            run_stage_evaluate_model(logger, cfg, best_model, X_validation, X_test, y_validation, y_test)

        except Exception:
            log_run_outcome("FAILED", logger, Exception, traceback)

        else:
            log_run_outcome("SUCCESS", logger)

        finally:
            finalise_run(cfg, logger, final_params, file_handler_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config_path', type=str, default='configs', help='Configs project path')
    parser.add_argument('-g', '--config_global_file_name', type=str, default='global.json', help='Global JSON config file name')
    parser.add_argument('-d', '--config_default_file_name', type=str, default='default.json', help='Default experiment JSON config file name')
    parser.add_argument('-e', '--config_experiment_file_name', type=str, default='iris_tuning.json', help='Experiment JSON config file name')
    args = parser.parse_args()

    config_global = load_json(get_relative_path(args.config_path, args.config_global_file_name))
    config_default = load_json(get_relative_path(args.config_path, args.config_default_file_name))
    config_experiment = load_json(get_relative_path(args.config_path, args.config_experiment_file_name))
    config_combined = combine_configs(config_global, config_default, config_experiment)

    main(cfg=config_combined)
