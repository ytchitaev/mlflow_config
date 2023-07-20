import argparse
import mlflow

from loggers.python_logger import LoggingTypes, init_python_logger
from utils.file_processor import load_json, get_relative_path
from extensions.plot_lgbm_tree import ExtensionLGBMTree
from extensions.plot_cv_results import ExtensionCVResults
from extensions.plot_best_estimator_evals_result import ExtensionBestEstimatorEvalResult


def main(cfg: dict, debug: bool):

    # init logger
    logger, file_handler_path = init_python_logger(cfg, LoggingTypes.EXTENSION)
    # run extensions
    ExtensionLGBMTree(logger, cfg, "lgbm_tree", debug)
    ExtensionCVResults(logger, cfg, "cv_results", debug)
    ExtensionBestEstimatorEvalResult(logger, cfg, "best_estimator_eval_result", debug)
    # finalise
    mlflow.log_artifact(file_handler_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', type=str, default='outputs', help='Outputs project path')
    parser.add_argument('-c', '--last_execution_config', type=str, default='last_execution_config.json', help='Last execution config file name')
    parser.add_argument('-d', '--debug', action='store_true', help='Flag to enable debug functionality of extension(s)')
    args = parser.parse_args()

    cfg = load_json(get_relative_path(args.output_path, args.last_execution_config))
    main(cfg=cfg, debug=args.debug)
 