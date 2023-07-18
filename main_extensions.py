import argparse
from utils.config_loader import get_config, is_list_item_in_dict
from utils.file_processor import load_json, load_csv, get_relative_path
from utils.image_processor import ImageFormat, save_image, show_image
from extensions.plot_lgbm_tree import plot_lightgbm_tree
from extensions.plot_best_estimator_evals_result import plot_best_estimator_evals_result
from extensions.plot_cv_results import plot_cv_results


def main(cfg: dict, ext: str, debug: bool):

    #TODO - ExtensionFactory / ExtensionRunner
    # https://chat.openai.com/share/eb595742-e4c9-408b-861b-506cf0ba99eb

    if 'plot_lgbm_tree' in ext:
        plt = plot_lightgbm_tree(get_config(cfg, 'execution.model_path'))
        full_output_path = save_image(plt, get_config(cfg, 'execution.artifact_path'), 'plot_lgbm_tree', ImageFormat.PNG)
        show_image(full_output_path) if debug else None

    if ('plot_best_estimator_evals_result' in ext) and is_list_item_in_dict('best_estimator_evals_result', cfg['artifacts']):
        best_estimator_evals_result = load_json(get_relative_path(get_config(cfg, 'execution.artifact_path'), "best_estimator_evals_result.json"))
        plt = plot_best_estimator_evals_result(best_estimator_evals_result)
        full_output_path = save_image(plt, get_config(cfg, 'execution.artifact_path'), 'plot_best_estimator_evals_result', ImageFormat.PNG)
        show_image(full_output_path) if debug else None

    if ('plot_cv_results' in ext) and is_list_item_in_dict('cv_results', cfg['artifacts']):
        cv_results = load_csv(get_relative_path(get_config(cfg, 'execution.artifact_path'), "cv_results.csv"))
        plt = plot_cv_results(cv_results)
        full_output_path = save_image(plt, get_config(cfg, 'execution.artifact_path'), 'plot_cv_results', ImageFormat.PNG)
        show_image(full_output_path) if debug else None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', type=str, default='outputs', help='Outputs project path')
    parser.add_argument('-c', '--last_execution_config', type=str, default='last_execution_config.json', help='Last execution config file name')
    parser.add_argument('-e', '--extensions', nargs='+', type=str, default=['plot_lgbm_tree', 'plot_cv_results', 'plot_best_estimator_evals_result'], help='Extensions to run')
    parser.add_argument('-d', '--debug', action='store_true', help='Flag to enable debug functionality of extension(s)')
    args = parser.parse_args()

    cfg = load_json(get_relative_path(args.output_path, args.last_execution_config))
    main(cfg=cfg, ext=args.extensions, debug=args.debug)
