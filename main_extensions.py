import argparse
from utils.config_loader import get_config
from utils.file_processor import load_json, load_csv, get_full_path
from utils.image_processor import ImageFormat, save_image, show_image
from extensions.plot_lgbm_tree import plot_lightgbm_tree
from extensions.plot_best_estimator import plot_best_estimator
from extensions.plot_cv_results import plot_cv_results


def main(cfg: dict, ext: str, debug: bool, last_exec: dict):

    if 'plot_lgbm_tree' in ext:
        plt = plot_lightgbm_tree(last_exec['model_path'])
        full_output_path = save_image(
            plt, last_exec['artifacts_path'], 'plot_lgbm_tree', ImageFormat.PNG)
        show_image(full_output_path) if debug else None

    if 'plot_best_estimator' in ext:
        best_estimator = load_json(get_full_path(
            last_exec['artifacts_path'], "best_estimator.json"))
        plt = plot_best_estimator(best_estimator)
        full_output_path = save_image(
            plt, last_exec['artifacts_path'], 'plot_best_estimator', ImageFormat.PNG)
        show_image(full_output_path) if debug else None

    if 'plot_cv_results' in ext:
        cv_results = load_csv(get_full_path(
            last_exec['artifacts_path'], "cv_results.csv"))
        plt = plot_cv_results(cv_results)
        full_output_path = save_image(
            plt, last_exec['artifacts_path'], 'plot_cv_results', ImageFormat.PNG)
        show_image(full_output_path) if debug else None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config_path', type=str, default='configs', help='Configs project path')
    parser.add_argument('-g', '--config_global_file_name', type=str, default='global.json', help='Global JSON config file name')
    parser.add_argument('-e', '--extensions', nargs='+', type=str, default=['plot_lgbm_tree', 'plot_cv_results', 'plot_best_estimator'], help='Extensions to run')
    parser.add_argument('-d', '--debug', action='store_true', help='Flag to enable debug functionality of extension(s)')
    args = parser.parse_args()

    config_global = load_json(get_full_path(args.config_path, args.config_global_file_name))
    last_exec = load_json(get_full_path(get_config(config_global, 'global.outputs_dir'), get_config(config_global, 'global.last_exec_file_name')))

    main(cfg=config_global, ext=args.extensions, debug=args.debug, last_exec=last_exec)
