import argparse
from extensions.utils import ImageFormat, save_image, load_json, get_full_path, show_image_in_browser_if_debug
from extensions.plot_lgbm_tree import plot_lightgbm_tree

OUTPUTS_DIR = "outputs"
LAST_EXEC_FILE_NAME = "last_exec.json"


def main(ext: str, debug: bool, last_exec: dict):

    if 'plot_lgbm_tree' in ext:
        plt = plot_lightgbm_tree(last_exec['model_path'])
        full_output_path = save_image(plt, last_exec['artifacts_path'], 'plot_lgbm_tree', ImageFormat.PNG)
        show_image_in_browser_if_debug(full_output_path) if debug else None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--extensions', nargs='+', type=str,
                        default='plot_lgbm_tree', help='Extensions to run')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Flag to enable debug functionality of extension(s)')
    args = parser.parse_args()

    last_exec = load_json(get_full_path(OUTPUTS_DIR, LAST_EXEC_FILE_NAME))
    main(ext=args.extensions, debug=args.debug, last_exec=last_exec)
