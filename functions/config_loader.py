import json


def load_json(file_name: str):
    with open(file_name, 'r') as f:
        return json.load(f)


def load_configurations(config_file: str):
    config = load_json(config_file)
    configurations = {
        # root
        'setup': config.get('setup', {}),
        # setup
        'library_name': config.get('setup', {}).get('library_name', {}),
        'model_name': config.get('setup', {}).get('model_name', {}),
        # 'verbose_fit_flag': config.get('setup', {}).get('verbose_fit_flag', '0'),
        # name
        'experiment_name': config.get('setup', {}).get('experiment_name', 'Default'),
        # split
        'train_percentage': config.get('setup', {}).get('split', {}).get('train_percentage', 80),
        'validation_percentage': config.get('setup', {}).get('split', {}).get('validation_percentage', 10),
        'test_percentage': config.get('setup', {}).get('split', {}).get('test_percentage', 10),
        # grid search
        'grid_search': config.get('grid_search', {}),
        'param_grid': config.get('grid_search', {}).get('param_grid', {}),
        'cv': config.get('grid_search', {}).get('cv', {}),
        # paths
        'run_temp_folder': config.get('setup', {}).get('paths', {}).get('run_temp_folder', 'temp'),
        # artefacts
        'python_logging_file_name': config.get('setup', {}).get('artefacts', {}).get('python_logging', {}).get('file_name', 'python_logging.txt'),
        'cv_results_file_name': config.get('setup', {}).get('artefacts', {}).get('cv_results', {}).get('file_name', 'cv_results.csv'),
    }
    return configurations
