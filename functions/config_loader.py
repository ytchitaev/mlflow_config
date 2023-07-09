import json
from pathlib import Path

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def get_config_dir(file_name: str):
    "navigate up from functions down to configs"
    config_dir = Path(__file__).parent.parent / 'configs'
    return config_dir / file_name

def load_configurations(config_file_name: str):
    config_path = get_config_dir(config_file_name)
    config = load_json(config_path)
    configurations = {
        # root
        'setup': config.get('setup', {}),
        # setup
        'data_source': config.get('setup', {}).get('data_source', {}),
        'dataset_name': config.get('setup', {}).get('dataset_name', {}),    
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
