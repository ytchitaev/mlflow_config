import json
from typing import Any, List, Union
from pathlib import Path

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def get_config_dir(file_name: str):
    "navigate up from functions down to configs"
    config_dir = Path(__file__).parent.parent / 'configs'
    return config_dir / file_name


def get_config_value(config: Union[dict, list], keys: List[str], default: Any = None):
    """
    Recursive function to get a configuration value from a nested dictionary or list.
    :param config: The configuration dictionary or list.
    :param keys: The list of keys representing the path in the configuration dictionary.
    :param default: The default value to return if the configuration value is not found.
    :return: The configuration value or the default value.
    """
    # Base case: if there are no more keys, return the configuration or the default value
    if not keys or isinstance(config, list):
        return config if config is not None else default

    # Get the next key
    key = keys.pop(0)

    # If the key is in the dictionary, recurse with the remaining keys
    if key in config:
        return get_config_value(config[key], keys, default)

    # If the key is not in the dictionary, return the default value
    return default

def load_configurations(config_file_name: str, mapping_specification: dict):
    config_path = get_config_dir(config_file_name)
    config = load_json(config_path)

    configurations = {'': config}  # Include the root element

    for path, default in mapping_specification.items():
        keys = path.split('.')
        value = get_config_value(config, keys, default)
        configurations[path] = value

    return configurations
