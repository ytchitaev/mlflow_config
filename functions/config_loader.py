import os
import json


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(last_run, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(last_run, json_file)


def get_path(relative_path: str = 'configs', file_name: str = 'config_global.json'):
    return os.path.join(os.getcwd(), relative_path, file_name)


def combine_configs(*configs):
    combined_config = {}
    for config in configs:
        if config is not None:
            combined_config = _recursive_merge(combined_config, config)
    return combined_config


def _recursive_merge(dict1, dict2):
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(value, dict):
            merged_dict[key] = _recursive_merge(merged_dict[key], value)
        else:
            merged_dict[key] = value
    return merged_dict


def get_config(json_obj, attribute_path="", default=None):
    """
    Get the value of an attribute from a JSON object.

    Args:
        json_obj (dict or list): The JSON object to retrieve the attribute from.
        attribute_path (str, optional): The attribute path in dot notation. Defaults to an empty string.
        default (any, optional): The default value to return if the attribute is not found. Defaults to None.

    Returns:
        any: The value of the attribute if found, otherwise the default value.

    Examples:
        # Retrieve the full json object
        value = get_config(json_obj)

        # Retrieve an attribute
        value = get_config(json_obj, "element1.element2")

        # Retrieve a list item
        value = get_config(json_obj, "element1.element2.0")
    """

    if not attribute_path:
        return json_obj

    attributes = attribute_path.split(".")
    current_obj = json_obj

    for attribute in attributes:
        try:
            current_obj = current_obj[int(attribute)] if isinstance(
                current_obj, list) else current_obj[attribute]
        except (KeyError, IndexError, TypeError) as e:
            element_type = "list index" if isinstance(
                current_obj, list) else "attribute"
            print(
                f"Config {element_type} '{attribute}' is not defined {str(e)}")
            return default
    else:
        return current_obj if current_obj is not None else default


def define_last_exec(run_id, experiment_id):
    last_run = {
        "run_id": run_id,
        "experiment_id": experiment_id
    }
    return last_run


def setup_run(run):
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    experiment_run_path = os.path.join("mlruns", experiment_id, run_id)
    model_path = os.path.join("mlruns", experiment_id,
                              run_id, "artifacts/model/model.pkl")
    return run_id, experiment_id, experiment_run_path, model_path


def output_last_exec_json(run_id, experiment_id):
    last_exec_path = get_path("outputs", "last_exec.json")
    last_exec_data = define_last_exec(run_id, experiment_id)
    write_json(last_exec_data, last_exec_path)