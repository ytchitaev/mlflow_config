import json
import os
import pandas as pd
from typing import Union, List



def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def load_csv(path: str):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("File not found at the specified path.")
        return None
    except Exception as e:
        print("An error occurred while loading the CSV file:", str(e))
        return None


def get_full_path(path_values: Union[str, List[Union[str, int]]], file_name: Union[str, int] = None):
    """Get the full path by joining the path values list or single string and an optional file name."""
    path_values = [str(path_values)] if isinstance(
        path_values, (str, int)) else list(map(str, path_values))
    path = os.path.join(*[p.strip('/') for p in path_values])
    if file_name is not None:
        path = os.path.join(path, str(file_name))
    return path

def write_json(json_obj, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(json_obj, json_file)


def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

