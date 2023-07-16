import json
import os
import pandas as pd
from enum import Enum
from typing import Union, List
import webbrowser


class ImageFormat(Enum):
    PNG = 'png'
    JPG = 'jpg'
    SVG = 'svg'


def save_image(image, output_path, filename="image", file_format=ImageFormat.PNG):
    supported_formats = [f.value for f in ImageFormat]

    if file_format.value not in supported_formats:
        print(f"Unsupported file format: {file_format.value}")
        return

    full_output_path = f"{output_path}/{filename}.{file_format.value}"

    format_options = {
        ImageFormat.PNG: {'format': 'png'},
        ImageFormat.JPG: {'format': 'jpg'},
        ImageFormat.SVG: {'format': 'svg'}
    }

    try:
        image.savefig(full_output_path, **format_options[file_format])
        print(f"Image saved as {file_format.name} at {full_output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
    return full_output_path


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
    path = os.path.join(os.getcwd(), *[p.strip('/') for p in path_values])
    if file_name is not None:
        path = os.path.join(path, str(file_name))
    return path


def show_image_in_browser_if_debug(path):
    webbrowser.open(path)
