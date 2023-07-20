from enum import Enum
import webbrowser


class ImageFormat(Enum):
    PNG = 'png'
    JPG = 'jpg'
    SVG = 'svg'


def save_image(logger, image, output_path, filename="image", file_format=ImageFormat.PNG):
    supported_formats = [f.value for f in ImageFormat]

    if file_format.value not in supported_formats:
        logger.info(f"Unsupported file format: {file_format.value}")
        #print(f"Unsupported file format: {file_format.value}")
        return

    full_output_path = f"{output_path}/{filename}.{file_format.value}"

    format_options = {
        ImageFormat.PNG: {'format': 'png'},
        ImageFormat.JPG: {'format': 'jpg'},
        ImageFormat.SVG: {'format': 'svg'}
    }

    try:
        image.savefig(full_output_path, **format_options[file_format])
        logger.info(f"Image saved as {file_format.name} at {full_output_path}")
        #print(f"Image saved as {file_format.name} at {full_output_path}")
    except Exception as e:
        logger.info(f"Error saving image: {e}")
        #print(f"Error saving image: {e}")
    return full_output_path


def show_image(path):
    webbrowser.open(path)
