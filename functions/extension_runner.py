from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
import matplotlib.pyplot as plt

from utils.config_loader import get_config
from utils.image_processor import ImageFormat, save_image, show_image


@dataclass
class ExtensionImplementation(ABC):
    """Generic base class for extensions implementations"""
    logger: Any
    cfg: Dict[str, Any]
    extension_name: str
    debug: bool
    plt_figure: plt.figure = None

    @abstractmethod
    def check_extension_viability(self) -> bool:
        ...

    @abstractmethod
    def load_extension(self) -> Any:
        ...

    @abstractmethod
    def plot_extension(self, data: Any) -> plt.figure:
        ...

    def __post_init__(self):
        "process extension activities"
        if self.check_extension_viability():
            data = self.load_extension()
            self.plt_figure = self.plot_extension(data)
            self.full_output_path = self.write_extension()
            if self.debug:
                self.show_extension(self.full_output_path)

    def write_extension(self) -> str:
        "generic write function from matplotlib.pyplot plt"
        output_path = get_config(self.cfg, 'execution.artifact_path')
        full_output_path = save_image(self.logger, self.plt_figure, output_path, self.extension_name, ImageFormat.PNG)
        return full_output_path

    def show_extension(self, image_path: str):
        if self.debug:
            show_image(image_path)
