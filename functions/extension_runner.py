from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
import matplotlib.pyplot as plt

from utils.config_loader import get_config
from utils.image_processor import ImageFormat, save_image, show_image


@dataclass
class ExtensionImplementation(ABC):
    """Generic base class for extensions implementations"""
    cfg: Dict[str, Any]
    extension_name: str
    debug: bool

    @abstractmethod
    def check_extension_viability(self, cfg: dict) -> bool:
        pass

    @abstractmethod
    def load_extension(self, cfg: dict) -> Any:
        pass

    @abstractmethod
    def plot_extension(self, data: Any) -> plt.figure:
        pass

    """Methods on init and common method implementation"""

    def __post_init__(self):
        if self.check_extension_viability(self.cfg):
            data = self.load_extension(self.cfg)
            plt_figure = self.plot_extension(data)
            self.full_output_path = self.write_extension(plt_figure)
            self.show_extension(self.full_output_path) if self.debug else None

    def write_extension(self, plt_figure: plt.figure) -> str:
        output_path = get_config(self.cfg, 'execution.artifact_path')
        return save_image(plt_figure, output_path, self.extension_name, ImageFormat.PNG)

    def show_extension(self, image_path: str):
        if self.debug:
            show_image(image_path)
