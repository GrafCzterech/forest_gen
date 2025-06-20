import numpy as np
from PIL import Image
from .ExportStrategy import ExportStrategy


class PNGExporter(ExportStrategy):
    """
    Exports heightmap as a PNG grayscale image.
    """

    def __init__(self, max_elevation: float = 100.0) -> None:
        self.max_elevation = max_elevation

    def export(self, heightmap: np.ndarray, path: str) -> None:
        img_arr = (heightmap * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode='L')
        img.save(path)

