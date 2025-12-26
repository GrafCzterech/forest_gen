import numpy as np
from PIL import Image
from .export_strategy import ExportStrategy


class PNGExporter(ExportStrategy):
    """
    Concrete Strategy exporting a heightmap as a grayscale PNG image.
    """

    def __init__(self, max_elevation: float = 100.0) -> None:
        self.max_elevation = max_elevation

    def export(self, heightmap: np.ndarray, path: str) -> None:
        """
        Export a heightmap to a PNG file.

        :param heightmap: Heightmap array with values in ``[0.0, 1.0]``.
        :type heightmap: numpy.ndarray
        :param path: Output file path.
        :type path: str
        """
        img_arr = (heightmap * 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="L")
        img.save(path)
