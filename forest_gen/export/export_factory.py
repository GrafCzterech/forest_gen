from typing import Literal

from .export_strategy import ExportStrategy
from .glb_exporter import GLBExporter
from .png_exporter import PNGExporter


class ExportFactory:
    """
    Factory for creating ExportStrategy instances by format.
    """

    @staticmethod
    def create(fmt: Literal["glb", "png", "image"], **kwargs) -> ExportStrategy:
        """Return an ExportStrategy matching the given format."""
        if fmt == "glb":
            return GLBExporter(**kwargs)
        elif fmt in ("png", "image"):
            return PNGExporter(**kwargs)
        else:
            raise ValueError(f"Unknown export format: {fmt}")
