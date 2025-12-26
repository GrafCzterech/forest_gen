from typing import Literal

from .export_strategy import ExportStrategy
from .glb_exporter import GLBExporter
from .png_exporter import PNGExporter


class ExportFactory:
    """
    Factory for creating :class:`ExportStrategy` instances.
    """

    @staticmethod
    def create(fmt: Literal["glb", "png", "image"], **kwargs) -> ExportStrategy:
        """
        Create an export strategy by format identifier.

        :param fmt: Export format identifier.
        :type fmt: Literal["glb", "png", "image"]
        :param kwargs: Keyword arguments forwarded to the exporter constructor.
        :type kwargs: dict
        :return: Export strategy instance.
        :rtype: ExportStrategy
        :raises ValueError: If the format is unknown.
        """

        if fmt == "glb":
            return GLBExporter(**kwargs)
        elif fmt in ("png", "image"):
            return PNGExporter(**kwargs)
        else:
            raise ValueError(f"Unknown export format: {fmt}")
