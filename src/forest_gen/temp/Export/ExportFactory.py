from .ExportStrategy import ExportStrategy
from .GLBExporter import GLBExporter
from .PNGExporter import PNGExporter


class ExportFactory:
    """
    Factory for creating ExportStrategy instances by format.
    """

    @staticmethod
    def create(fmt: str, **kwargs) -> ExportStrategy:
        """Return an ExportStrategy matching the given format."""
        key = fmt.lower()
        if key == 'glb':
            return GLBExporter(**kwargs)
        elif key in ('png', 'image'):
            return PNGExporter(**kwargs)
        else:
            raise ValueError(f"Unknown export format: {fmt}")

