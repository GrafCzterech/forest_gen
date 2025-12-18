from abc import ABC, abstractmethod
import numpy as np


class ExportStrategy(ABC):
    """Interface for export strategies."""

    @abstractmethod
    def export(self, heightmap: np.ndarray, path: str) -> None:
        """Export heightmap to given file path."""
        pass
