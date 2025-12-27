from abc import ABC, abstractmethod
import numpy as np


class ExportStrategy(ABC):
    """
    Strategy interface for exporting terrain data.

    Defines a common interface for exporting heightmaps to external
    representations (e.g. images, meshes).
    """
    @abstractmethod
    def export(self, heightmap: np.ndarray, path: str) -> None:
        """
        Export a heightmap to the specified path.

        :param heightmap: Heightmap array to export.
        :type heightmap: numpy.ndarray
        :param path: Output file path.
        :type path: str
        """      
        pass
