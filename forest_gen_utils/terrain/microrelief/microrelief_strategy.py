from abc import ABC, abstractmethod

import numpy as np


class MicroreliefStrategy(ABC):
    """
    Strategy interface for applying microrelief to terrain heightmaps.
    """

    @abstractmethod
    def apply(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Apply micro-scale terrain variation to a heightmap.

        :param heightmap: Base terrain heightmap.
        :type heightmap: numpy.ndarray
        :return: Heightmap with microrelief applied.
        :rtype: numpy.ndarray
        """
        pass
