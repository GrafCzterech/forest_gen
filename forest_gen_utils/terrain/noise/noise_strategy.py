from abc import ABC, abstractmethod

import numpy as np

from ..terrain_config import TerrainConfig


class NoiseStrategy(ABC):
    """
    Abstract base class for terrain noise generation strategies.
    This class defines the **Strategy** in the *Strategy Design Pattern*.

    This interface defines the contract for all noise generators used
    to produce terrain heightmaps. Implementations must provide a
    concrete realization of the :meth:`generate` method.
    """

    @abstractmethod
    def generate(self, config: TerrainConfig) -> np.ndarray:
        """
        Generate a heightmap based on the provided terrain configuration.

        Implementations are expected to return a 2D array of floating-point
        values representing terrain elevation. The interpretation and
        normalization of the output are left to the concrete strategy.

        :param config: Terrain configuration defining grid dimensions,
                       scale, and resolution.
        :type config: TerrainConfig
        :return: A 2D array representing the generated heightmap.
        :rtype: numpy.ndarray
        """
        pass
