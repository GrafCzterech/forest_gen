from abc import ABC, abstractmethod
from ..TerrainConfig import TerrainConfig
import numpy as np


class NoiseStrategy(ABC):
    """
    Interface for noise generation strategies.
    """

    @abstractmethod
    def generate(self, config: TerrainConfig) -> np.ndarray:
        """Generate a heightmap given the terrain config."""
        pass
