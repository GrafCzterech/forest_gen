from abc import ABC, abstractmethod
import numpy as np


class MicroreliefStrategy(ABC):
    """
    Interface for microrelief application.

    Methods:
        apply: Apply microrelief to a heightmap.
    """

    @abstractmethod
    def apply(self, heightmap: np.ndarray) -> np.ndarray:
        """Apply microrelief to a heightmap."""
        pass
