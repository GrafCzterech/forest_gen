import numpy as np
from .MicroreliefStrategy import MicroreliefStrategy


class NoneMicrorelief(MicroreliefStrategy):
    """No-op microrelief (returns heightmap unchanged)."""

    def apply(self, heightmap: np.ndarray) -> np.ndarray:
        """Apply microrelief to a heightmap."""
        return heightmap

