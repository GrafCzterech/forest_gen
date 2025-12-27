import numpy as np

from .microrelief_strategy import MicroreliefStrategy


class NoneMicrorelief(MicroreliefStrategy):
    """
    Concrete Strategy implementing a no-op microrelief.

    Returns the input heightmap unchanged.
    """

    def apply(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Return the heightmap without modification.

        :param heightmap: Base terrain heightmap.
        :type heightmap: numpy.ndarray
        :return: Unmodified heightmap.
        :rtype: numpy.ndarray
        """
        return heightmap
