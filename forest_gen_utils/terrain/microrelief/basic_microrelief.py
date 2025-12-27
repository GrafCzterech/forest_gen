import numpy as np
from scipy.ndimage import gaussian_filter

from .microrelief_strategy import MicroreliefStrategy


class BasicMicrorelief(MicroreliefStrategy):
    """
    Concrete Strategy applying small-scale Gaussian-filtered noise.

    Adds subtle, high-frequency variation to a heightmap and clamps
    the result to ``[0.0, 1.0]``.
    """

    def __init__(self, strength: float = 0.001, sigma: float = 0.8):
        self.strength = strength
        self.sigma = sigma

    def apply(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian-smoothed microrelief noise.

        :param heightmap: Base terrain heightmap.
        :type heightmap: numpy.ndarray
        :return: Heightmap with microrelief applied.
        :rtype: numpy.ndarray
        """
        rows, cols = heightmap.shape
        micro = np.random.randn(rows, cols)
        micro = gaussian_filter(micro, sigma=self.sigma, mode="wrap")
        micro -= micro.mean()
        micro /= micro.std() + 1e-8
        micro *= self.strength
        return np.clip(heightmap + micro, 0.0, 1.0)
