import numpy as np
from scipy.ndimage import gaussian_filter


class DrainageCarver:
    """
    Apply valley carving based on flow accumulation.

    Regions with higher accumulated flow are lowered proportionally
    to simulate drainage channels.
    """

    def __init__(self, strength: float = 0.3, sigma: float = 1.5):
        self.strength = strength
        self.sigma = sigma

    def apply(self, heightmap: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Carve valleys into a heightmap using flow accumulation.

        :param heightmap: Base terrain heightmap.
        :type heightmap: numpy.ndarray
        :param flow: Flow accumulation array.
        :type flow: numpy.ndarray
        :return: Heightmap with drainage carving applied.
        :rtype: numpy.ndarray
        """
        flow_norm = (flow - flow.min()) / (np.ptp(flow) + 1e-8)
        carved = heightmap * (1.0 - flow_norm * self.strength)
        carved = gaussian_filter(carved, sigma=self.sigma, mode="wrap")
        return np.clip(carved, 0.0, 1.0)
