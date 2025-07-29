from .MicroreliefStrategy import MicroreliefStrategy

from scipy.ndimage import gaussian_filter
import numpy as np


class BasicMicrorelief(MicroreliefStrategy):
    """Applies small-scale Gaussian-filtered noise."""

    def __init__(self, strength: float = 0.001, sigma: float = 0.8):
        self.strength = strength
        self.sigma = sigma

    def apply(self, heightmap: np.ndarray) -> np.ndarray:
        rows, cols = heightmap.shape
        micro = np.random.randn(rows, cols)
        micro = gaussian_filter(micro, sigma=self.sigma, mode="wrap")
        micro -= micro.mean()
        micro /= micro.std() + 1e-8
        micro *= self.strength
        return np.clip(heightmap + micro, 0.0, 1.0)
