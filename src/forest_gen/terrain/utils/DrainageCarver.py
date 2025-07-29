import numpy as np
from scipy.ndimage import gaussian_filter

class DrainageCarver:
    """Applies valley carving based on flow accumulation."""
    def __init__(self, strength: float = 0.3, sigma: float = 1.5):
        self.strength = strength
        self.sigma = sigma

    def apply(self, heightmap: np.ndarray, flow: np.ndarray) -> np.ndarray:
        flow_norm = (flow - flow.min()) / (np.ptp(flow) + 1e-8)
        carved = heightmap * (1.0 - flow_norm * self.strength)
        carved = gaussian_filter(carved, sigma=self.sigma, mode='wrap')
        return np.clip(carved, 0.0, 1.0)
