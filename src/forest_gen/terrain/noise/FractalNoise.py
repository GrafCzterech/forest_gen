import numpy as np
from scipy.ndimage import gaussian_filter
from .NoiseStrategy import NoiseStrategy
from ..TerrainConfig import TerrainConfig


class FractalNoise(NoiseStrategy):
    """Fractal Brownian Motion (fBm) noise implementation."""

    def __init__(
        self,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: int | None = None,
    ):
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed

    def generate(self, config: TerrainConfig) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)

        rows, cols = config.rows, config.cols
        heightmap = np.zeros((rows, cols), dtype=np.float32)
        freq, amp, total_amp = 1.0, 1.0, 0.0

        for _ in range(config.octaves):
            noise = np.random.rand(rows, cols)
            sigma = (rows + cols) / (config.scale * freq * 2.0)
            smooth = gaussian_filter(noise, sigma=sigma, mode="wrap")
            heightmap += smooth * amp  # type: ignore[operator]
            total_amp += amp
            amp *= self.persistence
            freq *= self.lacunarity

        # Normalize to [0,1]
        heightmap /= total_amp
        heightmap -= heightmap.min()
        heightmap /= heightmap.max() + 1e-8
        return heightmap
