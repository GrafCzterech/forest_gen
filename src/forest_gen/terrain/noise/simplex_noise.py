from .noise_strategy import NoiseStrategy
from ..terrain_config import TerrainConfig
import numpy as np
import opensimplex as osx


class SimplexNoise(NoiseStrategy):
    """
    OpenSimplex noise implementation.
    Requires `opensimplex` package.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self, config: TerrainConfig) -> np.ndarray:
        osx.seed(self.seed or 0)
        rows, cols = config.rows, config.cols
        heightmap = np.zeros((rows, cols), dtype=np.float32)
        freq = 1.0 / config.scale

        for i in range(rows):
            for j in range(cols):
                heightmap[i, j] = osx.noise2(
                    i * freq * config.resolution, j * freq * config.resolution
                )

        min_h, max_h = heightmap.min(), heightmap.max()
        heightmap = (heightmap - min_h) / (max_h - min_h + 1e-8)
        return heightmap
