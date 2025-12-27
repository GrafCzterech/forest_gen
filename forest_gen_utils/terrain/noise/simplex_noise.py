from .noise_strategy import NoiseStrategy
from ..terrain_config import TerrainConfig
import numpy as np
import opensimplex as osx


class SimplexNoise(NoiseStrategy):
    """
    Concrete Strategy implementing OpenSimplex noise generation.


    .. note::
        This implementation depends on the external ``opensimplex`` package.
        Noise generation is deterministic with respect to the provided seed.

    :ivar seed: Seed used to initialize the OpenSimplex noise generator.
                If ``None``, a default seed of ``0`` is used.
    :vartype seed: int or None

    """

    def __init__(self, seed: int | None = None):
        self.seed = seed

    def generate(self, config: TerrainConfig) -> np.ndarray:
        """
        Generate a normalized heightmap using OpenSimplex noise.

        This method implements the algorithm defined by the Strategy
        interface. Noise is sampled at each grid cell using a frequency
        derived from the terrain scale and resolution, then normalized
        linearly to the range ``[0.0, 1.0]``.

        :type config: TerrainConfig
        :return: A 2D NumPy array representing the generated heightmap.
        :rtype: numpy.ndarray

        :raises ZeroDivisionError: If ``config.scale`` is zero.
        """
         
        osx.seed(self.seed or 0)
        rows, cols = config.rows, config.cols
        heightmap = np.zeros((rows, cols), dtype=np.float32)
        freq = (1.0 / config.scale) * config.resolution

        for i in range(rows):
            for j in range(cols):
                heightmap[i, j] = osx.noise2(
                    i * freq,
                    j * freq,
                )

        min_h, max_h = heightmap.min(), heightmap.max()
        heightmap = (heightmap - min_h) / (max_h - min_h)
        return heightmap
