import numpy as np
from scipy.ndimage import gaussian_filter

from ..terrain_config import TerrainConfig
from .noise_strategy import NoiseStrategy


class FractalNoise(NoiseStrategy):
    """
    Concrete Strategy implementing fractal Brownian motion (fBm) noise.

    This class is a ConcreteStrategy in the Strategy pattern and can be
    used interchangeably with other :class:`NoiseStrategy` implementations.
    The generated heightmap is normalized to ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        seed: int | None = None,
    ):
        """
        :param persistence: Amplitude decay between octaves.
        :type persistence: float
        :param lacunarity: Frequency multiplier between octaves.
        :type lacunarity: float
        :param seed: Optional random seed.
        :type seed: int or None
        """
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed

    def generate(self, config: TerrainConfig) -> np.ndarray:
        """
        Generate a normalized heightmap using multi-octave fractal noise.

        :param config: Terrain configuration (size, scale, resolution,
                       and number of octaves).
        :type config: TerrainConfig
        :return: 2D heightmap array with values in ``[0.0, 1.0]``.
        :rtype: numpy.ndarray
        """

        rng = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()

        rows, cols = config.rows, config.cols
        heightmap = np.zeros((rows, cols), dtype=np.float32)
        freq, amp, total_amp = 1.0, 1.0, 0.0

        if getattr(config, "octaves", 0) <= 0:
                    return heightmap

        for _ in range(config.octaves):
            noise = rng.rand(rows, cols)
            sigma = (rows + cols) * config.resolution / (config.scale * freq * 2.0)
            smooth = gaussian_filter(noise, sigma=sigma, mode="wrap")
            heightmap += smooth * amp  # type: ignore[operator]
            total_amp += amp
            amp *= self.persistence
            freq *= self.lacunarity

        # Normalize to [0,1]
        if total_amp <= 1e-12:
            return np.zeros_like(heightmap, dtype=np.float32)

        heightmap /= total_amp
        heightmap -= heightmap.min()
        heightmap /= heightmap.max() + 1e-8
        return heightmap
