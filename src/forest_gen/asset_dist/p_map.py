import opensimplex


def normalize_noise(value: float) -> float:
    """Normalize the noise value to be between 0 and 1."""
    return (value + 1.0) / 2.0


class PMap:
    """A 2D probability map using Simplex noise."""

    def __init__(self, smoothness: float = 1.0, seed: int = 0):
        """Create a PMap object.

        Args:
            smoothness (float, optional): How smooth the PMap should be. Higher values equal small rate of change. Defaults to 1.0.
            seed (int, optional): The seed for the Simplex noise. Defaults to 0.
        """
        self.seed = seed
        self.smoothness = smoothness
        self.noise = opensimplex.OpenSimplex(seed=seed)

    def __call__(self, x: float, y: float) -> float:
        """Get the noise value at the given coordinates."""
        return normalize_noise(
            self.noise.noise2(x / self.smoothness, y / self.smoothness)
        )
