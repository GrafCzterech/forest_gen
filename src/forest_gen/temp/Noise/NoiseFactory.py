
from .FractalNoise import FractalNoise
from .NoiseStrategy import NoiseStrategy
from .SimplexNoise import SimplexNoise


class NoiseFactory:
    """
    Factory for creating NoiseStrategy instances by name.
    """
    @staticmethod
    def create(name: str) -> NoiseStrategy:
        """Return a NoiseStrategy matching the given name."""
        key = name.lower()
        if key == "fractal":
            return FractalNoise()
        elif key == "simplex":
            return SimplexNoise()
        else:
            raise ValueError(f"Unknown NoiseStrategy: {name}")

