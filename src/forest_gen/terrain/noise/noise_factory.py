from .fractal_noise import FractalNoise
from .noise_strategy import NoiseStrategy
from .simplex_noise import SimplexNoise

from typing import Literal


class NoiseFactory:
    """
    Factory for creating NoiseStrategy instances by name.
    """

    @staticmethod
    def create(name: Literal["fractal", "simplex"]) -> NoiseStrategy:
        """Return a NoiseStrategy matching the given name."""
        key = name.lower()
        if key == "fractal":
            return FractalNoise()
        elif key == "simplex":
            return SimplexNoise()
        else:
            raise ValueError(f"Unknown NoiseStrategy: {name}")
