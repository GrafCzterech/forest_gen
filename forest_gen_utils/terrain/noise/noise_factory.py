from typing import Literal

from .fractal_noise import FractalNoise
from .noise_strategy import NoiseStrategy
from .simplex_noise import SimplexNoise


class NoiseFactory:
    """
    Factory for creating :class:`NoiseStrategy` instances by name.

    Encapsulates strategy selection logic and decouples callers from
    concrete noise implementations.
    """

    @staticmethod
    def create(name: Literal["fractal", "simplex"]) -> NoiseStrategy:
        """
        Create a noise strategy by identifier.

        :param name: Strategy name (``"fractal"`` or ``"simplex"``).
        :type name: Literal["fractal", "simplex"]
        :return: Instantiated noise strategy.
        :rtype: NoiseStrategy
        :raises ValueError: If the strategy name is unknown.
        """
        key = name.lower()
        if key == "fractal":
            return FractalNoise()
        elif key == "simplex":
            return SimplexNoise()
        else:
            raise ValueError(f"Unknown NoiseStrategy: {name}")
