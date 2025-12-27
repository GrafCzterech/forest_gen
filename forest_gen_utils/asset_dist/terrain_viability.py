from collections.abc import Mapping
from typing import Callable

import numpy as np


class TerrainViabilityMap:
    """
    Callable terrain-based viability lookup.

    Samples one or more raster layers at world coordinates and combines
    the values into a single viability multiplier.
    """

    def __init__(
        self,
        data: np.ndarray | Mapping[str, np.ndarray],
        resolution: float,
        combine: Callable[[Mapping[str, float]], float] | None = None,
    ):
        """
        Initialize a terrain viability map.

        :param data: Single raster array or mapping of named raster layers.
                     All layers must share the same shape.
        :type data: numpy.ndarray or Mapping[str, numpy.ndarray]
        :param resolution: Spatial resolution of the raster grids.
        :type resolution: float
        :param combine: Optional function combining sampled layer values.
                        Defaults to multiplicative combination.
        :type combine: Callable[[Mapping[str, float]], float] or None
        """

        if isinstance(data, Mapping):
            self.layers = {name: np.asarray(layer) for name, layer in data.items()}
        else:
            self.layers = {"layer": np.asarray(data)}

        if not self.layers:
            raise ValueError("At least one viability layer must be provided")

        self._shape = next(iter(self.layers.values())).shape
        for name, layer in self.layers.items():
            if layer.shape != self._shape:
                raise ValueError(
                    f"Layer '{name}' has shape {layer.shape}, expected {self._shape}"
                )
        self.resolution = resolution
        self.combine = combine or self._default_combine

    @staticmethod
    def _default_combine(values: Mapping[str, float]) -> float:
        result = 1.0
        for value in values.values():
            result *= value
        return result

    def _in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self._shape[0] and 0 <= j < self._shape[1]

    def __call__(self, x: float, y: float) -> float:
        """
        Sample terrain viability at world coordinates.

        :param x: X coordinate in world units.
        :type x: float
        :param y: Y coordinate in world units.
        :type y: float
        :return: Combined viability value.
        :rtype: float
        """
        i = int(y / self.resolution)
        j = int(x / self.resolution)
        if not self._in_bounds(i, j):
            return 0.0

        sample = {name: float(layer[i, j]) for name, layer in self.layers.items()}
        return float(self.combine(sample))
