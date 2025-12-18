from collections.abc import Mapping
from typing import Callable

import numpy as np


class TerrainViabilityMap:
    """Callable wrapper returning terrain-derived values for coordinates."""

    def __init__(
        self,
        data: np.ndarray | Mapping[str, np.ndarray],
        resolution: float,
        combine: Callable[[Mapping[str, float]], float] | None = None,
    ):
        """Create a terrain viability lookup.

        Args:
            data: Either a single raster array or a mapping from layer names to
                raster arrays. All arrays must share the same shape.
            resolution: The spatial resolution of the raster grids.
            combine: Optional callable receiving a mapping of sampled values
                and returning a single viability multiplier. Defaults to
                multiplying all layer values together.
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
        i = int(y / self.resolution)
        j = int(x / self.resolution)
        if not self._in_bounds(i, j):
             return 0.0
        
        sample = {name: float(layer[i, j]) for name, layer in self.layers.items()}
        return float(self.combine(sample))