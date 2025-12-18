from collections.abc import Mapping
from typing import Callable

import numpy as np

from .definitions import Species
from .distribution_generator import DistributionGenerator
from .sim import Simulation
from .terrain_viability import TerrainViabilityMap


class DistributionBuilder:
    """Fluent builder mirroring the Terrain/Forest construction pattern."""

    def __init__(self):
        self._species: dict[str, set[Species]] = {}
        self._size: tuple[float, float] = (100.0, 100.0)
        self._terrain_layers: Mapping[str, np.ndarray] | None = None
        self._layer_combiner: Callable[[Mapping[str, float]], float] | None = None
        self._layer_resolution: float | None = None
        self._max_population: int | None = None

    def with_size(self, size: tuple[float, float]) -> "DistributionBuilder":
        self._size = size
        return self

    def with_max_population(self, max_population: int | None) -> "DistributionBuilder":
        self._max_population = max_population
        return self

    def add_species(self, kind: str, species: Species) -> "DistributionBuilder":
        self._species.setdefault(kind, set()).add(species)
        return self

    def with_terrain_viability_layers(
        self,
        layers: Mapping[str, np.ndarray],
        resolution: float,
        combine: Callable[[Mapping[str, float]], float] | None = None,
    ) -> "DistributionBuilder":
        self._terrain_layers = layers
        self._layer_combiner = combine
        self._layer_resolution = resolution
        return self

    def _apply_viability_layers(self) -> None:
        if not self._terrain_layers:
            return
        if self._layer_resolution is None:
            raise ValueError("Terrain viability layers require a resolution to sample")

        tvm = TerrainViabilityMap(
            self._terrain_layers, self._layer_resolution, combine=self._layer_combiner
        )
        for type_species in self._species.values():
            for sp in type_species:
                original_map = sp.viability_map

                def wrapped(x: float, y: float, *, _orig=original_map) -> float:
                    return _orig(x, y) * tvm(x, y)

                sp.viability_map = wrapped

    def build(self) -> DistributionGenerator:
        self._apply_viability_layers()
        simulation = Simulation(self._size, self._species)
        return DistributionGenerator(simulation, max_population=self._max_population)