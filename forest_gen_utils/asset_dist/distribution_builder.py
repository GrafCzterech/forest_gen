from collections.abc import Mapping
from typing import Callable

import numpy as np

from .definitions import Species
from .distribution_generator import DistributionGenerator
from .sim import Simulation
from .terrain_viability import TerrainViabilityMap


class DistributionBuilder:
    """
    Fluent builder for configuring a plant distribution simulation.

    """

    def __init__(self):
        self._species: dict[str, set[Species]] = {}
        self._size: tuple[float, float] = (100.0, 100.0)
        self._terrain_layers: Mapping[str, np.ndarray] | None = None
        self._layer_combiner: Callable[[Mapping[str, float]], float] | None = None
        self._layer_resolution: float | None = None
        self._max_population: int | None = None

    def with_size(self, size: tuple[float, float]) -> "DistributionBuilder":
        """
        Set the simulation area size.

        :param size: Simulation area ``(width, height)``.
        :type size: tuple[float, float]
        :return: Builder instance.
        :rtype: DistributionBuilder
        """
        self._size = size
        return self

    def with_max_population(self, max_population: int | None) -> "DistributionBuilder":
        """
        Set a global population cap.

        :param max_population: Maximum allowed population or ``None``.
        :type max_population: int or None
        :return: Builder instance.
        :rtype: DistributionBuilder
        """
        self._max_population = max_population
        return self

    def add_species(self, kind: str, species: Species) -> "DistributionBuilder":
        """
        Register a species.

        :param kind: Species group identifier.
        :type kind: str
        :param species: Species specification.
        :type species: Species
        :return: Builder instance.
        :rtype: DistributionBuilder
        """
        self._species.setdefault(kind, set()).add(species)
        return self

    def with_terrain_viability_layers(
        self,
        layers: Mapping[str, np.ndarray],
        resolution: float,
        combine: Callable[[Mapping[str, float]], float] | None = None,
    ) -> "DistributionBuilder":
        """
        Apply terrain-based viability modifiers.

        :param layers: Named terrain layers sampled for viability.
        :type layers: Mapping[str, numpy.ndarray]
        :param resolution: Spatial resolution of the layers.
        :type resolution: float
        :param combine: Optional layer-combination function.
        :type combine: Callable[[Mapping[str, float]], float] or None
        :return: Builder instance.
        :rtype: DistributionBuilder
        """
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
        """
        Construct the distribution generator.

        :return: Configured distribution generator.
        :rtype: DistributionGenerator
        """
        self._apply_viability_layers()
        simulation = Simulation(self._size, self._species)
        return DistributionGenerator(simulation, max_population=self._max_population)
