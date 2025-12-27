from collections.abc import Mapping
from typing import Callable

import numpy as np

from ..asset_dist import Species
from ..terrain import Terrain
from .forest_generator import ForestGenerator


class ForestBuilder:
    """
    Fluent builder for constructing a :class:`ForestGenerator`.
    """

    def __init__(self):
        self._species: dict[str, set[Species]] = {}
        self._size: tuple[float, float] = (100.0, 100.0)
        self._terrain = None
        self._terrain_layers = None
        self._layer_combiner = None

    def with_size(self, size: tuple[float, float]) -> "ForestBuilder":
        """
        Set the simulation area size.

        :param size: Simulation area ``(width, height)``.
        :type size: tuple[float, float]
        :return: Builder instance.
        :rtype: ForestBuilder
        """
        self._size = size
        return self

    def add_species(self, kind: str, species: Species) -> "ForestBuilder":
        """
        Register a species under a category.

        :param kind: Species group identifier.
        :type kind: str
        :param species: Species specification.
        :type species: Species
        :return: Builder instance.
        :rtype: ForestBuilder
        """
        self._species.setdefault(kind, set()).add(species)
        return self

    def with_terrain(self, terrain: Terrain) -> "ForestBuilder":
        """
        Attach a terrain used for viability sampling.

        :param terrain: Terrain providing derived layers (e.g. slope, moisture).
        :type terrain: Terrain
        :return: Builder instance.
        :rtype: ForestBuilder
        """
        self._terrain = terrain
        return self

    def with_terrain_viability_layers(
        self,
        layers: Mapping[str, np.ndarray],
        combine: Callable[[Mapping[str, float]], float] | None = None,
    ) -> "ForestBuilder":
        """
        Provide additional terrain viability layers.

        :param layers: Named raster layers sampled for viability.
        :type layers: Mapping[str, numpy.ndarray]
        :param combine: Optional layer-combination function.
        :type combine: Callable[[Mapping[str, float]], float] or None
        :return: Builder instance.
        :rtype: ForestBuilder
        """
        self._terrain_layers = layers
        self._layer_combiner = combine
        return self

    def build(self):
        """
        Construct the configured forest generator.

        :return: Forest generator instance.
        :rtype: ForestGenerator
        """
        return ForestGenerator(
            self._size,
            self._species,
            self._terrain,
            self._terrain_layers,
            self._layer_combiner,
        )
