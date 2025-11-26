from collections.abc import Mapping
from typing import Callable

import numpy as np

from .forest_generator import ForestGenerator
from ..terrain import Terrain
from ..asset_dist import Species


class ForestBuilder:
    """Fluent builder for constructing a ForestGenerator."""

    def __init__(self):
        self._species: dict[str, set[Species]] = {}
        self._size: tuple[float, float] = (100.0, 100.0)
        self._terrain = None
        self._terrain_layers = None
        self._layer_combiner = None

    def with_size(self, size: tuple[float, float]) -> "ForestBuilder":
        self._size = size
        return self

    def add_species(self, kind: str, species: Species) -> "ForestBuilder":
        self._species.setdefault(kind, set()).add(species)
        return self

    def with_terrain(self, terrain: Terrain) -> "ForestBuilder":
        self._terrain = terrain
        return self

    def with_terrain_viability_layers(
        self,
        layers: Mapping[str, np.ndarray],
        combine: Callable[[Mapping[str, float]], float] | None = None,
    ) -> "ForestBuilder":

        self._terrain_layers = layers
        self._layer_combiner = combine
        return self


    def build(self):

        return ForestGenerator(
            self._size,
            self._species,
            self._terrain,
            self._terrain_layers,
            self._layer_combiner,
        )