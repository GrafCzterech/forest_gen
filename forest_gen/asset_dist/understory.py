from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

import numpy as np
from opensimplex import OpenSimplex

from .definitions import Species
from .state import SimulationState
from ..forest import ForestBuilder, ForestConfig
from ..terrain import Terrain

"""
Understory plant distribution utilities.

Implements spatial viability maps and a distributor that places
understory vegetation using the shared forest simulation pipeline.
"""

class PatchyUnderstoryMap:
    """
    Binary patchiness mask for understory vegetation.

    Uses simplex noise to avoid uniform understory carpets.
    """

    def __init__(
        self,
        scale: float = 0.1,
        threshold: float = 0.35,
        seed: int | None = None,
    ):
        self.scale = scale
        self.threshold = threshold
        self.noise = OpenSimplex(seed if seed is not None else random.randint(0, 10_000))

    def __call__(self, x: float, y: float) -> float:
        """
        Evaluate patch presence at world coordinates.

        :return: ``1.0`` if location is inside a patch, otherwise ``0.0``.
        :rtype: float
        """
        return 1.0 if self.noise.noise2(x * self.scale, y * self.scale) > self.threshold else 0.0


class CanopyShadeMap:
    """
    Viability mask favoring locations near canopy trees.

    Suppresses growth near trunks and attenuates viability with distance.
    """
    def __init__(
        self,
        canopy_positions: Iterable[tuple[float, float]],
        preferred_distance: float = 2.5,
        avoid_radius: float = 0.75,
        falloff_radius: float = 8.0,
    ):
        self.canopy_positions = tuple(canopy_positions)
        self.preferred_distance = preferred_distance
        self.avoid_radius = avoid_radius
        self.falloff_radius = max(falloff_radius, preferred_distance)

    def __call__(self, x: float, y: float) -> float:
        """
        Evaluate canopy-shade viability at world coordinates.

        :return: Viability multiplier in ``[0.0, 1.0]``.
        :rtype: float
        """
        if not self.canopy_positions:
            return 1.0

        nearest = min(math.dist((x, y), canopy) for canopy in self.canopy_positions)
        if nearest <= self.avoid_radius:
            return 0.0

        peak = max(self.preferred_distance, self.avoid_radius)
        spread = max(self.falloff_radius - peak, 1e-6)
        gaussian = math.exp(-((nearest - peak) ** 2) / (2 * (0.45 * spread) ** 2))

        tail = max(0.0, 1.0 - max(0.0, nearest - self.falloff_radius) / spread)
        return max(0.0, min(1.0, gaussian * tail))


class UnderstoryDistributor:
    """
    Distribute understory plants using the forest simulation pipeline.
    """
    def __init__(
        self,
        terrain: Terrain,
        canopy_positions: Iterable[tuple[float, float]] | None = None,
        *,
        preferred_distance: float = 4.0,
        avoid_radius: float = 1.5,
        falloff_radius: float = 9.0,
        patch_scale: float = 0.12,
        patch_threshold: float = 0.45,
        species_density: float = 0.035,
        reproduction_rate: int = 1,
        reproduction_radius: float = 4.5,
        radius: float = 1.8,
        max_age: int = 35,
    ):
        self.terrain = terrain
        self.canopy_map = CanopyShadeMap(
            canopy_positions or [],
            preferred_distance=preferred_distance,
            avoid_radius=avoid_radius,
            falloff_radius=falloff_radius,
        )
        self.patchiness = PatchyUnderstoryMap(patch_scale, threshold=patch_threshold)
        self.species_density = species_density
        self.reproduction_rate = reproduction_rate
        self.reproduction_radius = reproduction_radius
        self.radius = radius
        self.max_age = max_age

    def _terrain_layers(self) -> Mapping[str, np.ndarray]:
        layers: dict[str, np.ndarray] = {}
        if self.terrain.moisture is not None:
            layers["moisture"] = self.terrain.moisture

        if self.terrain.slope is not None:
            max_slope = float(np.max(self.terrain.slope))
            if max_slope > 0:
                layers["slope_viability"] = 1.0 - np.clip(
                    self.terrain.slope / max_slope, 0.0, 1.0
                )

        return layers

    def _combine_layers(self, values: Mapping[str, float]) -> float:
        result = 1.0
        for value in values.values():
            result *= value
        return result

    def _understory_species(self) -> Species:
        def viability(x: float, y: float) -> float:
            return self.patchiness(x, y) * self.canopy_map(x, y)

        return Species(
            "Understory",
            self.max_age,
            species_density=self.species_density,
            reproduction_rate=self.reproduction_rate,
            reproduction_radius=self.reproduction_radius,
            radius=self.radius,
            viability_map=viability,
        )

    def generate(self, config: ForestConfig) -> SimulationState:
        """
        Generate understory vegetation for the given forest configuration.

        :param config: Forest generation configuration.
        :type config: ForestConfig
        :return: Resulting simulation state.
        :rtype: SimulationState
        """
        builder = (
            ForestBuilder()
            .with_size((self.terrain.config.size, self.terrain.config.size))
            .with_terrain(self.terrain)
            .with_terrain_viability_layers(
                self._terrain_layers(), combine=self._combine_layers
            )
            .add_species("understory", self._understory_species())
        )

        forest = builder.build()
        return forest.generate(config)