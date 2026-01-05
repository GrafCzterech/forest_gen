from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

import numpy as np
from opensimplex import OpenSimplex

from ..forest import ForestBuilder, ForestConfig
from ..terrain import Terrain
from .definitions import Species
from .state import SimulationState

"""
Understory plant distribution utilities.

Implements spatial viability maps and a distributor that places
understory vegetation using the shared forest simulation pipeline.
"""


class PatchyUnderstoryMap:
    """
    Soft patchiness mask for understory placement driven by simplex noise.

    The map samples 2D simplex noise, remaps it from ``[-1, 1]`` to ``[0, 1]``,
    applies a hard cutoff at ``threshold``, then rescales the remaining range to
    ``(0, 1]`` and shapes it using a power curve (``gamma``).

    This produces clustered “patches” of understory rather than a uniform carpet
    """
    def __init__(self, scale: float = 0.1, threshold: float = 0.35, seed: int | None = None, gamma: float = 0.8):
        """
        Initialize the patchiness map.

        :param scale: Spatial frequency multiplier applied to world coordinates
            before sampling noise.
        :type scale: float
        :param threshold: Cutoff applied after remapping noise into ``[0, 1]``.
            Values at or below this threshold return ``0.0``.
        :type threshold: float
        :param seed: Optional seed for the simplex noise generator. If ``None``,
            a random seed is chosen.
        :type seed: int or None
        :param gamma: Power curve applied to the normalized patch strength above
            the threshold. ``gamma < 1`` boosts patch strength; ``gamma > 1``
            suppresses it.
        :type gamma: float
        """
        self.scale = scale
        self.threshold = threshold 
        self.gamma = gamma
        self.noise = OpenSimplex(seed if seed is not None else random.randint(0, 10_000))

    def __call__(self, x: float, y: float) -> float:
        """
        Evaluate patch viability at world coordinates.

        :param x: World-space x coordinate.
        :type x: float
        :param y: World-space y coordinate.
        :type y: float
        :return: Patch strength in ``[0.0, 1.0]`` (0 means no patch).
        :rtype: float
        """
        v = self.noise.noise2(x * self.scale, y * self.scale) * 0.5 + 0.5
        v = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

        t = self.threshold
        if v <= t:
            return 0.0
        v = (v - t) / max(1.0 - t, 1e-9)

        return float(v ** self.gamma)


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
        terrain_floor = 0.30

        avg = sum(values.values()) / max(len(values), 1)
        avg = 0.0 if avg < 0.0 else 1.0 if avg > 1.0 else avg
        return float(terrain_floor + (1.0 - terrain_floor) * avg)

    def _understory_species(self) -> Species:
        species_floor = 0.10  # try 0.10..0.25

        def viability(x: float, y: float) -> float:
            base = self.patchiness(x, y) * self.canopy_map(x, y)
            base = 0.0 if base < 0.0 else 1.0 if base > 1.0 else base
            return float(species_floor + (1.0 - species_floor) * base)

        return Species(
            "Understory",
            self.max_age,
            species_density=self.species_density,
            reproduction_rate=self.reproduction_rate,
            reproduction_radius=self.reproduction_radius,
            radius=self.radius,
            viability_map=viability,
            juvenile_recovery_age=0.0,
            juvenile_mortality_depth=0.0,
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
