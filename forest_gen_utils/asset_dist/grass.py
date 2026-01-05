from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

import numpy as np
from opensimplex import OpenSimplex
from scipy.stats.qmc import PoissonDisk

from ..forest import ForestBuilder, ForestConfig
from ..terrain import Terrain
from .definitions import Species
from .state import SimulationState

"""
Grass distribution utilities built on the forest simulation pipeline.

Provides spatial viability maps and a distributor that reuses the
existing Simulation / ForestBuilder infrastructure.
"""


class PatchyGrassMap:
    """Soft patchiness mask for Grass placement. Look at PatchyUnderstoryMap for more information."""

    def __init__(self, scale: float = 0.2, seed: int | None = None, gamma: float = 0.7):
        self.scale = scale
        self.gamma = gamma
        self.noise = OpenSimplex(seed if seed is not None else random.randint(0, 10_000))

    def __call__(self, x: float, y: float) -> float:
        v = self.noise.noise2(x * self.scale, y * self.scale) * 0.5 + 0.5  # [0,1]
        v = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
        return float(v ** self.gamma)

class TreeProximityMap:
    """
    Viability mask attenuating grass near trees.
    """

    def __init__(
        self,
        tree_positions: Iterable[tuple[float, float]],
        hard_radius: float = 2.0,
        falloff_radius: float = 6.0,
    ):
        self.tree_positions = tuple(tree_positions)
        self.hard_radius = hard_radius
        self.falloff_radius = max(falloff_radius, hard_radius)

    def __call__(self, x: float, y: float) -> float:
        """
        Evaluate proximity-based viability at world coordinates.

        :return: Viability multiplier in ``[0.0, 1.0]``.
        :rtype: float
        """
        if not self.tree_positions:
            return 1.0

        closest = min(math.dist((x, y), tree) for tree in self.tree_positions)
        if closest <= self.hard_radius:
            return 0.0
        if closest >= self.falloff_radius:
            return 1.0

        return (closest - self.hard_radius) / (self.falloff_radius - self.hard_radius)


class GrassDistributor:
    """
    Distribute grass using the standard forest simulation pipeline.
    """

    def __init__(
        self,
        terrain: Terrain,
        tree_positions: Iterable[tuple[float, float]] | None = None,
        patch_scale: float = 0.2,
        hard_radius: float = 2.0,
        falloff_radius: float = 6.0,
        *,
        max_age: int = 6,
        species_density: float = 0.35,
        reproduction_rate: int = 3,
        reproduction_radius: float = 2.5,
        radius: float = 0.6,
        patch_gamma: float = 0.7,
        species_floor: float = 0.15, 
        terrain_floor: float = 0.35, 
    ):
        self.terrain = terrain
        self.tree_map = TreeProximityMap(
            tree_positions or [], hard_radius, falloff_radius
        )
        self.patchiness = PatchyGrassMap(patch_scale, gamma=patch_gamma) 
        self.max_age = max_age
        self.species_density = species_density
        self.reproduction_rate = reproduction_rate
        self.reproduction_radius = reproduction_radius
        self.radius = radius
        
        self.species_floor = species_floor
        self.terrain_floor = terrain_floor

    def _terrain_layers(self) -> Mapping[str, np.ndarray]:
        """Build terrain viability layers emphasizing gentle slopes."""

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
        avg = sum(values.values()) / max(len(values), 1)
        avg = 0.0 if avg < 0.0 else 1.0 if avg > 1.0 else avg
        return float(self.terrain_floor + (1.0 - self.terrain_floor) * avg)

    def _grass_species(self) -> Species:
        def viability(x: float, y: float) -> float:
            base = self.patchiness(x, y) * self.tree_map(x, y)
            base = 0.0 if base < 0.0 else 1.0 if base > 1.0 else base
            return float(self.species_floor + (1.0 - self.species_floor) * base)

        return Species(
            "Grass",
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
        Generate grass distribution for the given forest configuration.

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
            .add_species("grass", self._grass_species())
        )

        forest = builder.build()
        return forest.generate(config)


# Legacy helpers kept for quick sampling in notebooks or other assets.
def grass_points(width: int, height: int, r: float) -> list[tuple[float, float]]:
    """
    Generate Poisson-distributed grass points.

    Intended for quick sampling or prototyping.
    """
    sampler = PoissonDisk(
        2,
        radius=r,
        ncandidates=30,
        l_bounds=[0, 0],
        u_bounds=[width, height],
    )
    points = sampler.random(n=int(width * height / (r * r))).tolist()
    sampler.reset()
    return [tuple(point) for point in points]


def remove_grass_near_tree(
    grass: list[tuple[float, float]], trees: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Filter grass points near trees using proximity masking.
    """
    proximity = TreeProximityMap(trees)
    return [point for point in grass if proximity(point[0], point[1]) > 0.0]
