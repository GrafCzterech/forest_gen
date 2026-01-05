from collections.abc import Mapping
from typing import Callable

import numpy as np

from ..asset_dist import (
    DistributionBuilder,
    DistributionConfig,
    SimulationState,
    Species,
)
from ..terrain import Terrain
from .forest_config import ForestConfig


class ForestGenerator:
    """
    High-level generator for forest plant distributions.

    Wraps the distribution simulation pipeline and optionally integrates
    terrain-derived viability layers.
    """

    def __init__(
        self,
        size: tuple[float, float],
        species: dict[str, set[Species]],
        terrain: Terrain | None = None,
        terrain_layers: Mapping[str, np.ndarray] | None = None,
        layer_combiner: Callable[[Mapping[str, float]], float] | None = None,
    ):
        """
        Initialize the forest generator.

        :param size: Simulation area size ``(width, height)``.
        :type size: tuple[float, float]
        :param species: Species grouped by category.
        :type species: dict[str, set[Species]]
        :param terrain: Optional terrain providing viability layers.
        :type terrain: Terrain or None
        :param terrain_layers: Optional additional terrain layers.
        :type terrain_layers: Mapping[str, numpy.ndarray] or None
        :param layer_combiner: Optional function combining terrain layer samples.
        :type layer_combiner: Callable[[Mapping[str, float]], float] or None
        """
        builder = DistributionBuilder().with_size(size)
        for kind, typed_species in species.items():
            for sp in typed_species:
                builder.add_species(kind, sp)

        if terrain is not None:

            available_layers = {
                "moisture": terrain.moisture,
                **(terrain_layers or {}),
            }

            if terrain.slope is not None:
                max_slope = float(np.max(terrain.slope)) or 1.0
                available_layers["slope_viability"] = 1.0 - np.clip(terrain.slope / max_slope, 0.0, 1.0)
            filtered_layers = {
                name: layer
                for name, layer in available_layers.items()
                if layer is not None
            }

            if filtered_layers:
                builder.with_terrain_viability_layers(
                    filtered_layers, terrain.config.resolution, combine=layer_combiner
                )
        self._generator = builder.build()

    def generate(self, config: ForestConfig) -> SimulationState:
        """
        Generate a forest distribution.

        :param config: Forest generation configuration.
        :type config: ForestConfig
        :return: Resulting simulation state.
        :rtype: SimulationState
        """
        distribution_cfg = DistributionConfig(
            scene_density=config.scene_density,
            years=config.years,
        )
        return self._generator.generate(distribution_cfg)
