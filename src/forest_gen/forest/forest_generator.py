from collections.abc import Mapping
from typing import Callable

import numpy as np


from ..asset_dist import DistributionBuilder, DistributionConfig, SimulationState, Species
from .forest_config import ForestConfig
from ..terrain import Terrain


class ForestGenerator:
    """Generate plant distributions using the Simulation logic."""

    def __init__(
        self,
        size: tuple[float, float],
        species: dict[str, set[Species]],
        terrain: Terrain | None = None,
        terrain_layers: Mapping[str, np.ndarray] | None = None,
        layer_combiner: Callable[[Mapping[str, float]], float] | None = None,
    ):
        builder = DistributionBuilder().with_size(size)
        for kind, typed_species in species.items():
            for sp in typed_species:
                builder.add_species(kind, sp)

        if terrain is not None:

            available_layers = {
                "moisture": terrain.moisture,
                "slope": terrain.slope,
                "aspect": terrain.aspect,
                **(terrain_layers or {}),
            }
            filtered_layers = {
                name: layer for name, layer in available_layers.items() if layer is not None
            }
            
            if filtered_layers:
                builder.with_terrain_viability_layers(                    
                    filtered_layers, terrain.config.resolution, combine=layer_combiner
                )
        self._generator = builder.build()

    def generate(self, config: ForestConfig) -> SimulationState:
        distribution_cfg = DistributionConfig(
                    scene_density=config.scene_density,
                    years=config.years,
                )
        return self._generator.generate(distribution_cfg)
