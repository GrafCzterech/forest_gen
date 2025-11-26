from ..asset_dist import (
    Simulation,
    SimulationState,
    TerrainViabilityMap,
    Species,
)
from .forest_config import ForestConfig
from ..terrain import Terrain


class ForestGenerator:
    """Generate plant distributions using the Simulation logic."""

    def __init__(
        self,
        size: tuple[float, float],
        species: dict[str, set[Species]],
        terrain: Terrain | None = None,
    ):
        self._sim = Simulation(size, species)
        if terrain is not None and terrain.moisture is not None:

            tvm = TerrainViabilityMap(
                terrain.moisture, terrain.config.resolution
            )
            for type_species in species.values():
                for sp in type_species:
                    orig = sp.viability_map

                    def wrapped(x, y, orig=orig):
                        return orig(x, y) * tvm(x, y)

                    sp.viability_map = wrapped

    def generate(self, config: ForestConfig) -> SimulationState:
        state = self._sim.new_state(config.scene_density)
        if config.years:
            state.run_state(config.years)
        return state
