from ..sim import Simulation
from ..state import SimulationState
from .ForestConfig import ForestConfig


class ForestGenerator:
    """Generate plant distributions using the Simulation logic."""

    def __init__(
        self, size: tuple[float, float], species: dict[str, set], terrain=None
    ):
        self._sim = Simulation(size, species)
        if (
            terrain is not None
            and getattr(terrain, "moisture", None) is not None
        ):
            from ..Terrain import TerrainViabilityMap

            tvm = TerrainViabilityMap(terrain.moisture, terrain.resolution)
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
