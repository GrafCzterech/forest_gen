from ..sim import Simulation
from ..state import SimulationState
from .ForestConfig import ForestConfig


class ForestGenerator:
    """Generate plant distributions using the Simulation logic."""

    def __init__(self, size: tuple[float, float], species: dict[str, set]):
        self._sim = Simulation(size, species)

    def generate(self, config: ForestConfig) -> SimulationState:
        state = self._sim.new_state(config.scene_density)
        if config.years:
            state.run_state(config.years)
        return state
