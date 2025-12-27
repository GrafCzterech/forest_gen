from __future__ import annotations

from .distribution_config import DistributionConfig
from .sim import Simulation
from .state import SimulationState


class DistributionGenerator:
    """Orchestrates Simulation creation using a fluent builder interface."""

    def __init__(self, simulation: Simulation, *, max_population: int | None = None):
        self._simulation = simulation
        self._max_population = max_population

    def generate(self, config: DistributionConfig) -> SimulationState:
        """
        Run the simulation using the given configuration.

        :param config: Distribution simulation configuration.
        :type config: DistributionConfig
        :return: Final simulation state.
        :rtype: SimulationState
        """
        state = self._simulation.new_state(config.scene_density)
        if config.years:
            max_population = config.max_population
            if max_population is None:
                max_population = self._max_population
            state.run_state(config.years, max_population=max_population)
        return state
