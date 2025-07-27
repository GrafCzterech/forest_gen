import random
import math

from scipy.stats.qmc import PoissonDisk

from .definitions import Species, Plant
from .state import SimulationState

# this file mainly implments the initial state of the simulation


class Simulation:
    """A simulation of a forest used to generate realistic plant positions."""

    def __init__(
        self, size: tuple[float, float], species: dict[str, set[Species]]
    ):
        """Initialize the simulation.

        Args:
            size (int): Size of the simulation in meters.
            species (dict[str, set[Species]]): Species, categorized by type.
        """
        self.size = size
        self.species = species
        self.disk = PoissonDisk(
            2,
            radius=max(
                species.radius
                for type_species in species.values()
                for species in type_species
            ),
            l_bounds=(0, 0),
            u_bounds=size,
        )

    def new_state(
        self,
        scene_density: float,
    ) -> SimulationState:
        """Create a new simulation state.

        Args:
            scene_density (float): Density of the scene. Higher values mean more plants.

        Returns:
            SimulationState: A new simulation state.
        """
        instances: list[Plant] = []
        points: list[list[float]] = self.disk.fill_space().tolist()
        self.disk.reset()
        ns = [
            (
                scene_density
                * species.species_density
                * self.size[0]
                * self.size[1]
            )
            / len(type_species)
            for type_species in self.species.values()
            for species in type_species
        ]
        tot_n = min(len(points) / sum(ns), 1.0)
        for i, type_species in enumerate(self.species.values()):
            for species in type_species:
                n = math.floor(ns[i] * tot_n)
                for _ in range(n):
                    point = points.pop(random.randint(0, len(points) - 1))
                    instances.append(Plant((point[0], point[1]), species, 0))

        # simulation initialized

        return SimulationState(instances, self.size)
