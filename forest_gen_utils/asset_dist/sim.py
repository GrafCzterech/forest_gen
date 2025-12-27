import math
import random

from scipy.stats.qmc import PoissonDisk

from .definitions import Plant, Species
from .state import SimulationState

# this file mainly implments the initial state of the simulation


class Simulation:
    """
    Forest simulation initializer.

    Generates an initial plant distribution based on species parameters
    and scene density, producing a populated :class:`SimulationState`.
    """

    def __init__(self, size: tuple[float, float], species: dict[str, set[Species]]):
        """
        Initialize the simulation definition.

        :param size: Simulation area size ``(width, height)``.
        :type size: tuple[float, float]
        :param species: Species grouped by category.
        :type species: dict[str, set[Species]]
        """
        self.size = size
        self.species = species

    def new_state(
        self,
        scene_density: float,
    ) -> SimulationState:
        """
        Create a new initial simulation state.

        Plants are placed using Poisson disk sampling per species,
        scaled by scene density and species-specific target density.
        Larger-radius species are placed first to reduce overlap.

        :param scene_density: Global density multiplier.
        :type scene_density: float
        :return: Initialized simulation state.
        :rtype: SimulationState
        """
        instances: list[Plant] = []
        # species_list = [
        #     sp for type_species in self.species.values() for sp in type_species
        # ]
        # ns = [
        #     scene_density * sp.species_density * self.size[0] * self.size[1]
        #     for sp in species_list
        # ]

        # total_n = sum(ns)
        # points: list[list[float]] = self.disk.random(int(total_n)).tolist()
        # self.disk.reset()
        # random.shuffle(points)
        # tot_n = len(points) / total_n if total_n else 0

        # i = 0
        # for sp, n_val in zip(species_list, ns):
        #     n = math.floor(n_val * tot_n)
        #     for _ in range(n):
        #         point = points[i]
        #         i += 1
        #         instances.append(Plant((point[0], point[1]), sp, 0))

        species_list = sorted(
            (sp for type_species in self.species.values() for sp in type_species),
            key=lambda sp: sp.radius,
            reverse=True,
        )

        def _has_conflict(point: tuple[float, float], radius: float) -> bool:
            for plant in instances:
                max_radius = max(radius, plant.species.radius)
                dx = point[0] - plant.coords[0]
                dy = point[1] - plant.coords[1]
                if (dx**2 + dy**2) < (max_radius**2):
                    return True
            return False

        area = self.size[0] * self.size[1]

        for sp in species_list:
            desired_n = scene_density * sp.species_density * area
            n = math.floor(desired_n)

            if n <= 0:
                continue

            disk = PoissonDisk(
                2,
                radius=sp.radius,
                l_bounds=(0, 0),
                u_bounds=self.size,
            )

            points = disk.random(n).tolist()
            disk.reset()

            for point in points:
                coords = (point[0], point[1])
                if not _has_conflict(coords, sp.radius):
                    instances.append(Plant(coords, sp, 0))

        # simulation initialized

        return SimulationState(instances, self.size)
