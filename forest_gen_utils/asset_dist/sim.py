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

        species_list = sorted(
            (sp for type_species in self.species.values() for sp in type_species),
            key=lambda sp: sp.radius,
            reverse=True,
        )

        def _has_conflict(point: tuple[float, float], radius: float) -> bool:
            for plant in instances:
                req = max(radius, plant.species.radius)  # your current rule
                dx = point[0] - plant.coords[0]
                dy = point[1] - plant.coords[1]
                if (dx * dx + dy * dy) < (req * req):
                    return True
            return False

        area = self.size[0] * self.size[1]
        rng = random.Random(0)  # deterministic within a run; optional

        for sp in species_list:
            desired_n = scene_density * sp.species_density * area
            target = math.floor(desired_n)
            if target <= 0:
                continue

            disk = PoissonDisk(
                2,
                radius=sp.radius,
                l_bounds=(0, 0),
                u_bounds=self.size,
            )

            accepted = 0
            # oversample to compensate for viability rejection
            oversample = 3
            max_rounds = 5  # bounded

            for _ in range(max_rounds):
                if accepted >= target:
                    break

                need = (target - accepted) * oversample
                points = disk.random(int(need)).tolist()
                disk.reset()

                for point in points:
                    if accepted >= target:
                        break
                    coords = (point[0], point[1])

                    if _has_conflict(coords, sp.radius):
                        continue

                    v = float(sp.viability_map(*coords))
                    v = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
                    if rng.random() > v:
                        continue

                    instances.append(Plant(coords, sp, 0))
                    accepted += 1

        return SimulationState(instances, self.size)
