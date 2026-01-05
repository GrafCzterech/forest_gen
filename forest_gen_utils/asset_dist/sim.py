import math
import random

from scipy.stats import qmc
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

    def __init__(self, size: tuple[float, float], species: dict[str, set[Species]], seed: int | None = None):        
        """
        Initialize the simulation definition.

        :param size: Simulation area size ``(width, height)``.
        :type size: tuple[float, float]
        :param species: Species grouped by category.
        :type species: dict[str, set[Species]]
        """
        self.size = size
        self.species = species
        self.rng = random.Random(seed)
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
                req = max(radius, plant.species.radius)
                dx = point[0] - plant.coords[0]
                dy = point[1] - plant.coords[1]
                if (dx * dx + dy * dy) < (req * req):
                    return True
            return False

        def _clamp01(v: float) -> float:
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        width, height = float(self.size[0]), float(self.size[1])
        scale = max(width, height)
        area = width * height
        rng = self.rng  # <-- use seeded RNG from __init__

        oversample = 5
        rounds = 6
        min_batch = 1200  # key: forces PoissonDisk to “grow” across the domain

        for sp in species_list:
            target = math.floor(scene_density * sp.species_density * area)
            if target <= 0:
                continue

            radius_unit = max(float(sp.radius) / scale, 1e-9)

            accepted = 0
            for _ in range(rounds):
                if accepted >= target:
                    break

                need = max((target - accepted) * oversample, min_batch)
                disk = PoissonDisk(2, radius=radius_unit, seed=rng.randrange(2**32))

                pts = disk.random(int(need))  # in [0,1)^2, may return fewer if space is “full”
                pts = qmc.scale(pts, l_bounds=(0.0, 0.0), u_bounds=(width, height))

                pts_list = pts.tolist()
                rng.shuffle(pts_list)  # critical: don’t just take the early-grown cluster front

                for x, y in pts_list:
                    if accepted >= target:
                        break

                    coords = (float(x), float(y))

                    if _has_conflict(coords, sp.radius):
                        continue

                    v = _clamp01(float(sp.viability_map(*coords)))
                    if rng.random() > v:
                        continue

                    instances.append(Plant(coords, sp, 0))
                    accepted += 1

        return SimulationState(instances, self.size)
