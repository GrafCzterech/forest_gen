from dataclasses import dataclass, field
import random
from typing import Callable, Iterable
import math

from opensimplex import OpenSimplex

# the whole algorithm utilizes classes laid out here


@dataclass
class ViabilityMap:
    """A function that classifies the viability of the plant in the given location."""

    def __init__(self, eps: float = 0.1):

        self.noise = OpenSimplex(random.randint(0, 1000))
        self.eps = eps

    def __call__(self, x: float, y: float) -> float:
        return self.noise.noise2(x / self.eps, y / self.eps) * 0.5 + 0.5


@dataclass
class Species:
    """A species specification"""

    name: str
    max_age: int
    """Maximum age of the plant."""
    species_density: float
    """Density used to calculate the initial number of plants in the simulation. Higher values mean more plants."""
    reproduction_rate: int = 5
    """Maximum number of seeds produced by a plant in one year."""
    reproduction_radius: float = 20.0
    """Radius in which the seeds can be planted."""
    radius: float = 0.5
    """Radius needed for the plant to consider itself as clear of obstacles."""
    viability_map: Callable[[float, float], float] = field(
        default_factory=ViabilityMap
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Species):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


# radius for tree is its minimal distance from other trees
@dataclass
class Plant:
    coords: tuple[float, float]
    species: Species
    age: int

    def vt(self) -> float:
        """Viability of the plant.

        Returns:
            float: Viability of the plant. Value between 0 and 1.
        """
        norm_age = self.age / self.species.max_age

        if norm_age < 0.5:
            return norm_age
        else:
            return 1 - norm_age

    def vt_prim(self, a: dict[Species, int], sum_a: int) -> float:
        """Modified viability of the plant.
        This function takes into account the population of the species, and the
        viability of the plant in it's location.

        Args:
            a (dict[Species, int]): Population of the species.

        Returns:
            float: Modified viability of the plant. Value between 0 and 1.
        """
        # TODO the problem currently is that this is sort of unrealistic with population being global
        return (
            self.species.viability_map(*self.coords)
            * self.vt()
            * a[self.species]
            / sum_a
        )

    def seed(self) -> Iterable["Plant"]:
        """Allow the plant to reproduce.

        Returns:
            Iterable[Plant]: A collection of new plants.
        """
        res = []
        for _ in range(random.randint(0, self.species.reproduction_rate)):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(
                self.species.species_density, self.species.reproduction_radius
            )
            new_x = self.coords[0] + distance * math.cos(angle)
            new_y = self.coords[1] + distance * math.sin(angle)
            res.append(Plant((new_x, new_y), self.species, 0))
        return res
