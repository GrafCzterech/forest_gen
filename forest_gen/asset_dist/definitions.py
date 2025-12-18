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
    species_density: float = 0.02
    """Target density in plants per square meter used for the initial number of
    plants in the simulation."""
    reproduction_rate: int = 5
    """Maximum number of seeds produced by a plant in one year."""
    reproduction_radius: float = 20.0
    """Radius in which the seeds can be planted."""
    radius: float = 0.5
    """Radius needed for the plant to consider itself as clear of obstacles."""
    viability_map: Callable[[float, float], float] = field(
        default_factory=ViabilityMap
    )
    juvenile_mortality_depth: float = 0.4
    """Peak reduction in early-life viability (0-1)."""
    juvenile_mortality_peak: float = 0.05
    """Normalized age of highest juvenile mortality risk."""
    juvenile_mortality_width: float = 0.03
    """Spread of the juvenile mortality spike as a fraction of max age."""
    juvenile_recovery_age: float = 0.2
    """Normalized age by which viability recovers to its peak."""
    senescence_start: float = 0.7
    """Normalized age when senescence effects start."""
    senescence_plateau: float = 0.5
    """Viability level maintained during senescence plateau."""
    senescence_plateau_span: float = 0.15
    """Duration of the senescence plateau as a fraction of lifespan."""

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
        sp = self.species
        if sp.max_age <= 0:
            return 0.0

        norm_age = max(0.0, min(1.0, self.age / sp.max_age))

        # Juvenile growth toward peak viability with an early-life mortality spike.
        growth_phase = min(1.0, norm_age / max(sp.juvenile_recovery_age, 1e-6))
        juvenile_width = max(sp.juvenile_mortality_width, 1e-6)
        juvenile_penalty = sp.juvenile_mortality_depth * math.exp(
            -((norm_age - sp.juvenile_mortality_peak) ** 2)
            / (2 * juvenile_width**2)
        )
        juvenile_modifier = max(0.0, 1.0 - juvenile_penalty)

        # If the plant has not reached maturity, viability is limited by growth
        # and the mortality spike.
        if norm_age < sp.senescence_start:
            base_viability = (
                1.0 if norm_age >= sp.juvenile_recovery_age else growth_phase
            )
            return max(0.0, min(1.0, base_viability * juvenile_modifier))

        # Past senescence start, interpolate toward a plateau and hold viability steady.
        plateau_end = min(1.0, sp.senescence_start + sp.senescence_plateau_span)

        if norm_age < plateau_end:
            t = (norm_age - sp.senescence_start) / max(
                plateau_end - sp.senescence_start, 1e-6
            )
            viability = 1.0 - t * (1.0 - sp.senescence_plateau)
        else:
            viability = sp.senescence_plateau

        return max(0.0, min(1.0, viability))

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
