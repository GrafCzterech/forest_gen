from dataclasses import dataclass
import random
import math
from typing import Callable
import logging

from opensimplex import OpenSimplex
from scipy.stats.qmc import PoissonDisk


class ViabilityMap:

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
    viability_map: Callable[[float, float], float] = ViabilityMap()

    def __hash__(self) -> int:
        return hash(self.name)


# radius for tree is its minimal distance from other trees
@dataclass
class Plant:
    coords: tuple[float, float]
    species: Species
    age: int

    @property
    def radius(self) -> float:
        return self.species.radius

    @property
    def max_age(self) -> int:
        return self.species.max_age

    def vt(self) -> float:
        """Viability of the plant.

        Returns:
            float: Viability of the plant. Value between 0 and 1.
        """
        norm_age = self.age / self.max_age

        if norm_age < 0.5:
            return norm_age
        else:
            return 1 - norm_age

    def vt_prim(self, a: dict[Species, int]) -> float:
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
            / sum(a.values())
        )

    def seed(self) -> list["Plant"]:
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

    def run_state(self, num_years: int, state: list[Plant]) -> list[Plant]:
        """Run the simulation for a given number of years.

        Args:
            num_years (int): Number of years to simulate.
            state (list[Plant]): Initial state of the simulation.

        Returns:
            list[Plant]: Final state of the simulation.
        """
        # return value imples the argument will not be modified
        state = state.copy()
        for year in range(num_years):
            logging.debug(f"Year {year + 1}/{num_years}")
            pop_counter: dict[Species, int] = {}
            for plant in state:
                count = pop_counter.get(plant.species, 0)
                pop_counter[plant.species] = count + 1
            for plant in tuple(state):  # frozen to avoid mutation
                # kill plants that are too old
                if plant.age > plant.species.max_age:
                    state.remove(plant)
                    continue
                plant.age += 1
                # allow the plant to reproduce
                for new_plant in plant.seed():
                    # sanity check
                    if (
                        new_plant.coords[0] < 0
                        or new_plant.coords[0] > self.size[0]
                        or new_plant.coords[1] < 0
                        or new_plant.coords[1] > self.size[1]
                    ):
                        continue
                    state.append(new_plant)
            # remove plants that are too close to each other
            for plant in tuple(state):
                if plant not in state:
                    continue
                for other_plant in state:
                    if plant is other_plant:
                        continue
                    if math.dist(plant.coords, other_plant.coords) < min(
                        plant.radius, other_plant.radius
                    ):
                        # remove the plant with the lower viability
                        if plant.vt_prim(pop_counter) < other_plant.vt_prim(
                            pop_counter
                        ):
                            state.remove(plant)
                            break
                        else:
                            state.remove(other_plant)
        return state

    def run(
        self,
        num_years: int,
        scene_density: float,
    ) -> list[Plant]:
        """Run the simulation.

        Args:
            num_years (int): Number of years to simulate.
            scene_density (float): Density of the scene. Higher values mean more plants.

        Returns:
            list[Plant]: Set of plants at the end of the simulation.
        """
        instances: list[Plant] = []
        for type_species in self.species.values():
            for species in type_species:
                n = (
                    scene_density
                    * species.species_density
                    * self.size[0]
                    * self.size[1]
                ) / len(type_species)
                for _ in range(int(n)):
                    point = self.disk.random()
                    if len(point) == 0:
                        self.disk.reset()
                        point = self.disk.random()
                    point = point[0]
                    instances.append(
                        Plant(
                            (point[0], point[1]),
                            species,
                            random.randint(0, species.max_age),
                        )
                    )

        # simulation initialized

        return self.run_state(num_years, instances)
