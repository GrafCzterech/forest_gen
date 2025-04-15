from copy import copy
from dataclasses import dataclass
import random
import math
from typing import Callable, Generator, Iterable
import logging
from itertools import chain

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


class SimulationState:
    """A simulation state used to store the plants in the simulation."""

    def __init__(
        self, plants: Iterable[Plant], size: tuple[float, float], div: int = 10
    ):
        """Initialize the simulation state.

        Args:
            plants (Iterable[Plant]): Plants to add to the simulation state.
            size (tuple[float, float]): Size of the simulation state.
        """
        self.cell_width = size[0] / div
        self.cell_height = size[1] / div
        self.grid_width = int(size[0] / self.cell_width)
        self.grid_height = int(size[1] / self.cell_height)
        self.size = size
        self.map = tuple(
            tuple(
                [[] for _ in range(self.grid_height)]
                for _ in range(self.grid_width)
            )
        )
        for plant in plants:
            self.append(copy(plant))

    def get_cell(self, coords: tuple[float, float]) -> tuple[int, int]:
        """Get the cell coordinates for the given coordinates.

        Args:
            coords (tuple[float, float]): Coordinates of the plant.

        Returns:
            tuple[int, int]: Cell coordinates.
        """
        x = int(coords[0] / self.cell_width)
        y = int(coords[1] / self.cell_height)
        return x, y

    def get_nearby(self, plant: Plant) -> chain[Plant]:
        """Get the plants in a radius around the given coordinates.

        Args:
            plant (Plant): The plant to find neighbors for.

        Returns:
            chain[Plant]: Plants in the radius around the given coordinates.
        """
        x, y = self.get_cell(plant.coords)
        radius = math.ceil(plant.radius / self.cell_width)
        return chain.from_iterable(
            self.map[i][j]
            for i in range(
                max(0, x - radius), min(self.grid_width - 1, x + radius) + 1
            )
            for j in range(
                max(0, y - radius), min(self.grid_height - 1, y + radius) + 1
            )
        )

    def remove(self, plant: Plant) -> None:
        """Remove a plant from the simulation state.

        Args:
            plant (Plant): Plant to remove.
        """
        x, y = self.get_cell(plant.coords)
        self.map[x][y].remove(plant)

    def append(self, plant: Plant) -> None:
        """Add a plant to the simulation state.

        Args:
            plant (Plant): Plant to add.
        """
        x, y = self.get_cell(plant.coords)
        self.map[x][y].append(plant)

    def __iter__(self) -> chain[Plant]:
        """Iterate over the plants in the simulation state.

        Returns:
            chain[Plant]: Plants in the simulation state.
        """
        return chain.from_iterable(chain.from_iterable(self.map))

    def __contains__(self, plant: Plant) -> bool:
        """Check if a plant is in the simulation state.

        Args:
            plant (Plant): Plant to check.

        Returns:
            bool: True if the plant is in the simulation state, False otherwise.
        """
        x, y = self.get_cell(plant.coords)
        return plant in self.map[x][y]

    def run_state(self, num_years: int) -> None:
        """Run the simulation state for a given number of years.

        Args:
            num_years (int): Number of years to run the simulation state.
        """
        for year in range(num_years):
            logging.debug(f"Year {year + 1}/{num_years}")
            pop_counter: dict[Species, int] = {}
            for plant in self:
                count = pop_counter.get(plant.species, 0)
                pop_counter[plant.species] = count + 1
            for plant in tuple(self):  # frozen to avoid mutation
                # kill plants that are too old
                if plant.age > plant.species.max_age:
                    if plant in self:
                        self.remove(plant)
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
                    # check if the plant is viable
                    viable = True
                    for other_plant in tuple(self.get_nearby(new_plant)):
                        if math.dist(
                            new_plant.coords, other_plant.coords
                        ) < max(new_plant.radius, other_plant.radius):
                            if new_plant.vt_prim(
                                pop_counter
                            ) < other_plant.vt_prim(pop_counter):
                                viable = False
                                break
                            else:
                                self.remove(other_plant)
                    if viable:
                        self.append(new_plant)


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
