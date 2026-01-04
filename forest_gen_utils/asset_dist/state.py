import math
from copy import copy
from itertools import chain
from logging import getLogger
from typing import Iterable
import random

import numpy as np

logger = getLogger(__name__)

from .definitions import Plant, Species

# here is where the magic happens


class SimulationState:
    """
    Mutable plant population state with spatial indexing.

    Stores plants in a coarse grid to accelerate neighborhood queries and
    supports advancing the simulation in discrete yearly steps.
    """

    def __init__(
        self, plants: Iterable[Plant], size: tuple[float, float], div: int = 10
    ):
        """
        Initialize a simulation state.

        :param plants: Initial plants to insert.
        :type plants: Iterable[Plant]
        :param size: Simulation area size ``(width, height)``.
        :type size: tuple[float, float]
        :param div: Number of grid divisions per axis for spatial indexing.
        :type div: int
        """

        self.cell_width = size[0] / div
        self.cell_height = size[1] / div
        self.grid_width = int(size[0] / self.cell_width)
        self.grid_height = int(size[1] / self.cell_height)
        self.size = size
        self.map = tuple(
            tuple([] for _ in range(self.grid_height + 1))
            for _ in range(self.grid_width + 1)
        )
        for plant in plants:
            self.add(copy(plant))

    def get_cell(self, coords: tuple[float, float]) -> tuple[int, int]:
        """
        Map world coordinates to a grid cell index.

        :param coords: World-space coordinates ``(x, y)``.
        :type coords: tuple[float, float]
        :return: Cell indices ``(cx, cy)``.
        :rtype: tuple[int, int]
        """
        x = int(coords[0] / self.cell_width)
        y = int(coords[1] / self.cell_height)
        return x, y

    def get_nearby(
        self,
        coords_or_plant: Plant | tuple[float, float],
        radius: float | None = None,
    ) -> chain[Plant]:
        """
        Iterate plants in cells intersecting a neighborhood radius.

        If a :class:`Plant` is passed and ``radius`` is ``None``, the plant's
        species radius is used.

        :param coords_or_plant: Plant or world-space coordinates.
        :type coords_or_plant: Plant | tuple[float, float]
        :param radius: Search radius (required when passing coordinates).
        :type radius: float or None
        :return: Nearby plants (not distance-filtered).
        :rtype: itertools.chain[Plant]
        :raises TypeError: If coordinates are passed without ``radius``.
        """
        if isinstance(coords_or_plant, Plant):
            coords = coords_or_plant.coords
            if radius is None:
                radius = coords_or_plant.species.radius
        else:
            coords = coords_or_plant
            if radius is None:
                raise TypeError("radius must be provided when passing coordinates")

        x, y = self.get_cell(coords)
        radius = math.ceil(radius / self.cell_width)
        return chain.from_iterable(
            self.map[i][j]
            for i in range(max(0, x - radius), min(self.grid_width, x + radius) + 1)
            for j in range(max(0, y - radius), min(self.grid_height, y + radius) + 1)
        )

    def get_nearby_plant(self, plant: Plant) -> chain[Plant]:
        """
        Shortcut for neighborhood query around a plant.

        :param plant: Reference plant.
        :type plant: Plant
        :return: Nearby plants.
        :rtype: itertools.chain[Plant]
        """
        return self.get_nearby(plant.coords, plant.species.radius)

    def remove(self, plant: Plant) -> None:
        """
        Remove a plant from the state.

        :param plant: Plant to remove.
        :type plant: Plant
        """
        x, y = self.get_cell(plant.coords)
        self.map[x][y].remove(plant)

    def add(self, plant: Plant) -> None:
        """
        Add a plant to the state.

        :param plant: Plant to add.
        :type plant: Plant
        """
        x, y = self.get_cell(plant.coords)
        self.map[x][y].append(plant)

    def __iter__(self) -> chain[Plant]:

        return chain.from_iterable(chain.from_iterable(self.map))

    def __len__(self) -> int:
        return sum(len(cell) for row in self.map for cell in row)

    def _evaluate_seed(
        self,
        new_plant: Plant,
        pop_counter: dict[Species, int],
        total_population: int,
    ) -> tuple[bool, list[Plant]]:
        """
        Evaluate whether a candidate seed can be inserted.

        Performs overlap checks against nearby plants and compares
        population-weighted viability to decide acceptance and possible
        removals.

        :param new_plant: Candidate plant.
        :type new_plant: Plant
        :param pop_counter: Population counts for the current step.
        :type pop_counter: dict[Species, int]
        :param total_population: Total population for the current step.
        :type total_population: int
        :return: ``(accepted, removable)`` where ``removable`` are displaced plants.
        :rtype: tuple[bool, list[Plant]]
        """

        nearby_plants = tuple(self.get_nearby_plant(new_plant))
        if not nearby_plants:
            return True, []

        neighbor_coords = np.array([plant.coords for plant in nearby_plants])
        neighbor_radii = np.array(
            [plant.species.radius for plant in nearby_plants], dtype=float
        )

        deltas = neighbor_coords - np.asarray(new_plant.coords, dtype=float)
        dist_sq = np.einsum("ij,ij->i", deltas, deltas)
        max_radii = np.maximum(neighbor_radii, new_plant.species.radius)
        overlap_mask = dist_sq < (max_radii**2)

        overlapping_indices = np.flatnonzero(overlap_mask)
        if overlapping_indices.size == 0:
            return True, []

        new_viability = new_plant.vt_prim(pop_counter, total_population)
        other_viabilities = np.array(
            [
                nearby_plants[idx].vt_prim(pop_counter, total_population)
                for idx in overlapping_indices
            ],
            dtype=float,
        )

        if other_viabilities.max() > new_viability:
            return False, []

        removable = [
            nearby_plants[idx]
            for idx, viability in zip(overlapping_indices, other_viabilities)
            if new_viability >= viability
        ]
        return True, removable

    def run_state(self, num_years: int, max_population: int | None = None) -> None:
        """
        Advance the simulation by a number of years.

        :param num_years: Number of years to simulate.
        :type num_years: int
        :param max_population: Optional population cap.
        :type max_population: int or None
        """

        def _clamp01(v: float) -> float:
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v
    
        for year in range(num_years):
            logger.debug(f"Year {year + 1}/{num_years}")
            pop_counter: dict[Species, int] = {}

            plants_now = list(self)
            for plant in plants_now:
                pop_counter[plant.species] = pop_counter.get(plant.species, 0) + 1

            sum_a = len(plants_now)
            if max_population is not None and sum_a >= max_population:
                logger.debug("Population limit reached; stopping simulation")
                break

            for plant in plants_now:  # frozen to avoid mutation
                if plant.age > plant.species.max_age:
                    if plant in self:
                        self.remove(plant)
                    continue
                plant.age += 1

                if max_population is not None and len(self) >= max_population:
                    continue

                for new_plant in plant.seed():
                    if (
                        new_plant.coords[0] < 0
                        or new_plant.coords[0] > self.size[0]
                        or new_plant.coords[1] < 0
                        or new_plant.coords[1] > self.size[1]
                    ):
                        continue

                    v = _clamp01(float(new_plant.species.viability_map(*new_plant.coords)))
                    if random.random() > v:
                        continue

                    viable, removable = self._evaluate_seed(new_plant, pop_counter, sum_a)
                    if not viable:
                        continue

                    for other_plant in removable:
                        self.remove(other_plant)

                    self.add(new_plant)
                    sum_a += 1

                    if max_population is not None and sum_a >= max_population:
                        break

                if max_population is not None and sum_a >= max_population:
                    break

            if max_population is not None and sum_a >= max_population:
                break
