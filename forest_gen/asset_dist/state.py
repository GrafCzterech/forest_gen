from typing import Iterable
from itertools import chain
import math
from copy import copy
from logging import getLogger

import numpy as np

logger = getLogger(__name__)

from .definitions import Species, Plant

# here is where the magic happens


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
            tuple([] for _ in range(self.grid_height + 1))
            for _ in range(self.grid_width + 1)
        )
        for plant in plants:
            self.add(copy(plant))

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

    def get_nearby(
        self,
        coords_or_plant: Plant | tuple[float, float],
        radius: float | None = None,
    ) -> chain[Plant]:
        """Get the plants in a radius around the given coordinates or Plant.

        Args:
            coords_or_plant: Coordinates tuple or Plant instance.
            radius: Optional radius to search for plants. If ``coords_or_plant``
                is a :class:`Plant` and ``radius`` is ``None`` its species
                radius is used.

        Returns:
            chain[Plant]: Plants in the radius around the given coordinates.
        """
        if isinstance(coords_or_plant, Plant):
            coords = coords_or_plant.coords
            if radius is None:
                radius = coords_or_plant.species.radius
        else:
            coords = coords_or_plant
            if radius is None:
                raise TypeError(
                    "radius must be provided when passing coordinates"
                )

        x, y = self.get_cell(coords)
        radius = math.ceil(radius / self.cell_width)
        return chain.from_iterable(
            self.map[i][j]
            for i in range(
                max(0, x - radius), min(self.grid_width, x + radius) + 1
            )
            for j in range(
                max(0, y - radius), min(self.grid_height, y + radius) + 1
            )
        )

    def get_nearby_plant(self, plant: Plant) -> chain[Plant]:
        """Get the plants in a radius around the given plant.

        Args:
            plant (Plant): Plant to search for.

        Returns:
            chain[Plant]: Plants in the radius around the given plant.
        """
        return self.get_nearby(plant.coords, plant.species.radius)

    def remove(self, plant: Plant) -> None:
        """Remove a plant from the simulation state.

        Args:
            plant (Plant): Plant to remove.
        """
        x, y = self.get_cell(plant.coords)
        self.map[x][y].remove(plant)

    def add(self, plant: Plant) -> None:
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

    def __len__(self) -> int:
        """Return number of plants in the simulation state."""
        return sum(len(cell) for row in self.map for cell in row)

    def _evaluate_seed(
        self,
        new_plant: Plant,
        pop_counter: dict[Species, int],
        total_population: int,
    ) -> tuple[bool, list[Plant]]:
        """Evaluate whether a seed can be added to the simulation.

        This function batches spatial and viability checks to reduce Python
        overhead when working with large plant populations.

        Args:
            new_plant: The candidate plant to insert.
            pop_counter: Population counts for the current year.
            total_population: Total population for the current year.

        Returns:
            A tuple containing a viability flag and the plants that should be
            removed if the seed is viable.
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

    def run_state(
        self, num_years: int, max_population: int | None = None
    ) -> None:
        """Run the simulation state for a given number of years.

        Args:
            num_years (int): Number of years to run the simulation state.
            max_population (int | None): Optional cap on the population size. If
                provided the simulation stops spawning new plants once the
                number of plants reaches this limit.
        """
        for year in range(num_years):
            logger.debug(f"Year {year + 1}/{num_years}")
            pop_counter: dict[Species, int] = {}

            plants_now = list(self)
            for plant in plants_now:
                pop_counter[plant.species] = (
                    pop_counter.get(plant.species, 0) + 1
                )

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
                    viable, removable = self._evaluate_seed(
                        new_plant, pop_counter, sum_a
                    )

                    if not viable:
                        continue

                    for other_plant in removable:
                        self.remove(other_plant)

                    self.add(new_plant)
                    sum_a += 1

                    if (
                        max_population is not None
                        and sum_a >= max_population
                    ):
                        break

                if max_population is not None and sum_a >= max_population:
                    break

            if max_population is not None and sum_a >= max_population:
                break
