from typing import Iterable
from itertools import chain
import math
from copy import copy
from logging import getLogger

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
        self, coords_or_plant: Plant | tuple[float, float], radius: float | None = None
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
                raise TypeError("radius must be provided when passing coordinates")

        x, y = self.get_cell(coords)
        radius = math.ceil(radius / self.cell_width)
        return chain.from_iterable(
            self.map[i][j]
            for i in range(
                max(0, x - radius), min(self.grid_width - 1, x + radius) + 1
            )
            for j in range(
                max(0, y - radius), min(self.grid_height - 1, y + radius) + 1
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

    def run_state(self, num_years: int) -> None:
        """Run the simulation state for a given number of years.

        Args:
            num_years (int): Number of years to run the simulation state.
        """
        for year in range(num_years):
            logger.debug(f"Year {year + 1}/{num_years}")
            sum_a = 0
            pop_counter: dict[Species, int] = {}
            for plant in self:
                count = pop_counter.get(plant.species, 0)
                pop_counter[plant.species] = count + 1
                sum_a += 1
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
                    for other_plant in tuple(self.get_nearby_plant(new_plant)):
                        # faster than max()
                        if (
                            new_plant.species.radius
                            > other_plant.species.radius
                        ):
                            max_radius = new_plant.species.radius
                        else:
                            max_radius = other_plant.species.radius
                        dx = new_plant.coords[0] - other_plant.coords[0]
                        dy = new_plant.coords[1] - other_plant.coords[1]
                        if (dx**2 + dy**2) < (max_radius**2):
                            if new_plant.vt_prim(
                                pop_counter, sum_a
                            ) < other_plant.vt_prim(pop_counter, sum_a):
                                viable = False
                                break
                            else:
                                self.remove(other_plant)
                    if viable:
                        self.add(new_plant)
