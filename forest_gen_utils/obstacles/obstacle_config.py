from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ObstacleSpec:
    """
    Specification describing an obstacle type.
    """

    name: str
    """Obstacle type identifier."""
    radius: float
    """Collision radius used for spacing."""
    weight: float = 1.0
    """Relative sampling weight."""


_default_obstacle_specs = (
    ObstacleSpec("fallen_tree", radius=4.0, weight=0.4),
    ObstacleSpec("rock", radius=2.5, weight=0.4),
    ObstacleSpec("log", radius=3.0, weight=0.2),
)


def default_obstacle_specs() -> tuple[ObstacleSpec, ...]:
    """
    Return the default obstacle specifications.
    """

    return _default_obstacle_specs


@dataclass
class ObstacleConfig:
    """
    Configuration for random obstacle generation.
    """

    size: tuple[float, float]
    """Generation area size ``(width, height)``."""
    density: float = 0.0025
    """Obstacle density per unit area."""
    min_distance: float = 2.0
    """Minimum spacing between obstacles."""
    seed: int | None = None
    """Optional random seed."""
    specs: tuple[ObstacleSpec, ...] | None = None
    """Available obstacle specifications."""

    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]

    def expected_obstacle_count(self) -> int:
        return max(1, int(round(self.area * self.density)))

    def with_specs(self, specs: Iterable[ObstacleSpec]) -> "ObstacleConfig":
        """
        Return a copy of this config with custom obstacle specs.

        :param specs: Obstacle specifications to use.
        :type specs: Iterable[ObstacleSpec]
        :return: New obstacle configuration.
        :rtype: ObstacleConfig
        """

        return ObstacleConfig(
            size=self.size,
            density=self.density,
            min_distance=self.min_distance,
            seed=self.seed,
            specs=tuple(specs),
        )
