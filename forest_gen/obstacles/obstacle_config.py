from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ObstacleSpec:
    """Specification for an obstacle type.

    Attributes:
        name: Friendly label describing the obstacle kind.
        radius: Collision radius used when spacing obstacles apart.
        weight: Relative weight when randomly sampling obstacle kinds.
    """

    name: str
    radius: float
    weight: float = 1.0


_default_obstacle_specs = (
    ObstacleSpec("fallen_tree", radius=4.0, weight=0.4),
    ObstacleSpec("rock", radius=2.5, weight=0.4),
    ObstacleSpec("log", radius=3.0, weight=0.2),
)


def default_obstacle_specs() -> tuple[ObstacleSpec, ...]:
    """Return the default obstacle specs used when none are provided."""

    return _default_obstacle_specs


@dataclass
class ObstacleConfig:
    """Configuration used when generating random obstacles."""

    size: tuple[float, float]
    density: float = 0.0025
    min_distance: float = 2.0
    seed: int | None = None
    specs: tuple[ObstacleSpec, ...] | None = None

    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]

    def expected_obstacle_count(self) -> int:
        return max(1, int(round(self.area * self.density)))

    def with_specs(self, specs: Iterable[ObstacleSpec]) -> "ObstacleConfig":
        """Return a copy of this config using custom obstacle specs."""

        return ObstacleConfig(
            size=self.size,
            density=self.density,
            min_distance=self.min_distance,
            seed=self.seed,
            specs=tuple(specs),
        )
    