from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    """
    Immutable representation of a navigational obstacle.
    """

    kind: str
    """Obstacle type identifier."""

    coords: tuple[float, float]
    """World-space coordinates ``(x, y)``."""

    radius: float
    """Obstacle influence radius."""
