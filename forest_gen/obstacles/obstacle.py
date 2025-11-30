from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    """Represents a navigational obstacle on the terrain."""

    kind: str
    coords: tuple[float, float]
    radius: float