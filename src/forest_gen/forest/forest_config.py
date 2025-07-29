from dataclasses import dataclass


@dataclass
class ForestConfig:
    """Configuration for forest generation."""

    scene_density: float = 1.0
    years: int = 0
