from dataclasses import dataclass


@dataclass
class ForestConfig:
    """Configuration for forest generation."""

    scene_density: float = 1.0
    """Global density multiplier for initial placement."""
    years: int = 0
    """Number of simulation years to run."""
