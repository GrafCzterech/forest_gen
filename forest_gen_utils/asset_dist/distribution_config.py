from dataclasses import dataclass


@dataclass
class DistributionConfig:
    """Configuration for running a plant distribution simulation."""

    scene_density: float = 1.0
    """Global density scaling factor."""
    years: int = 0
    """Number of simulation steps (years)."""
    max_population: int | None = None
    """Optional hard cap on total population size."""
