from dataclasses import dataclass


@dataclass
class DistributionConfig:
    """Configuration for running a plant distribution simulation."""

    scene_density: float = 1.0
    years: int = 0
    max_population: int | None = None