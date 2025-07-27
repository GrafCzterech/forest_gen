from dataclasses import dataclass, field
from typing import Dict

@dataclass
class TerrainConfig:
    """
    Configuration for terrain generation.

    Attributes:
        rows: Number of rows in the heightmap grid.
        cols: Number of columns in the heightmap grid.
        resolution: Spatial resolution (distance between grid points).
        scale: "Zoom" factor for noise frequency.
        octaves: Number of noise octaves (layers) to combine.
        apply_microrelief: Whether to apply microrelief (small-scale noise).
        moisture_weights: Weights for moisture model components.
    """
    rows: int
    cols: int
    resolution: float = 1.0
    scale: float = 50.0
    octaves: int = 4
    apply_microrelief: bool = True
    moisture_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "flow": 0.5,
            "slope": 0.3,
            "aspect": 0.2
        }
    )
