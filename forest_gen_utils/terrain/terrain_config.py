from dataclasses import dataclass, field


@dataclass
class TerrainConfig:
    """
    Configuration container for terrain generation parameters.
    """

    size: int
    """Terrain size."""
    resolution: float = 1.0
    """Spatial resolution  (units per grid cell)."""
    scale: float = 50.0
    """Base noise scale controlling feature size."""
    octaves: int = 4
    """Number of noise octaves for multi-scale generation."""
    height_scale: float = 1.0
    """Global height scaling factor."""
    apply_microrelief: bool = True
    """Whether microrelief should be applied to the terrain."""
    moisture_weights: dict[str, float] = field(
        default_factory=lambda: {"flow": 0.5, "slope": 0.3, "aspect": 0.2}
    )
    """Default weighting factors for moisture computation."""

    def transform(self, x: float) -> int:
        """
        Convert a ws coordinate to grid-space.

        :param x: Coordinate in world units.
        :type x: float
        :return: Grid index.
        :rtype: int
        """
        return int(round(x / self.resolution, 0))

    @property
    def rows(self) -> int:
        return self.transform(self.size) + 1

    @property
    def cols(self) -> int:
        return self.rows
