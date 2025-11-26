from dataclasses import dataclass, field


@dataclass
class TerrainConfig:

    size: int
    resolution: float = 1.0
    scale: float = 50.0
    octaves: int = 4
    height_scale: float = 1.0
    apply_microrelief: bool = True
    moisture_weights: dict[str, float] = field(
        default_factory=lambda: {"flow": 0.5, "slope": 0.3, "aspect": 0.2}
    )

    def transform(self, x: float) -> int:
        return int(round(x / self.resolution, 0))

    @property
    def rows(self) -> int:
        return self.transform(self.size) + 1

    @property
    def cols(self) -> int:
        return self.rows
