class ForestBuilder:
    """Fluent builder for constructing a ForestGenerator."""

    def __init__(self):
        self._species: dict[str, set] = {}
        self._size: tuple[float, float] = (100.0, 100.0)
        self._terrain = None

    def with_size(self, size: tuple[float, float]) -> "ForestBuilder":
        self._size = size
        return self

    def add_species(self, kind: str, species) -> "ForestBuilder":
        self._species.setdefault(kind, set()).add(species)
        return self

    def with_terrain(self, terrain) -> "ForestBuilder":
        self._terrain = terrain
        return self

    def with_terrain_data(self, moisture, resolution: float) -> "ForestBuilder":
        """Provide pre-generated terrain data instead of a TerrainGenerator."""
        from dataclasses import dataclass
        import numpy as np

        @dataclass
        class _Terrain:
            moisture: np.ndarray
            resolution: float

        self._terrain = _Terrain(np.asarray(moisture, dtype=float), resolution)
        return self

    def build(self):
        from .ForestGenerator import ForestGenerator

        return ForestGenerator(self._size, self._species, self._terrain)
