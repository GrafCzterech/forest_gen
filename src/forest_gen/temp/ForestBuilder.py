class ForestBuilder:
    """Fluent builder for constructing a ForestGenerator."""

    def __init__(self):
        self._species: dict[str, set] = {}
        self._size: tuple[float, float] = (100.0, 100.0)

    def with_size(self, size: tuple[float, float]) -> "ForestBuilder":
        self._size = size
        return self

    def add_species(self, kind: str, species) -> "ForestBuilder":
        self._species.setdefault(kind, set()).add(species)
        return self

    def build(self):
        from .ForestGenerator import ForestGenerator

        return ForestGenerator(self._size, self._species)
