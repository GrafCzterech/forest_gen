class TerrainViabilityMap:
    """Callable wrapper returning terrain-derived values for coordinates."""

    def __init__(self, data, resolution: float):
        self.data = data
        self.resolution = resolution

    def __call__(self, x: float, y: float) -> float:
        i = int(y / self.resolution)
        j = int(x / self.resolution)
        if i < 0 or i >= self.data.shape[0] or j < 0 or j >= self.data.shape[1]:
            return 0.0
        return float(self.data[i, j])
