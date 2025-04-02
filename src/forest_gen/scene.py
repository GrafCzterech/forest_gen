from neuroforgelab import SceneSpec
from neuroforgelab.terrain import TerrainInstance

import numpy as np

from .heightmap import NOISE_FUNC, heightmap_to_mesh


class ForestGenSpec(SceneSpec):
    def __init__(self, size: int = 512):
        super().__init__(size=(size, size))

    def generate(self) -> TerrainInstance:

        return TerrainInstance(
            [heightmap_to_mesh(NOISE_FUNC, int(self.size[0]))],
            np.zeros_like(self.size),
            self.size,
        )
