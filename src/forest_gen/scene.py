from neuroforgelab import SceneSpec, AssetSpec
from neuroforgelab.terrain import TerrainInstance

import numpy as np

from .heightmap import NOISE_FUNC, heightmap_to_mesh


class ForestGenSpec(SceneSpec):
    def __init__(self, size: int = 512):
        super().__init__(size=(size, size), palette=[], static=[])

    def generate(self) -> TerrainInstance:

        # TODO add here some sort of nodes that trees should generate at

        return TerrainInstance(
            [heightmap_to_mesh(NOISE_FUNC, int(self.size[0]))],
            np.zeros_like(self.size),
            self.size,
        )  # then add those pos's to this object


# so that in AssetSpec::find_positions


class TreeSpec(AssetSpec): ...  # TODO
