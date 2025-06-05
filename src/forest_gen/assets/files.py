from logging import getLogger

logger = getLogger(__name__)

from ..asset_dist import Plant
import os
from neuroforgelab import AssetMesh, USDMesh

# this file handles how models are generated. The idea is to create an abstract
# fasade that won't change if we choose to load or generate assets

MODEL_CACHE_PATH = os.path.abspath("cache")
EXTENSION = "usd"


# verbose? yeah but necessary cuz CACHING
class TreeModelFactory:
    """A factory for creating tree models."""

    def __init__(self, scale: float = 0.1):
        """Initialize the factory."""
        self.models: dict[tuple[str, int], AssetMesh] = {}
        self.scale = scale

    def get_model(self, plant: Plant) -> AssetMesh:
        """Get the model for a given plant.

        Args:
            plant (Plant): The plant to get the model for.

        Returns:
            AssetMesh: Mesh of loaded asset.
        """

        key = (plant.species.name, plant.age)
        if key not in self.models:
            # MAYBE remove this debug statement, its a pain in the ass
            logger.debug(
                f"Loading model for {plant.species.name} age {plant.age}"
            )
            self.models[key] = USDMesh(
                f"{MODEL_CACHE_PATH}/{plant.species.name}_{plant.age}.{EXTENSION}",
                scale=(self.scale, self.scale, self.scale),
            )
        return self.models[key]
