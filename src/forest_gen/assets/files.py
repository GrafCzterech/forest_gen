from logging import getLogger

logger = getLogger(__name__)

from ..asset_dist import Plant
import os
from neuroforgelab import AssetMesh, UniversalMesh

# USDMesh is weird and doesn't work in Rl env

# this file handles how models are generated. The idea is to create an abstract
# fasade that won't change if we choose to load or generate assets

MODEL_CACHE_PATH = os.path.abspath("assets")
EXTENSION = "glb"


# verbose? yeah but necessary cuz CACHING
class PlantModelFactory:
    """A factory for creating plant models."""

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
        return self.get_model_by_name(plant.species.name, plant.age)

    def get_model_by_name(self, name: str, age: int) -> AssetMesh:
        """Get the model for a given plant.

        Args:
            name (str): The name of the plant species.
            age (int): The age of the plant.
        Returns:
            AssetMesh: Mesh of loaded asset named "name_age".
        """

        key = (name, age)
        if key not in self.models:
            # MAYBE remove this debug statement, its a pain in the ass
            logger.debug(f"Loading model for {name} age {age}")
            self.models[key] = UniversalMesh(
                f"{MODEL_CACHE_PATH}/{name}_{age}.{EXTENSION}",
                scale=(self.scale, self.scale, self.scale),
            )
        return self.models[key]
