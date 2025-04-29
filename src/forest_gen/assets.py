from .asset_dist import Plant

from neuroforgelab import AssetMesh, DynamicMesh
import trimesh

MODEL_CACHE_PATH = "cache"
EXTENSION = "usd"


class TreeModelFactory:
    """A factory for creating tree models."""

    def __init__(self):
        """Initialize the factory."""
        self.models: dict[tuple[str, int], AssetMesh] = {}

    def get_model(self, plant: Plant) -> AssetMesh:
        """Get the model for a given plant.

        Args:
            plant (Plant): The plant to get the model for.

        Returns:
            str: The file path to the model.
        """
        key = (plant.species.name, plant.age)
        if key not in self.models:
            self.models[key] = DynamicMesh(trimesh.creation.capsule())
        return self.models[key]
