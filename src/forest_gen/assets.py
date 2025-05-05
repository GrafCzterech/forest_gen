from .asset_dist import Plant

from neuroforgelab import AssetMesh, USDMesh, DynamicMesh

MODEL_CACHE_PATH = "cache"
EXTENSION = "glb"


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
            AssetMesh: Mesh of loaded asset.
        """
        key = (plant.species.name, plant.age)
        if key not in self.models:
            self.models[key] = USDMesh(
                f"{MODEL_CACHE_PATH}/{plant.species.name}_{plant.age}.{EXTENSION}"
            )
        return self.models[key]
