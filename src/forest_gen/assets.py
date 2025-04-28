from .asset_dist import Plant

from neuroforgelab import AssetMesh, UniversalMesh

from os import path

MODEL_CACHE_PATH = "cache"
EXTENSION = "usd"


class TreeModelFactory:
    """A factory for creating tree models."""

    def __init__(self):
        """Initialize the factory."""
        self.models: dict[tuple[str, int], AssetMesh] = {}

    def get_model(self, plant: Plant) -> str:
        """Get the model for a given plant.

        Args:
            plant (Plant): The plant to get the model for.

        Returns:
            str: The file path to the model.
        """
        key = (plant.species.name, plant.age)
        if key not in self.models:
            model_path = plant_to_model(plant)
            self.models[key] = UniversalMesh(model_path)
        return model_path


def plant_to_model(plant: Plant) -> str:
    """Convert a plant to a model file path.

    Args:
        plant (Plant): The plant to convert.

    Returns:
        str: The file path to the model.
    """
    model_path = path.join(
        MODEL_CACHE_PATH, f"{plant.species.name}_{plant.age}.{EXTENSION}"
    )
    if not path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {path.realpath(model_path)} does not exist. Please generate the model first."
        )
    return model_path
