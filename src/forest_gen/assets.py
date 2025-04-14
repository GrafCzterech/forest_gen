from .asset_dist import Plant

from os import path

MODEL_CACHE_PATH = "cache"
EXTENSION = ".stl"


def plant_to_model(plant: Plant) -> str:
    """Convert a plant to a model file path.

    Args:
        plant (Plant): The plant to convert.

    Returns:
        str: The file path to the model.
    """
    return path.join(
        MODEL_CACHE_PATH, f"{plant.species.name}_{plant.age}.{EXTENSION}"
    )
