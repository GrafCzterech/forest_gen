from .asset_dist import Plant

from os import path

MODEL_CACHE_PATH = "cache"
EXTENSION = ".glb"


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
