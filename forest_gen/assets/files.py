import os
from logging import getLogger

from stripe_kit import AssetMesh, UniversalMesh

from forest_gen_utils.asset_dist import Plant

logger = getLogger(__name__)

# USDMesh is weird and doesn't work in Rl env

# this file handles how models are generated. The idea is to create an abstract
# fasade that won't change if we choose to load or generate assets

# MODEL_CACHE_PATH = os.path.abspath("../forest-gen/models")
EXTENSION = "glb"


"""
Plant model loading and caching utilities.

Provides a stable facade for loading plant asset meshes while hiding
asset format and caching details.
"""


# verbose? yeah but necessary cuz CACHING
class PlantModelFactory:
    """
    Factory for loading and caching plant asset meshes.

    Acts as a facade over asset loading to decouple simulation logic
    from concrete asset formats and caching behavior.
    """

    def __init__(self, scale: float = 0.1, path: str = "../forest-gen/forest_gen/models"):
        """
        Initialize the model factory.

        :param scale: Base scale applied to loaded models.
        :type scale: float
        :param path: Path to the directory containing plant models.
        :type path: str
        """
        self.models: dict[tuple[str, int], AssetMesh] = {}
        self.scale = scale
        self.path = path

    def get_model(self, plant: Plant) -> AssetMesh:
        """
        Retrieve the asset model for a plant instance.

        :param plant: Plant instance.
        :type plant: Plant
        :return: Loaded asset mesh.
        :rtype: AssetMesh
        """
        return self.get_model_by_name(plant.species.name, plant.age)

    def get_model_by_name(self, name: str, age: int) -> AssetMesh:
        """
        Retrieve or load a plant model by species name and age.

        Models are cached by ``(name, age)`` to avoid repeated loads.

        :param name: Species name.
        :type name: str
        :param age: Plant age.
        :type age: int
        :return: Loaded asset mesh.
        :rtype: AssetMesh
        """

        key = (name, age)
        if key not in self.models:
            # MAYBE remove this debug statement, its a pain in the ass
            logger.debug(f"Loading model for {name} age {age}")
            self.models[key] = UniversalMesh(
                f"{self.path}/{name}_{age}.{EXTENSION}",
                scale=(self.scale, self.scale, self.scale),
            )
        return self.models[key]

    def get_usdz_model_by_name(
        self, name: str, age: int, scale_mult: float = 1
    ) -> AssetMesh:
        """
        Retrieve or load a USDZ plant model by species name and age.

        :param name: Species name.
        :type name: str
        :param age: Plant age.
        :type age: int
        :param scale_mult: Additional scale multiplier.
        :type scale_mult: float
        :return: Loaded asset mesh.
        :rtype: AssetMesh
        """

        key = (name, age)
        if key not in self.models:
            # MAYBE remove this debug statement, its a pain in the ass
            logger.debug(f"Loading model for {name} age {age}")
            self.models[key] = UniversalMesh(
                f"{self.path}/{name}_{age}.usdz",
                scale=(
                    self.scale * scale_mult,
                    self.scale * scale_mult,
                    self.scale * scale_mult,
                ),
                # 43 sekundy przy 200 modelach Grassbed_1.usdz
                # 21 sekund przy ~300 modelach GrassBed_1.usdz
                # 176 sekund przy 5100 modelach GrassBed_1.usdz
                collision_props=None,
                # 18 sekund przy 200 modelach Grassbed_1.usdz
                # 20 sekund przy ~300 modelach GrassBed_1.usdz
                # 174 sekund przy 5000 modelach GrassBed_1.usdz
                # rigid_props=None,
                # 18 sekund przy 200 modelach Grassbed_1.usdz
                # razem z collision_props (nic nie zmieniło)
            )
        return self.models[key]
