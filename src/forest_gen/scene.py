from typing import Callable
from logging import getLogger
import math

logger = getLogger(__name__)

from neuroforgelab import (
    SceneSpec,
    AssetSpec,
    AssetInstance,
    TerrainInstance,
)

from trimesh import Trimesh

# this is sort of a fasade file for the whole module

from .heightmap import NOISE_FUNC, heightmap_to_meshes, normalized_noise2
from .asset_dist import Simulation, Species, grass_distribution
from .assets import PlantModelFactory

# i have heard many a voice from vile dissidents that showcase their weakness
# and complain about how convoluted this file is. As such overt comments
# have been added


# this is just a simple placeholder function that classifies the terrain,
# used for splitting the terrain into semantic classes
def classify_terrain(x: float, y: float) -> str:
    """Classify the terrain based on the x and y coordinates.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.

    Returns:
        str: The classification of the terrain.
    """
    val = normalized_noise2(x, y)
    if val > 0.5:
        return "forest"
    elif val > 0.1:
        return "grass"
    return "plain"


# we need this later on to properly place the trees
class HeightmapTerrain(TerrainInstance):
    """A wrapper over the TerrainInstance class, that holds the underlying
    heightmap Callable"""

    def __init__(
        self,
        mesh: list[tuple[Trimesh, list[tuple[str, str]]]],
        origin: tuple[float, float, float],
        size: tuple[float, float],
        raw: Callable[[float, float], float],
    ):
        """Initialize the HeightmapTerrain instance.

        Args:
            mesh (list[tuple[Trimesh, list[tuple[str, str]]]]): A list of meshes with their tags.
            origin (tuple[float, float, float]): The origin of the terrain.
            size (tuple[float, float]): The size of the terrain.
            raw (Callable[[float, float], float]): A callable that takes x and y coordinates and returns the height at that point.
        """
        super().__init__(mesh, origin, size)
        self.raw = raw


BORDER_MARGIN = 5.0


class ForestGenSpec(SceneSpec):
    """A specification for generating a forest scene."""

    def __init__(
        self,
        size: int = 256,
    ):
        """Initialize the forest generation specification.

        Args:
            size (int): The size of the terrain.
            robot (AssetBaseCfg | None): The robot configuration.
        """

        # here the assets are hooked up to the scene
        super().__init__(
            size=(size, size),
            palette=[TreeSpec(), GrassSpec()],
        )
        self.origin = (
            BORDER_MARGIN,
            BORDER_MARGIN,
            NOISE_FUNC(BORDER_MARGIN, BORDER_MARGIN) + 1.0,
        )

    def generate(self) -> HeightmapTerrain:

        # please note how we return a custom subclass that holds extra data,
        # so that the hooked up asset classes can depend on that extra data
        return HeightmapTerrain(
            heightmap_to_meshes(
                NOISE_FUNC,
                int(self.size[0]),
                step=0.1,
                classifier=classify_terrain,
            ),
            self.origin,
            self.size,
            NOISE_FUNC,
        )


# a single tree class, not identical to a tree species class
class TreeSpec(AssetSpec):
    """Specification for generating trees in a forest scene."""

    tree_species = {
        Species("Oak", 10, 0.005, radius=5.0),
    }
    """List of tree species we want to generate."""

    def __init__(self, sim_duration: int = 10, tree_density: float = 1.0):
        """Construct a TreeSpec.

        Args:
            sim_duration (int, optional): The duration in years of the simulation used for tree position generation. Defaults to 10.
            tree_density (float, optional): The density of initial trees in the scene. Defaults to 1.0.
        """
        super().__init__("tree")
        self.sim_duration = sim_duration
        self.tree_density = tree_density

    def generate(self, terrain: HeightmapTerrain) -> list[AssetInstance]:
        """Generate a list of tree instances based on the given terrain.

        Args:
            terrain (HeightmapTerrain): The terrain instance on which to generate trees.

        Returns:
            list[AssetInstance]: A list of generated tree asset instances.
        """
        # do the simulation
        logger.debug("Starting simulation")
        sim = Simulation(terrain.size, {self.name: self.tree_species})
        state = sim.new_state(self.tree_density)
        state.run_state(self.sim_duration)
        logger.debug("Simulation finished")

        origin_2d = (terrain.origin[0], terrain.origin[1])
        # then we create the tree instances
        model_factory = PlantModelFactory()
        return [
            self.create_instance(
                f"{plant.species.name}_{i}",
                model_factory.get_model(plant),
                (plant.coords[0], plant.coords[1], terrain.raw(*plant.coords)),
                (0.70711, 0.70711, 0.0, 0.0),
                {"color": "green", "species": plant.species.name},
            )
            for i, plant in enumerate(state)
            if math.dist(plant.coords, origin_2d) > 10.0
        ]


class GrassSpec(AssetSpec):
    """Specification for generating grass in a forest scene."""

    def __init__(self):
        """Construct a GrassSpec."""
        super().__init__("grass")

    def generate(self, terrain: HeightmapTerrain) -> list[AssetInstance]:
        """Generate a list of grass instances based on the given terrain.

        Args:
            terrain (HeightmapTerrain): The terrain instance on which to generate grass.

        Returns:
            list[AssetInstance]: A list of generated grass asset instances.
        """
        # do the simulation
        logger.debug("Generating grass")
        grass = grass_distribution(int(terrain.size[0]), int(terrain.size[1]))
        logger.debug("Generation finished")

        # then we create the tree instances
        model_factory = PlantModelFactory()

        return [
            self.create_instance(
                f"Grass_{i}",
                model_factory.get_model_by_name("Grass", 1),
                (plant[0], plant[1], terrain.raw(*plant)),
                (0.70711, 0.70711, 0.0, 0.0),
                {"color": "blue", "species": "Grass"},
            )
            for i, plant in enumerate(grass)
        ]
