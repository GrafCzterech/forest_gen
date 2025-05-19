from typing import Callable
from copy import deepcopy
import random
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

from isaaclab.assets import AssetBaseCfg

# this is sort of a fasade file for the whole module

from .heightmap import NOISE_FUNC, heightmap_to_meshes, normalized_noise2
from .asset_dist import Simulation, Species
from .assets import TreeModelFactory

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
    if normalized_noise2(x, y) > 0.5:
        return "forest"
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


class ForestGenSpec(SceneSpec):
    """The one class that specifies how the scene is generated.
    Size and robot go in, and a scene comes out."""

    def __init__(self, size: int = 256, robot: AssetBaseCfg | None = None):
        # determine the start position and save it
        self.start_point = (0.0, 0.0)
        if robot is not None:
            robot = deepcopy(robot)
            x = random.uniform(0, size)
            y = random.uniform(0, size)
            self.start_point = (x, y)
            robot.init_state = robot.InitialStateCfg(
                (x, y, NOISE_FUNC(x, y) + 1.0)
            )
            logger.debug(f"Robot initial pos: {robot.init_state.pos}")

        # here the assets are hooked up to the scene
        super().__init__(size=(size, size), robot=robot, palette=[TreeSpec()])

    def generate(self) -> HeightmapTerrain:

        # please note how we return a custom subclass that holds extra data,
        # so that the hooked up asset classes can depend on that extra data
        return HeightmapTerrain(
            heightmap_to_meshes(
                NOISE_FUNC, int(self.size[0]), classifier=classify_terrain
            ),
            (self.start_point[0], self.start_point[1], 0.0),
            self.size,
            NOISE_FUNC,
        )


# a single tree class, not identical to a tree species class
class TreeSpec(AssetSpec):
    """Specification for generating trees in a forest scene."""

    tree_species = {
        Species("Oak", 10, 0.005, radius=3.5),
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

        # then we create the tree instances
        model_factory = TreeModelFactory()
        origin_2d = (terrain.origin[0], terrain.origin[1])
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
