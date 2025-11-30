from logging import getLogger
import math
import random

logger = getLogger(__name__)

from neuroforgelab import (
    SceneSpec,
    AssetSpec,
    AssetInstance,
    TerrainInstance,
)

from trimesh import Trimesh
from opensimplex import noise2

# this is sort of a facade file for the whole module

from .terrain import TerrainConfig, TerrainBuilder, Terrain
from .forest import ForestBuilder, ForestConfig
from .asset_dist import GrassDistributor, Species, grass_points
from .assets import PlantModelFactory
from .travelsibilitymap import TraversabilityConfig, TraversabilityMapBuilder

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
    if noise2(x, y) > 0:
        return "forest"
    return "plain"


GRASS_BASE_COLOR = (0.07, 0.42, 0.07)


# we need this later on to properly place the trees
class HeightmapTerrain(TerrainInstance):
    """A wrapper over the TerrainInstance class, that holds the underlying
    heightmap Callable"""

    def __init__(
        self,
        mesh: list[tuple[Trimesh, list[tuple[str, str]]]],
        origin: tuple[float, float, float],
        size: tuple[float, float],
        raw: Terrain,
        traversability_cfg: TraversabilityConfig | None = None,
    ):
        """Initialize the HeightmapTerrain instance.

        Args:
            mesh (list[tuple[Trimesh, list[tuple[str, str]]]]): A list of meshes with their tags.
            origin (tuple[float, float, float]): The origin of the terrain.
            size (tuple[float, float]): The size of the terrain.
            raw (Terrain): The encapsulated logical heightmap.
        """
        super().__init__(mesh, origin, size, GRASS_BASE_COLOR)
        self.raw = raw
        self.traversability_cfg = traversability_cfg or TraversabilityConfig()
        self.traversability_map = TraversabilityMapBuilder(
            raw,
            resolution_factor=self.traversability_cfg.resolution_factor,
            max_slope_deg=self.traversability_cfg.max_slope_deg,
        )


class ForestGenSpec(SceneSpec):
    """A specification for generating a forest scene."""

    def __init__(
        self,
        size: int = 256,
        margin: int = 10,
        traversability_cfg: TraversabilityConfig | None = None,
    ):
        """Initialize the forest generation specification.

        Args:
            size (int): The size of the terrain.
            robot (AssetBaseCfg | None): The robot configuration.
        """

        # here the assets are hooked up to the scene
        super().__init__(
            size=(size, size),
            palette=[PlantSpec(origin_margin=margin)],
        )
        self.side = size
        self.origin = (
            random.randint(margin, size - margin),
            random.randint(margin, size - margin),
        )
        self.traversability_cfg = traversability_cfg or TraversabilityConfig()

    def generate(self) -> HeightmapTerrain:
        # please note how we return a custom subclass that holds extra data,
        # so that the hooked up asset classes can depend on that extra data

        generator = (
            TerrainBuilder()
            .with_noise("simplex")
            .with_microrelief(True)
            .with_moisture_model({})
            .build()
        )
        terrain = generator.generate(TerrainConfig(self.side, 0.1))

        return HeightmapTerrain(
            terrain.to_meshes(classify_terrain),
            (self.origin[0], self.origin[1], terrain(*self.origin) + 1.0),
            self.size,
            terrain,
            self.traversability_cfg,
        )


class PlantSpec(AssetSpec):
    """Specification for generating all plant assets in a forest scene. One Spec to rule them all."""

    def __init__(
        self,
        sim_duration: int = 10,
        scene_density: float = 1.0,
        origin_margin: float = 10.0,
    ):
        """Construct a PlantSpec.

        Args:
            sim_duration (int, optional): The duration in years of the simulation used for tree position generation. Defaults to 10.
            scene_density (float, optional): Global density multiplier applied to the generated scene. Defaults to 1.0.
            origin_margin (float, optional): The margin around the origin for generating assets. Defaults to 10.0.
        """
        super().__init__("all")
        self.forest_cfg = ForestConfig(scene_density, sim_duration)
        self.origin_margin = origin_margin

    def generate(self, terrain: HeightmapTerrain) -> list[AssetInstance]:
        """Generate a list of instances based on the given terrain.

        Args:
            terrain (HeightmapTerrain): The terrain instance on which to generate.

        Returns:
            list[AssetInstance]: A list of generated grass asset instances.
        """

        # List for all assets
        AssetList = []

        # create factory for assets
        model_factory = PlantModelFactory()

        forest = (
            ForestBuilder()
            .with_size(terrain.size)
            .with_terrain(terrain.raw)
            .add_species("trees", Species("Oak", 10, 0.005, radius=5.0))
            .build()
        )

        # do the trees simulation
        logger.debug("Starting Tree simulation")
        state = forest.generate(self.forest_cfg)
        logger.debug("Tree simulation finished")

        origin_2d = (terrain.origin[0], terrain.origin[1])
        # then we create the tree instances
        obstacles: list[tuple[float, float]] = []

        for i, plant in enumerate(state):
            if math.dist(plant.coords, origin_2d) > self.origin_margin:

                obstacles.append(plant.coords)

                AssetList.append(
                    self.create_instance(
                        f"{plant.species.name}_{i}",
                        model_factory.get_model(plant),
                        (
                            plant.coords[0],
                            plant.coords[1],
                            terrain.raw(*plant.coords),
                        ),
                        (0.70711, 0.70711, 0.0, 0.0),
                        {"color": "green", "species": plant.species.name},
                    )
                )

        if obstacles:
            terrain.traversability_map.add_obstacle_score(
                obstacles,
                obstacle_influence_radius=terrain.traversability_cfg.obstacle_influence_radius,
                obstacle_penalty=terrain.traversability_cfg.obstacle_penalty,
            )

        # do the grass simulation
        logger.debug("Generating grass")
        grass_state = GrassDistributor(
            terrain.raw, [plant.coords for plant in state]
        ).generate(ForestConfig(self.forest_cfg.scene_density * 3.0, 0))
        logger.debug("Grass generation finished")

        for i, plant in enumerate(grass_state):
            if math.dist(plant.coords, origin_2d) <= self.origin_margin:
                continue

            AssetList.append(
                self.create_instance(
                    f"Grass_{i}",
                    model_factory.get_usdz_model_by_name("GrassBed", 1),
                    (
                        plant.coords[0],
                        plant.coords[1],
                        terrain.raw(*plant.coords),
                    ),
                    (0.0, 0.0, 0.0, 0.0),  # for glb (0.70711, 0.70711, 0.0, 0.0),
                    {"color": "blue", "species": plant.species.name},
                )
            )

        # do the fern simulation (simple test for now, copy from grass)
        # do the grass simulation
        logger.debug("Generating ferns")
        unfiltered_ferns = grass_points(
            int(terrain.size[0]), int(terrain.size[1]), 3.0
        )
        ferns = remove_grass_near_tree(
            unfiltered_ferns, [plant.coords for plant in state]
        )
        logger.debug("Ferns generation finished")

        for i, plant in enumerate(ferns):
            cls = classify_terrain(plant[0], plant[1])
            AssetList.append(
                self.create_instance(
                    f"Fern{i}",
                    model_factory.get_model_by_name("Fern", 1),
                    (plant[0], plant[1], terrain.raw(*plant)),
                    (0.70711, 0.70711, 0.0, 0.0),
                    {"color": "red", "species": "Fern"},
                )
            )

        return AssetList
