from logging import getLogger
import math
import os
import random
from collections import Counter
from logging import getLogger

logger = getLogger(__name__)

from neuroforgelab import (
    SceneSpec,
    AssetSpec,
    AssetInstance,
    TerrainInstance,
)

from .asset_dist import (
    Species,
    grass_cover,
    grass_points,
    remove_grass_near_tree,
)
from .assets import PlantModelFactory
from .forest import ForestBuilder, ForestConfig

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
GRASS_BASE_MATERIAL = "../forest-gen/models/materials/Ground/Mulch.mdl"


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
            .with_noise("fractal")
            .with_microrelief(True)
            .with_moisture_model({})
            .build()
        )
        terrain_cfg = TerrainConfig(size=self.side, resolution=0.25, scale=4.0, octaves=3, height_scale=5, apply_microrelief=True)
        terrain_classic = TerrainConfig(self.side, 0.5, height_scale=20)
        terrain = generator.generate(terrain_cfg)

        return HeightmapTerrain(
            terrain.to_meshes(classify_terrain, face_varying_uv=True),
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

        birch = Species(
            name='Birch',
            max_age=120,
            species_density=0.012,
            reproduction_rate=2,
            reproduction_radius=10.0,
            radius=1.9,
            juvenile_mortality_depth=0.45,
            juvenile_mortality_peak=0.07,
            juvenile_mortality_width=0.05,
            juvenile_recovery_age=0.18,
        )
        pine = Species(
            name='Pine',
            max_age=110,
            species_density=0.018,
            reproduction_rate=6,
            reproduction_radius=22.0,
            radius=1.3,
            juvenile_mortality_depth=0.3,
            juvenile_mortality_peak=0.05,
            juvenile_mortality_width=0.035,
            juvenile_recovery_age=0.15,
        )
        # create factory for assets
        model_factory = PlantModelFactory()

        forest = (
            ForestBuilder()
            .with_size(terrain.size)
            .with_terrain(terrain.raw)
            .add_species("trees", Species("Pine", 10, 0.005, radius=2.0))
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
                        model_factory.get_usdz_model_by_name(plant.species.name, random.randint(1,3)),
                        (
                            plant.coords[0],
                            plant.coords[1],
                            terrain.raw(*plant.coords),
                        ),
                        (0.0, 0.0, 0.0, 0.0),
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

        grass = grass_cover(
            int(terrain.size[0]), int(terrain.size[1]), 0.45
        )

        # old grass distr when we hoped for terrain mesh textures
        #
        # unfiltered_grass = grass_points(
        #     int(terrain.size[0]), int(terrain.size[1]), 0.5
        # )
        # grass = remove_grass_near_tree(
        #     unfiltered_grass, [plant.coords for plant in state]
        # )
        logger.debug("Grass generation finished")

        for i, plant in enumerate(grass_state):
            if math.dist(plant.coords, origin_2d) <= self.origin_margin:
                continue

            AssetList.append(
                self.create_instance(
                    f"Grass_{i}",
                    model_factory.get_usdz_model_by_name(
                        "Grass", 1
                    ),
                    (plant[0], plant[1], terrain.raw(*plant)-0.1),
                    (0.0, 0.0, 0.0, 0.0),           # for glb (0.70711, 0.70711, 0.0, 0.0),
                    {"color": "blue", "species": "Grass"},
                )
            )

        # Do the fern simulation
        logger.debug("Generating ferns")
        unfiltered_ferns = grass_points(
            int(terrain.size[0]), int(terrain.size[1]), 5.0
        )
        ferns = remove_grass_near_tree(
            unfiltered_ferns, [plant.coords for plant in state]
        )
        logger.debug("Ferns generation finished")

        for i, plant in enumerate(ferns):
            cls = classify_terrain(plant[0], plant[1])
            AssetList.append(
                self.create_instance(
                    f"Fern_{i}",
                    model_factory.get_usdz_model_by_name("Fern", random.randint(1,3)),
                    (plant[0], plant[1], terrain.raw(*plant)),
                    (0.0, 0.0, 0.0, 0.0),
                    {"color": "red", "species": "Fern"},
                )
            )

        # Do the bush simulation
        logger.debug("Generating bushes")
        unfiltered_bushes = grass_points(
            int(terrain.size[0]), int(terrain.size[1]), 3.0
        )
        # bushes = remove_grass_near_tree(
        #     unfiltered_bushes, [plant.coords for plant in state]
        # )
        logger.debug("Bushes generation finished")

        for i, plant in enumerate(unfiltered_bushes):
            cls = classify_terrain(plant[0], plant[1])
            AssetList.append(
                self.create_instance(
                    f"Bush_{i}",
                    model_factory.get_usdz_model_by_name("Bush", 1),
                    (plant[0], plant[1], terrain.raw(*plant)),
                    (0.0, 0.0, 0.0, 0.0),
                    {"color": "purple", "species": "Bush"},
                )
            )

        logger.debug(f"{dict(Counter(ass.name.split('_', 1)[0] for ass in AssetList))}")
        return AssetList
