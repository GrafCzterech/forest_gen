from typing import Callable
import logging

from neuroforgelab import (
    SceneSpec,
    AssetSpec,
    AssetInstance,
    TerrainInstance,
    UniversalMesh,
)

from trimesh import Trimesh

from .heightmap import NOISE_FUNC, heightmap_to_mesh
from .asset_dist import Simulation, Species
from .assets import TreeModelFactory


class HeightmapTerrain(TerrainInstance):

    def __init__(
        self,
        mesh: Trimesh,
        origin: tuple[float, float, float],
        size: tuple[float, float],
        raw: Callable[[float, float], float],
    ):
        super().__init__(mesh, origin, size)
        self.raw = raw


class ForestGenSpec(SceneSpec):
    def __init__(self, size: int = 256):
        super().__init__(size=(size, size), robot=None, palette=[])
        self.add_asset(TreeSpec())

    def generate(self) -> HeightmapTerrain:

        return HeightmapTerrain(
            heightmap_to_mesh(NOISE_FUNC, int(self.size[0])),
            (0.0, 0.0, 0.0),
            self.size,
            NOISE_FUNC,
        )


class TreeSpec(AssetSpec):
    """Specification for generating trees in a forest scene."""

    tree_species = {
        Species("Oak", 10, 0.005, radius=2.5),
    }

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
        logging.debug("Starting simulation")
        sim = Simulation(terrain.size, {self.name: self.tree_species})
        state = sim.new_state(self.tree_density)
        state.run_state(self.sim_duration)
        logging.debug("Simulation finished")
        model_factory = TreeModelFactory()
        return [
            self.create_instance(
                f"{plant.species.name}_{i}",
                model_factory.get_model(plant),
                (plant.coords[0], terrain.raw(*plant.coords), plant.coords[1]),
                (0.0, 0.0, 0.0, 0.0),
            )
            for i, plant in enumerate(state)
        ]
