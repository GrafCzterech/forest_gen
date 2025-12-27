"""Submodule providing asset distribution generation"""

from typing import TYPE_CHECKING

from .definitions import Plant, Species
from .sim import Simulation
from .state import SimulationState

if TYPE_CHECKING:
    from .grass import GrassDistributor, grass_points, remove_grass_near_tree


from .distribution_builder import DistributionBuilder
from .distribution_config import DistributionConfig
from .distribution_generator import DistributionGenerator
from .terrain_viability import TerrainViabilityMap

__all__ = [
    "Simulation",
    "Species",
    "Plant",
    "SimulationState",
    "TerrainViabilityMap",
    "DistributionBuilder",
    "DistributionConfig",
    "DistributionGenerator",
    "GrassDistributor",
    "grass_points",
    "remove_grass_near_tree",
]


def __getattr__(name: str):
    if name in {"GrassDistributor", "grass_points", "remove_grass_near_tree"}:
        from .grass import GrassDistributor, grass_points, remove_grass_near_tree

        return {
            "GrassDistributor": GrassDistributor,
            "grass_points": grass_points,
            "remove_grass_near_tree": remove_grass_near_tree,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
