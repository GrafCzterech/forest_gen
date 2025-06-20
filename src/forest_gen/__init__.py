"""
forest_gen – procedural forest-generation toolkit
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:         
    __version__ = "0.0.0.dev0"

from .temp import TerrainGenerator, TerrainConfig, TerrainBuilder
# from .asset_dist import AssetDistributor          # <- adjust name if needed
# from .heightmap import HeightMap
# from .travelvisibilitymap import TravelVisibilityMap  # if the module exists

__all__ = [
    "TerrainGenerator",
    "TerrainConfig",
    "TerrainBuilder",
    "__version__",
]

    # "AssetDistributor",
    # "HeightMap",
    # "TravelVisibilityMap",



# from .scene import ForestGenSpec

# __all__ = ["ForestGenSpec"]
