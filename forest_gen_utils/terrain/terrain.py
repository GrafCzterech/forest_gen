import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from trimesh import Trimesh

from .mesh import heightmap_to_mesh, heightmap_to_meshes
from .terrain_config import TerrainConfig


@dataclass
class Terrain:
    """
    Container for generated terrain data and mesh export utilities.

    Holds all terrain-derived fields (height, flow, slope, aspect,
    moisture) together with helper methods for sampling and mesh export.
    """

    config: TerrainConfig
    """Terrain generation configuration."""
    heightmap: np.ndarray
    """Terrain heightmap."""
    flow: np.ndarray
    """Flow accumulation array."""
    slope: np.ndarray
    """Slope array in degrees."""
    aspect: np.ndarray
    """Aspect array in degrees."""
    moisture: np.ndarray
    """Moisture index array."""
    # materials: list[str] = ["Mulch", "Ground_Leaves_Oak"]  # FIXME: MAKE IT MORE DYNAMIC
    materials_path: str = "../forest-gen/models/materials/Ground"
    """Path to terrain material definitions."""

    def __call__(self, x: float, y: float) -> float:
        """
        Sample the terrain height at world-space coordinates.

        :param x: X coordinate in world units.
        :type x: float
        :param y: Y coordinate in world units.
        :type y: float
        :return: Height value at the given position.
        :rtype: float
        """
        return self.heightmap[self.config.transform(y), self.config.transform(x)]

    def to_mesh(self) -> Trimesh:
        """
        Convert the terrain heightmap to a single mesh.

        :return: Terrain mesh.
        :rtype: trimesh.Trimesh
        """
        return heightmap_to_mesh(self, self.config.size, self.config.resolution)

    def to_meshes(
        self, classify: Callable[[float, float], str] | None = None
    ) -> list[tuple[Trimesh, list[tuple[str, str]]]]:
        """
        Convert the terrain into multiple classified meshes.

        :param classify: Optional classifier mapping world coordinates
                         to material identifiers.
        :type classify: Callable[[float, float], str] or None
        :return: List of meshes with associated material assignments.
        :rtype: list[tuple[trimesh.Trimesh, list[tuple[str, str]]]]
        """
        return heightmap_to_meshes(
            self, self.config.size, self.config.resolution, classify
        )

    @property
    def __name__(self):
        """Return the class name."""
        return self.__class__.__name__

    @property
    def size(self) -> tuple[float, float]:
        """
        Terrain dimensions in units.

        :return: ``(width, height)``.
        :rtype: tuple[float, float]
        """
        return (self.config.size, self.config.size)
