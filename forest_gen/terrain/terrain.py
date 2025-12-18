from dataclasses import dataclass
from typing import Callable

import os
import numpy as np
from trimesh import Trimesh

from .terrain_config import TerrainConfig
from .mesh import heightmap_to_mesh, heightmap_to_meshes


@dataclass
class Terrain:
    config: TerrainConfig
    heightmap: np.ndarray
    flow: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    moisture: np.ndarray
    # materials: list[str] = ["Mulch", "Ground_Leaves_Oak"]  # FIXME: MAKE IT MORE DYNAMIC
    materials_path: str = "../forest-gen/models/materials/Ground"

    def __call__(self, x: float, y: float) -> float:
        return self.heightmap[
            self.config.transform(y), self.config.transform(x)
        ]

    def to_mesh(self) -> Trimesh:
        return heightmap_to_mesh(self, self.config.size, self.config.resolution)

    def to_meshes(
        self, classify: Callable[[float, float], str] | None = None
    ) -> list[tuple[Trimesh, list[tuple[str, str]]]]:
        return heightmap_to_meshes(
            self, self.config.size, self.config.resolution, classify
        )

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def size(self) -> tuple[float, float]:
        """Terrain dimensions in meters (width, height)."""
        return (self.config.size, self.config.size)
