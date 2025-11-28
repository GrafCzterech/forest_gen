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

    def to_mesh(self, *, face_varying_uv: bool = False) -> Trimesh:
        """Create a single Trimesh from the heightmap.

        Args:
            face_varying_uv: If True, expand UVs to be face-varying which is often
                required by USD/MDL pipelines so texture coordinates are stored as
                face-varying primvars.

        Returns:
            Trimesh: The generated mesh
        """
        return heightmap_to_mesh(self, self.config.size, self.config.resolution, face_varying_uv=face_varying_uv)

    def to_meshes(
        self,
        classify: Callable[[float, float], str] | None = None,
        *,
        face_varying_uv: bool = False,
    ) -> list[tuple[Trimesh, list[tuple[str, str]]]]:
        """Create multiple meshes (semantic split) from the heightmap.

        Args:
            classify: Optional classifier function returning tag for each cell.
            face_varying_uv: If True, expand UVs to be face-varying for exporter compatibility.

        Returns:
            list[tuple[Trimesh, list[tuple[str, str]]]]: A list of meshes with tags.
        """
        return heightmap_to_meshes(
            self, self.config.size, self.config.resolution, classify, face_varying_uv=face_varying_uv
        )

    @property
    def __name__(self):
        return self.__class__.__name__
