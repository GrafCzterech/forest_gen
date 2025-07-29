from dataclasses import dataclass
from typing import Callable

import numpy as np
from trimesh import Trimesh

from .TerrainConfig import TerrainConfig


@dataclass
class Terrain:
    config: TerrainConfig
    heightmap: np.ndarray
    flow: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    moisture: np.ndarray

    def __call__(self, x: float, y: float) -> float:
        return self.heightmap[int(y), int(x)]

    def to_mesh(self) -> Trimesh:
        vertices = np.zeros(
            (self.heightmap.shape[0], self.heightmap.shape[1], 3)
        )
        vertices[:, :, 0] = np.arange(self.heightmap.shape[1])
        vertices[:, :, 1] = np.arange(self.heightmap.shape[0])[:, None]
        vertices[:, :, 2] = self.heightmap

        faces = np.zeros(
            (self.heightmap.shape[0] - 1, self.heightmap.shape[1] - 1, 2, 3),
            dtype=np.int32,
        )
        faces[:, :, 0, 0] = np.arange(self.heightmap.shape[1] - 1)[:, None]
        faces[:, :, 0, 1] = np.arange(self.heightmap.shape[0] - 1)
        faces[:, :, 0, 2] = np.arange(self.heightmap.shape[1] - 1)[:, None] + 1
        faces[:, :, 1, 0] = np.arange(self.heightmap.shape[1] - 1)[:, None] + 1
        faces[:, :, 1, 1] = np.arange(self.heightmap.shape[0] - 1)
        faces[:, :, 1, 2] = np.arange(self.heightmap.shape[1] - 1)[:, None]

        return Trimesh(vertices=vertices, faces=faces)
