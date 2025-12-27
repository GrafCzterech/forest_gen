from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree
from trimesh import Trimesh

from .terrain import Terrain

"""
Traversability map construction utilities.

Computes slope-based traversability from terrain meshes and applies
additional penalties from spatial obstacles.
"""


def compute_slope_per_vertex(mesh: Trimesh) -> np.ndarray:
    """
    Compute per-vertex slope angles from a mesh.

    Slope is derived from the Z component of vertex normals.

    :param mesh: Terrain mesh.
    :type mesh: trimesh.Trimesh
    :return: Slope angles in radians.
    :rtype: numpy.ndarray
    """
    vertex_normals = mesh.vertex_normals
    slope = np.arccos(np.clip(vertex_normals[:, 2], -1.0, 1.0))
    return slope  # RADIANY!


@dataclass
class TraversabilityConfig:
    """
    Configuration parameters for traversability computation.
    """

    resolution_factor: int = 3
    """Upsampling factor relative to terrain resolution."""
    max_slope_deg: float = 30.0
    """Maximum traversable slope in degrees."""
    obstacle_influence_radius: float = 7.0
    """Radius of obstacle influence."""
    obstacle_penalty: float = 0.45
    """Maximum traversability penalty per obstacle."""


class TraversabilityMapBuilder:
    """
    Build a high-resolution traversability map from terrain data.
    """

    def __init__(
        self,
        terrain: Terrain,
        resolution_factor: int = 2,
        max_slope_deg: float = 30.0,
    ):
        """
        Initialize the traversability map builder.

        :param terrain: Terrain used for slope computation.
        :type terrain: Terrain
        :param resolution_factor: Upsampling factor for the output map.
        :type resolution_factor: int
        :param max_slope_deg: Maximum traversable slope in degrees.
        :type max_slope_deg: float
        """
        size = terrain.config.size
        self.high_res_size = int(round(size * resolution_factor))

        mesh: Trimesh = terrain.to_mesh()

        x = y = np.linspace(0, size, self.high_res_size)
        self.X, self.Y = np.meshgrid(x, y)

        points = np.c_[self.X.ravel(), self.Y.ravel()]

        slope_rad = compute_slope_per_vertex(mesh).reshape(
            (terrain.config.rows - 1, terrain.config.cols - 1)
        )
        slope_interp = RegularGridInterpolator(
            (
                np.linspace(0, size, terrain.config.rows - 1),
                np.linspace(0, size, terrain.config.cols - 1),
            ),
            slope_rad,
        )
        slope_highres = slope_interp(points).reshape(
            self.high_res_size, self.high_res_size
        )

        max_slope_rad = np.radians(max_slope_deg)
        self.score = 1.0 - np.clip(slope_highres / max_slope_rad, 0, 1)

    def add_obstacle_score(
        self,
        obstacles: list[tuple[float, float]],
        obstacle_influence_radius: float = 10.0,
        obstacle_penalty: float = 0.5,
    ) -> None:
        """
        Apply obstacle-based penalties to the traversability map.

        Each obstacle reduces traversability within a given radius,
        with penalty decreasing linearly with distance.

        :param obstacles: Obstacle positions as ``(x, y)`` coordinates.
        :type obstacles: list[tuple[float, float]]
        :param obstacle_influence_radius: Radius of obstacle influence.
        :type obstacle_influence_radius: float
        :param obstacle_penalty: Maximum penalty applied near obstacles.
        :type obstacle_penalty: float
        """
        tree_map = np.ones((self.high_res_size, self.high_res_size))
        tree_points = np.array(obstacles)
        kdtree = KDTree(tree_points)
        for i in range(self.high_res_size):
            for j in range(self.high_res_size):
                px, py = self.X[i, j], self.Y[i, j]
                indices = kdtree.query_ball_point([px, py], r=obstacle_influence_radius)
                if indices:
                    penalties = []
                    for idx in indices:
                        # Euklides
                        d = np.hypot(px - tree_points[idx, 0], py - tree_points[idx, 1])
                        # d = 0   --> penalty_value = tree_penalty (full penalty)
                        # d = tree_influence_radius --> penalty_value = tree_penalty * 0.3
                        penalty_value = obstacle_penalty * (
                            1 - 0.3 * (d / obstacle_influence_radius)
                        )
                        penalties.append(penalty_value)
                    max_penalty = max(penalties)
                    tree_map[i, j] = 1.0 - max_penalty

        self.score *= tree_map

    def get_score(self) -> np.ndarray:
        """
        Return the final traversability score map.

        :return: Traversability values in ``[0.0, 1.0]``.
        :rtype: numpy.ndarray
        """
        return np.clip(self.score, 0.0, 1.0)
