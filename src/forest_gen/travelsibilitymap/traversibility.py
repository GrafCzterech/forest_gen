import numpy as np
from trimesh import Trimesh
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator

from ..terrain import Terrain

def compute_slope_per_vertex(mesh: Trimesh) -> np.ndarray:
    vertex_normals = mesh.vertex_normals
    slope = np.arccos(np.clip(vertex_normals[:, 2], -1.0, 1.0))
    return slope  # RADIANY!


class TraversabilityMapBuilder:
    def __init__(self, terrain: Terrain, resolution_factor: int = 2, max_slope_deg: float = 30.0):
        """Initializes the TraversabilityMapBuilder

        Args:
            terrain (Terrain): The terrain instance to compute the traversability map for.
            resolution_factor (int, optional): The resolution factor of the traversability map. Defaults to 2.
            max_slope_deg (float, optional): The maximum slope in degrees. Defaults to 30.0.
        """
        size = terrain.config.size
        self.high_res_size = int(round(size * resolution_factor))

        mesh: Trimesh = terrain.to_mesh()

        x = y = np.linspace(0, size, self.high_res_size)
        self.X, self.Y = np.meshgrid(x, y)

        points = np.c_[self.X.ravel(), self.Y.ravel()]

        slope_rad = compute_slope_per_vertex(mesh).reshape((terrain.config.rows - 1, terrain.config.cols - 1))
        slope_interp = RegularGridInterpolator((np.linspace(0, size, terrain.config.rows - 1), np.linspace(0, size, terrain.config.cols - 1)), slope_rad)
        slope_highres = slope_interp(points).reshape(self.high_res_size, self.high_res_size)

        max_slope_rad = np.radians(max_slope_deg)
        self.score = 1.0 - np.clip(slope_highres / max_slope_rad, 0, 1)

    def add_obstacle_score(self, obstacles: list[tuple[float, float]], obstacle_influence_radius: float = 10.0, obstacle_penalty: float = 0.5,) -> None:
        """Adds an obstacle score to the traversability map.

        Args:
            obstacles (list[tuple[float, float]]): The list of obstacles. Really a list of 2D points.
            obstacle_influence_radius (float): The radius of influence of each obstacle.
            obstacle_penalty (float): The penalty for each obstacle.
        """
        tree_map = np.ones((self.high_res_size, self.high_res_size))
        tree_points = np.array(obstacles)
        kdtree = KDTree(tree_points)
        for i in range(self.high_res_size):
            for j in range(self.high_res_size):
                px, py = self.X[i, j], self.Y[i, j]
                indices = kdtree.query_ball_point(
                    [px, py], r=obstacle_influence_radius
                )
                if indices:
                    penalties = []
                    for idx in indices:
                        # Euklides
                        d = np.hypot(
                            px - tree_points[idx, 0], py - tree_points[idx, 1]
                        )
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
        return np.clip(self.score, 0.0, 1.0)
