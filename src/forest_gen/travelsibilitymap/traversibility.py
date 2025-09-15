import numpy as np
from trimesh import Trimesh
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator

from neuroforgelab import TerrainInstance


def compute_slope_per_vertex(mesh: Trimesh) -> np.ndarray:
    vertex_normals = mesh.vertex_normals
    slope = np.arccos(np.clip(vertex_normals[:, 2], -1.0, 1.0))
    return slope  # RADIANY!


class TraversabilityMapBuilder:
    def __init__(self, terrain: TerrainInstance, step: float, resolution_factor: int = 2, max_slope_deg: float = 30.0):
        """Initializes the TraversabilityMapBuilder

        Args:
            mesh (Trimesh): The mesh to compute the traversability map for.
            size (int): The size of the traversability map.
            step (float): The step size of the traversability map.
            resolution_factor (int, optional): The resolution factor of the traversability map. Defaults to 2.
            max_slope_deg (float, optional): The maximum slope in degrees. Defaults to 30.0.
        """
        self.high_res_size_x = int(round(terrain.size[0] * resolution_factor))
        self.high_res_size_y = int(round(terrain.size[1] * resolution_factor))

        i = iter(terrain.mesh)
        mesh = next(i)[0]
        for pair in i:
            mesh += pair[0]

        x = np.linspace(0, terrain.size[0] * step, self.high_res_size_x)
        y = np.linspace(0, terrain.size[1] * step, self.high_res_size_y)
        self.X, self.Y = np.meshgrid(x, y)

        base_grid_x = np.linspace(0, terrain.size[0] * step, self.high_res_size_x)
        base_grid_y = np.linspace(0, terrain.size[1] * step, self.high_res_size_y)
        Z_interp = RegularGridInterpolator(
            (base_grid_x, base_grid_y), mesh.vertices[:, 2].reshape(int(terrain.size[0]), int(terrain.size[1]))
        )
        points = np.c_[self.X.ravel(), self.Y.ravel()]
        self.Z = Z_interp(points).reshape(self.high_res_size_x, self.high_res_size_y)

        slope_rad = compute_slope_per_vertex(mesh).reshape((int(terrain.size[0]), int(terrain.size[1])))
        slope_interp = RegularGridInterpolator((base_grid_x, base_grid_y), slope_rad)
        slope_highres = slope_interp(points).reshape(self.high_res_size_x, self.high_res_size_y)

        max_slope_rad = np.radians(max_slope_deg)
        self.score = 1.0 - np.clip(slope_highres / max_slope_rad, 0, 1)

    def add_obstacle_score(self, obstacles: list[tuple[float, float]], obstacle_influence_radius: float = 10.0, obstacle_penalty: float = 0.5,) -> None:
        """Adds an obstacle score to the traversability map.

        Args:
            obstacles (list[tuple[float, float]]): The list of obstacles. Really a list of 2D points.
            obstacle_influence_radius (float): The radius of influence of each obstacle.
            obstacle_penalty (float): The penalty for each obstacle.
        """
        tree_map = np.ones((self.high_res_size_x, self.high_res_size_y))
        tree_points = np.array(obstacles)
        kdtree = KDTree(tree_points)
        for i in range(self.high_res_size_x):
            for j in range(self.high_res_size_y):
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

    def add_alt_score(self, min_altitude: float = 0.0, max_altitude: float = 100.0, altitude_penalty: float = 0.3, altitude_weight: float = 1.0) -> None:
        """Adds an altitude score to the traversability map.

        Args:
            min_altitude (float): The minimum altitude.
            max_altitude (float): The maximum altitude.
            altitude_penalty (float): The penalty for each altitude.
            altitude_weight (float): The weight of the altitude score.
        """
        alt_score = np.ones_like(self.Z)
        if min_altitude is not None or max_altitude is not None:
            mask = np.zeros_like(self.Z, dtype=bool)
            if min_altitude is not None:
                mask |= self.Z < min_altitude
            if max_altitude is not None:
                mask |= self.Z > max_altitude
            alt_score[mask] -= altitude_penalty

        self.score = self.score * (1.0 - altitude_weight) + altitude_weight * alt_score

    def get_score(self) -> np.ndarray:
        return np.clip(self.score, 0.0, 1.0)
