import numpy as np
from typing import List, Dict, Tuple
from trimesh import Trimesh
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator


def compute_slope_per_vertex(mesh: Trimesh) -> np.ndarray:
    vertex_normals = mesh.vertex_normals
    slope = np.arccos(np.clip(vertex_normals[:, 2], -1.0, 1.0))
    return slope  # RADIANY!


def generate_traversability_map(
    mesh: Trimesh,
    size: int,
    step: float,
    trees: List[Dict[str, float]],
    max_slope_deg: float = 30.0,
    slope_weight: float = 1.0,
    tree_penalty: float = 0.5,
    tree_influence_radius: float = 2.0,
    min_altitude: float = 0.0,
    max_altitude: float = 100.0,
    altitude_penalty: float = 0.3,
    altitude_weight: float = 0.5,
    resolution_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    high_res_size = size * resolution_factor

    # WYŻSZA ROZDIZELCZOŚĆ
    x = y = np.linspace(0, size * step, high_res_size)
    X, Y = np.meshgrid(x, y)

    base_grid = np.linspace(0, size * step, size)
    Z_interp = RegularGridInterpolator(
        (base_grid, base_grid), mesh.vertices[:, 2].reshape(size, size)
    )
    points = np.c_[X.ravel(), Y.ravel()]
    Z = Z_interp(points).reshape(high_res_size, high_res_size)

    slope_rad = compute_slope_per_vertex(mesh).reshape((size, size))
    slope_interp = RegularGridInterpolator((base_grid, base_grid), slope_rad)
    slope_highres = slope_interp(points).reshape(high_res_size, high_res_size)

    max_slope_rad = np.radians(max_slope_deg)
    slope_score = 1.0 - np.clip(slope_highres / max_slope_rad, 0, 1)

    tree_map = np.ones((high_res_size, high_res_size))
    if trees:
        tree_points = np.array([[t["x"], t["y"]] for t in trees])
        kdtree = KDTree(tree_points)
        for i in range(high_res_size):
            for j in range(high_res_size):
                px, py = X[i, j], Y[i, j]
                indices = kdtree.query_ball_point(
                    [px, py], r=tree_influence_radius
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
                        penalty_value = tree_penalty * (
                            1 - 0.3 * (d / tree_influence_radius)
                        )
                        penalties.append(penalty_value)
                    max_penalty = max(penalties)
                    tree_map[i, j] = 1.0 - max_penalty

    # Altitude
    alt_score = np.ones_like(Z)
    if min_altitude is not None or max_altitude is not None:
        mask = np.zeros_like(Z, dtype=bool)
        if min_altitude is not None:
            mask |= Z < min_altitude
        if max_altitude is not None:
            mask |= Z > max_altitude
        alt_score[mask] -= altitude_penalty

    # Połączenie, nie wiem jeszcze jak jest najlepiej
    traversability = (
        slope_weight * slope_score + (1 - slope_weight) * alt_score
    ) * tree_map

    return np.clip(traversability, 0.0, 1.0), Z
