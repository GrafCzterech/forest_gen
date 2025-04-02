from collections.abc import Callable

import numpy as np
from trimesh import Trimesh


def heightmap_to_mesh(
    heightmap: Callable[[float, float], float], size: int, step: float = 1.0
) -> Trimesh:
    """Convert a heightmap function to a 3D mesh.

    Args:
        heightmap (Callable[[float, float], float]): Function that takes x and y coordinates and returns height.
        size (int): Size of the mesh grid.

    Returns:
        Trimesh: A trimesh object representing the 3D mesh.
    """
    # Create a grid of points
    x = y = np.linspace(0, size * step, size)
    X, Y = np.meshgrid(x, y)

    # Compute the heights
    Z = np.zeros_like(X)
    for i in range(size):
        for j in range(size):
            Z[i, j] = heightmap(X[i, j], Y[i, j])

    # Create the mesh
    vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    faces = []

    for i in range(size - 1):
        for j in range(size - 1):
            faces.append([i * size + j, i * size + (j + 1), (i + 1) * size + j])
            faces.append(
                [
                    (i + 1) * size + j,
                    i * size + (j + 1),
                    (i + 1) * size + (j + 1),
                ]
            )

    faces = np.array(faces)

    return Trimesh(vertices=vertices, faces=faces)
