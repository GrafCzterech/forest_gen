from collections.abc import Callable

import numpy as np
from trimesh import Trimesh


def generate_points(
    heightmap: Callable[[float, float], float], size: int, step: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points from a heightmap function.

    Args:
        heightmap (Callable[[float, float], float]): Function that takes x and y coordinates and returns height.
        size (int): Size of the mesh grid.
        step (float, optional): Step size for the grid. Defaults to 1.0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: X, Y, Z coordinates of the mesh grid.
    """
    x = y = np.linspace(0, size * step, size, retstep=False)
    X, Y = np.meshgrid(x, y)

    # Compute the heights
    Z = np.zeros_like(X)
    for i in range(size):
        for j in range(size):
            Z[i, j] = heightmap(X[i, j], Y[i, j])
    return X, Y, Z


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
    X, Y, Z = generate_points(heightmap, size, step)

    # Create the mesh
    vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    faces = []

    for i in range(size - 1):
        for j in range(size - 1):
            faces.append([i * size + j, i * size + (j + 1), (i + 1) * size + j])
            faces.append(
                (
                    (i + 1) * size + j,
                    i * size + (j + 1),
                    (i + 1) * size + (j + 1),
                )
            )

    faces = np.array(faces)

    return Trimesh(vertices=vertices, faces=faces)


def heightmap_to_meshes(
    heightmap: Callable[[float, float], float],
    size: int,
    step: float = 1.0,
    classifier: Callable[[float, float], str] | None = None,
) -> list[tuple[Trimesh, list[tuple[str, str]]]]:
    """Convert a heightmap function to a list of 3D meshes.

    Args:
        heightmap (Callable[[float, float], float]): Function that takes x and y coordinates and returns height.
        size (int): Size of the mesh grid.
        step (float, optional): Step size for the grid. Defaults to 1.0.
        classifier (Callable[[float, float], str], optional): Function that classifies the terrain. Defaults to None.

    Returns:
        list[tuple[Trimesh, list[tuple[str, str]]]]: A list of tuples containing the trimesh object and a list of tags.
    """
    if classifier is None:
        return [(heightmap_to_mesh(heightmap, size, step), [])]
    else:

        X, Y, Z = generate_points(heightmap, size, step)

        # Create the mesh
        vertices = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

        classes: dict[str, list[tuple[int, int, int]]] = {}

        for i in range(size - 1):
            for j in range(size - 1):
                class1 = classifier(X[i, j], Y[i, j])

                faces = classes.get(class1, [])

                faces.append(
                    (i * size + j, i * size + (j + 1), (i + 1) * size + j)
                )
                faces.append(
                    (
                        (i + 1) * size + j,
                        i * size + (j + 1),
                        (i + 1) * size + (j + 1),
                    )
                )

                classes[class1] = faces

    return [
        (Trimesh(vertices=vertices, faces=faces), [("terrain_class", tag)])
        for tag, faces in classes.items()
    ]
