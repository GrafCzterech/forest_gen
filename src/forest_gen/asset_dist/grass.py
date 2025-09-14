from scipy.stats.qmc import PoissonDisk
import math


def grass_points(width: int, height: int, r: float):
    """Returns a list of points representing simple grass distribution over a given area.

    Args:
        width (int): The width of the area.
        height (int): The height of the area.
        r (float): Minimal radius between points.

    Returns:
        grass (ndarray): An array of points representing grass distribution.
    """
    grass_sampler = PoissonDisk(
        2,
        radius=r,
        ncandidates=30,
        l_bounds=[0, 0],
        u_bounds=[width, height],
    )
    grass = grass_sampler.random(n=int(width * height / (r * r)))
    return [tuple(point) for point in grass.tolist()]


def remove_grass_near_tree(grass: list, trees: list) -> list:
    """Remove grass points that are too close to a tree.

    Args:
        grass (list): The list of grass points.
        trees (list): The list of trees (x, y).

    Returns:
        list: The filtered list of grass points.
    """
    grass_filtered = []

    for grass_point in grass:
        check = True
        for tree in trees:
            if math.dist(tree, grass_point) < 2:
                check = False
                break
        if check:
            grass_filtered.append(grass_point)

    return grass_filtered
