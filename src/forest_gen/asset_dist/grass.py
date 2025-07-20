from scipy.stats.qmc import PoissonDisk
import math

from ..heightmap import normalized_noise2


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
        rng=1,
        ncandidates=30,
        l_bounds=[0, 0],
        u_bounds=[width, height],
    )
    grass = grass_sampler.random(n=int(width * height / (r * r)))
    return [tuple(point) for point in grass.tolist()]


def grass_distribution(width: int, height: int) -> list:
    """Distribute grass points based on terrain classification.

    Args:
        width (int): The width of the area.
        height (int): The height of the area.

    Returns:
        list: List of distributed grass points.
    """
    grass_plain = grass_points(width, height, 0.75)
    grass_forest = grass_points(width, height, 2)

    grass_end = []
    for item in grass_plain:
        if classify_terrain(item[0], item[1]) == "plain":
            grass_end.append(item)
    for item in grass_forest:
        if classify_terrain(item[0], item[1]) == "forest":
            grass_end.append(item)
    return grass_end


# Dunno czy to tu powinno być, czy gdzieś w heightmap
def classify_terrain(x: float, y: float) -> str:
    """Classify the terrain based on the x and y coordinates.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.

    Returns:
        str: The classification of the terrain.
    """
    if normalized_noise2(x, y) > 0.5:
        return "forest"
    return "plain"

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