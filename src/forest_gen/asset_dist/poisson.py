from scipy.stats.qmc import PoissonDisk

import numpy as np

from opensimplex import OpenSimplex

# Old scribble from Tomek

# first we create a grid
# then a map of grid cell -> radius
# then for each cell we create a poisson disk
# then we create a list of points using that disk
# and then we cull that disk using the pmap
# then we return the points

# OR we have a PMap of radii, that can be larger than the cell thus sometimes yielding empty cells


def sample_positions(
    width: float,
    height: float,
    base_radius: float,
    eps: float,
    seed: int = 0,
) -> list[tuple[float, float]]:
    noise = OpenSimplex(seed=seed)
    res = []
    # split the area into a grid
    for x in np.arange(0, width, eps):
        for y in np.arange(0, height, eps):
            # get the radius of the cell
            radius = (
                base_radius + noise.noise2(x / eps, y / eps) * base_radius / 2.0
            )
            # create a poisson disk
            poisson_disk = PoissonDisk(
                d=2,
                radius=radius,
                #l_bounds=[x, y],
                #u_bounds=[x + eps, y + eps],
                #rng=seed,
            )
            # sample points
            for point in poisson_disk.fill_space():
                res.append((point[0], point[1]))
    return res
