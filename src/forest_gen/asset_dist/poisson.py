from scipy.stats.qmc import PoissonDisk

from .p_map import PMap

# first we create a grid
# then a map of grid cell -> radius
# then for each cell we create a poisson disk
# then we create a list of points using that disk
# and then we cull that disk using the pmap
# then we return the points

# OR we have a PMap of radii, that can be larger than the cell thus sometimes yielding empty cells
