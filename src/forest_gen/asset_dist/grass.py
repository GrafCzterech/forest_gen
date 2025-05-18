from random import random
from scipy.stats import qmc

def grass_points(width, height, r):
    grass_sampler = qmc.PoissonDisk(2, radius=r, rng=1, ncandidates=30, l_bounds=[0, 0], u_bounds=[width, height])
    grass = grass_sampler.random(n=int(width * height / (r * r)))
    return grass