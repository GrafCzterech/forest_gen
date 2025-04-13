from dataclasses import dataclass
import random
import math

@dataclass
class Species:
    name: str
    max_age: int

# radius for tree is its minimal distance from other trees
@dataclass
class Tree:
    coords: tuple[float, float]
    species: Species
    radius: float
    age: int

def seed(assets: list[Tree], number: int, radius: float, size: int) -> list[Tree]:
    # put number of trees around each tree in set radius
    assets_new = []
    for item in assets:
        for i in range(number):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius)
            new_x = item.coords[0] + distance * math.cos(angle)
            new_y = item.coords[1] + distance * math.sin(angle)
            assets_new.append(Tree((new_x, new_y), item.species, item.radius, 0))
    # return only new trees
    return assets_new

def gen_asset_pos(size: int, start_coords: list[tuple[float, float]], iterations: int) -> list[Tree]:
    pine = Species("Pine", 40)
    assets = []
    for item in start_coords:
        assets.append(Tree(item, pine, 3.0, 0))
    for i in range(iterations):
        assets = seed(assets, 1, 50.0, size)

    return assets