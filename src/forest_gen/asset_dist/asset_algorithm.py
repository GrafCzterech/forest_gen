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

def calc_vt(tree: Tree) -> float:
    norm_age = tree.age / tree.species.max_age
    if norm_age < 0.5:
        return norm_age
    else:
        return 1 - norm_age

def seed(assets: list[Tree], radius: float, size: int, number: int) -> list[Tree]:
    # put number of trees around each tree in set radius
    assets_new = []
    for item in assets:
        item.age += 1
        # Ustawiłem że tylko drzewa starsze niż ćwierć maksymalnego wieku mogą się rozmnażać
        if item.age > item.species.max_age / 4:
            for i in range(number):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, radius)
                new_x = item.coords[0] + distance * math.cos(angle)
                new_y = item.coords[1] + distance * math.sin(angle)
                assets_new.append(Tree((new_x, new_y), item.species, item.radius, 0))
    assets_new = assets_new + assets
    return assets_new

def gen_asset_pos(size: int, start_coords: list[tuple[float, float]], iterations: int) -> list[Tree]:
    pine = Species("Pine", 50)
    assets = []
    # create a list of trees from starting coordinates
    for item in start_coords:
        assets.append(Tree(item, pine, 10.0, 0))
    
    # iterate
    for i in range(iterations):
        number = random.randrange(0, 5)
        assets = seed(assets, 20.0, size, number)
        length = len(assets)
        # remove trees that died of old age
        i = 0
        while i < length:
            item = assets[i]
            if item.age > item.species.max_age:
                assets.remove(item)
                length -= 1
                i -= 1
            i += 1
        # remove trees that are outside the area
        i = 0
        while i < length:
            item = assets[i]
            if item.coords[0] < 0 or item.coords[0] > float(size) or item.coords[1] < 0 or item.coords[1] > float(size):
                assets.remove(item)
                length -= 1
                i -= 2
            i += 1
        # remove trees that are too close to each other
        i = 0
        while i < length:
            item = assets[i]
            j = i + 1
            while j < length:
                other = assets[j]
                distance = math.dist(item.coords, other.coords)
                if distance < max(item.radius, other.radius):
                    vt1 = calc_vt(item)
                    vt2 = calc_vt(other)
                    if vt1 < vt2:
                        assets.remove(item)
                        length -= 1
                        i -= 1
                        break
                    else:
                        assets.remove(other)
                    length -= 1
                else:
                    j += 1
            i += 1



    return assets