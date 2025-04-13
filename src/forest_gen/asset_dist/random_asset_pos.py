import random

random.seed(0)

def gen_grid(size: int, step_size: int) -> list[tuple[int, int]]:
    # Generate grid of 100 squares each beeing 10% of size given to the function
    # (if size is not divisible by 10, function leaves out the difference (int amiright?
    grid = []
    step = int(size / step_size)
    for i in range(0, size, step):
        for j in range(0, size, step):
            if i + step < size and j + step < size:
                grid.append((i, j))
    return grid


def gen_coords(size: int) -> list[tuple[int, int]]:
    # Generate coordinates for assets in a grid leaving 20% space between edges of the grid
    grid = gen_grid(size, 5)
    coords = []
    step = int(size / 10)
    for item in grid:
        xmin = item[0] + int(step / 5)
        xmax = item[0] + step - int(step / 5)
        ymin = item[1] + int(step / 5)
        ymax = item[1] + step - int(step / 5)
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        coords.append((x, y))
    return coords

def gen_float_coords(size: int) -> list[tuple[float, float]]:
    # Generate coordinates for assets in a grid leaving 20% space between edges of the grid
    grid = gen_grid(size, 5)
    coords = []
    step = int(size / 10)
    for item in grid:
        xmin = item[0] + int(step / 5)
        xmax = item[0] + step - int(step / 5)
        ymin = item[1] + int(step / 5)
        ymax = item[1] + step - int(step / 5)
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        coords.append((float(x), float(y)))
    return coords
