import random


def gen_grid(size: int) -> list[tuple[int]]:
    # Generate grid of 100 squares each beeing 10% of size given to the function
    # (if size is not divisible by 10, function leaves out the difference (int amiright?
    grid = []
    step = int(size/10)
    for i in range(0, size, step):
        for j in range(0, size, step):
            if i + step < size and j + step < size:
                grid.append((i, j))
    return grid

def gen_coords(size: int) -> list[tuple[int]]:
    # Generate coordinates for assets in a grid leaving 20% space between edges of the grid
    grid = gen_grid(size)
    coords = []
    step = int(size/10)
    for item in grid:
        xmin = item[0] + int(step/5)
        xmax = item[0] + step - int(step/5)
        ymin = item[1] + int(step/5)
        ymax = item[1] + step - int(step/5)
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        coords.append((x, y))
    return coords

# linux test