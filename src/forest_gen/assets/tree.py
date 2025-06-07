# dead procedural tree experiment

# conda install openalea.core -c fredboudon -c conda-forge
# conda install openalea.weberpenn -c conda-forge -c openalea3
# conda install openalea.plantgl -c fredboudon -c conda-forge

from openalea.weberpenn.wralea.trunk_parameters import weber_penn
from openalea.weberpenn.tree_client import TreeParameter

# https://github.com/openalea/weberpenn/blob/88fc9c3f26c558ad77ca22ce0b0390ba0d7ee32a/src/openalea/weberpenn/tree_client.py#L874


def generate_tree(
    shape_id: int = 2,
    base_size: float = 0.05,
    scale: tuple[int, int] = (10, 10),
    order: int = 3,
    ratio: float = 0.018,
    ratio_power: float = 1.3,
    lobes: tuple[int, float] = (5, 0.1),
    flare: float = 1.2,
    base_split: int = 2,
    n_length: list[tuple[float, float]] = [
        (1, 0),
        (0.8, 0.1),
        (0.2, 0.05),
        (0.4, 0),
    ],
    n_seg_split: list[float] = [0.4, 0.2, 0.1, 0],
    n_split_angle: list[tuple[int, int]] = [
        (10, 0),
        (10, 10),
        (10, 10),
        (0, 0),
    ],
    n_down_angle: list[tuple[int, int]] = [
        (0, 0),
        (30, -30),
        (45, 10),
        (45, 10),
    ],
    n_curve: list[tuple[int, int, int, int]] = [
        (8, 0, 0, 90),
        (10, 40, -70, 150),
        (3, 0, 0, -30),
        (1, 0, 0, 0),
    ],
    n_rotate: list[tuple[int, int]] = [(80, 0), (140, 0), (140, 0)],
    n_branches: list[int] = [40, 120, 0],
    leaves: int = 25,
    leaf_scale: float = 0.12,
    leaf_scale_x: float = 0.66,
) -> None:

    # https://github.com/openalea/weberpenn/blob/88fc9c3f26c558ad77ca22ce0b0390ba0d7ee32a/src/openalea/weberpenn/wralea/trunk_parameters.py#L208

    sim = weber_penn(
        parameters=TreeParameter(
            shape_id,
            base_size,
            scale,
            order,
            ratio,
            ratio_power,
            lobes,
            flare,
            base_split,
            n_length,
            n_seg_split,
            n_split_angle,
            n_down_angle,
            n_curve,
            n_rotate,
            n_branches,
            leaves,
            leaf_scale,
            leaf_scale_x,
            rotate=[0] * (order + 1),
        ),
        seed=42,
        position=None,
    )

    if sim is None:
        raise ValueError("Simulation failed")

    if len(sim) != 1:
        raise ValueError("Unexpected number of simulations")

    sim[0].save("tree.obj")


generate_tree()
