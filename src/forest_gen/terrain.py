from opensimplex import noise2
import numpy as np

from .func_stack import MultiLayerFunc


def normalized_noise2(x: float, y: float) -> float:
    """A wrapper for the noise2 function from opensimplex.
    This function normalizes the output to be between 0 and 1.

    Args:
        x (float): The x coordinate.
        y (float): The y coordinate.

    Returns:
        float: The normalized noise value.
    """
    return (noise2(x, y) + 1.0) / 2.0


IDENTITY_POLY = np.array((0.0, 1.0))


def generate_generic_step(
    div: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Generate a step for the noise function.

    Args:
        div (int): The divisor for the step size.

    The argument transformations are defined as:
        - For the first argument: (0.0, 1.0 / div^2)
        - For the second argument: (0.0, 1.0 / div^2)
        - For the output: (0.0, 10.0 * div - 1)

    So for div=1 we have just a plain noise function. For higher div values,
    the noise function steps(so potential jumps) are smaller, and get smaller
    faster than the multiplier for the output. So low div values give us chaotic
    but small noise, while high div values give us more stable but larger noise.
    This is useful for generating terrain with different levels of detail.

    Returns:
        tuple[tuple[float, ...], tuple[float, ...]]: The step for the noise function.
    """
    eps = 1.0 / div**2
    return (
        (IDENTITY_POLY * eps, IDENTITY_POLY * eps),
        IDENTITY_POLY * 1.0 * div,
    )


def shift_step(diff: float, base: float = 1.0) -> tuple[np.ndarray, ...]:
    """Return the argument transformations for the noise function, where the
    noise is shifted by the vector.

    Args:
        diff (float): The difference between the x and y shift

    Returns:
        tuple[np.ndarray, ...]: The argument transformations for the noise function.
    """
    return IDENTITY_POLY * (base + diff), IDENTITY_POLY * (base - diff)


NOISE_FUNC = MultiLayerFunc(normalized_noise2)
NOISE_FUNC.add_step(*generate_generic_step(1))
NOISE_FUNC.add_step(*generate_generic_step(2))
NOISE_FUNC.add_step(*generate_generic_step(4))
NOISE_FUNC.add_step(shift_step(0.005, 0.001), np.array((-3.5, 5.0)))
