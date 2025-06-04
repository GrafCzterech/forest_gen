from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.polynomial import Polynomial

Func: TypeAlias = Callable[..., float]

# this entire module is a utility for the terrain generation
# YOU DONT NEED TO LOOK AT THIS


def call_func(
    func: Func,
    arg_transform: tuple[Polynomial | None, ...] | None,
    out_transform: Polynomial | None,
    *args: float,
) -> float:
    """Call a function with transformed arguments and output.

    Args:
        func (Func): Function to be called.
        arg_transform (tuple[Polynomial  |  None, ...] | None): Polynomials for transformations of each argument.
        out_transform (Polynomial | None): Transformation for the output.
        *args (float): Arguments to the function.

    Returns:
        float: Transformed function output.
    """
    if arg_transform is not None:
        value = func(
            *(
                trans(arg) if trans is not None else arg
                for arg, trans in zip(args, arg_transform)
            )
        )
    else:
        value = func(*args)

    if out_transform is not None:
        value: float = float(out_transform(value))

    return value


class MultiLayerFunc:
    """Multi-layer function class.
    This class allows you to 'stack' multiple transformations on a function.
    Each layer can have its own argument transformations and output transformations.
    """

    def __init__(self, func: Func):
        """Initialize the MultiLayerFunc with a function.

        Args:
            func (Func): Function to be transformed.
        """
        self.func = func
        self.layers: list[
            tuple[
                tuple[Polynomial | None, ...] | None,
                Polynomial | None,
            ]
        ] = []

    def add_step(
        self,
        arg_shift: tuple[np.ndarray | None, ...] | None,
        out_shift: np.ndarray | None,
    ) -> None:
        """Add a step to the function stack.

        Args:
            arg_shift (tuple[np.ndarray  |  None, ...]): Polynomial coefficients for transformations of each argument.
            out_shift (np.ndarray | None): Transformation for the output.
        """
        self.layers.append(
            (
                (
                    tuple(
                        Polynomial(arg_shift) if arg_shift is not None else None
                        for arg_shift in arg_shift
                    )
                    if arg_shift is not None
                    else None
                ),
                Polynomial(out_shift) if out_shift is not None else None,
            )
        )

    def __call__(self, *args: float) -> float:
        """Call the function with transformed arguments.

        Args:
            *args (float): Arguments to the function.

        Returns:
            float: Transformed function output.
        """
        value = 0.0
        for arg_transform, out_transform in self.layers:
            value += call_func(self.func, arg_transform, out_transform, *args)
        return value

    def __repr__(self) -> str:
        return f"MultiLayerFunc(func={self.func.__name__})"
