from collections.abc import Callable
from typing import TypeAlias

import numpy as np

Func: TypeAlias = Callable[..., float]


def call_func(
    func: Func,
    arg_transform: tuple[np.ndarray | None, ...] | None,
    out_transform: np.ndarray | None,
    *args: float,
) -> float:
    """Call a function with transformed arguments and output.

    Args:
        func (Func): Function to call.
        arg_transform (tuple[Polynomial  |  None, ...] | None): Argument transformations.
        out_transform (Polynomial | None | None): Output transformation.

    Returns:
        float: Transformed function output.
    """
    if arg_transform is not None:
        args = tuple(
            (float(np.polyval(trans, arg)) if trans is not None else arg)
            for arg, trans in zip(args, arg_transform)
        )
    value = func(*args)
    if out_transform is not None:
        value = float(np.polyval(out_transform, value))
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
                tuple[np.ndarray | None, ...],
                np.ndarray | None,
            ]
        ] = []

    def add_step(
        self,
        arg_shift: tuple[np.ndarray | None, ...],
        out_shift: np.ndarray | None,
    ) -> None:
        """Add a step to the function stack.

        Args:
            arg_shift (tuple[tuple[float, ...]  |  None, ...]): Polynomial coefficients for transformations of each argument.
            out_shift (tuple[float, ...] | None): Transformation for the output.
        """
        self.layers.append((arg_shift, out_shift))

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
