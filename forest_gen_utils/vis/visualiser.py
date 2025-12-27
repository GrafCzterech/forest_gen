from abc import ABC, abstractmethod

import numpy as np


class Visualizer(ABC):
    """
    Strategy interface: render a single 2-D array with a title.
    Matches UML — only one abstract method.
    """

    @abstractmethod
    def visualize(self, data: np.ndarray, title: str) -> None: ...
