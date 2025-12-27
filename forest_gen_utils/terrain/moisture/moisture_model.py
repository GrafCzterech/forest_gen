from abc import ABC, abstractmethod

import numpy as np


class MoistureModel(ABC):
    """
    Strategy interface for terrain moisture computation.

    Defines a common interface for models that derive a moisture index
    from hydrological flow, slope, and aspect data.
    """

    @abstractmethod
    def compute(
        self, flow: np.ndarray, slope: np.ndarray, aspect: np.ndarray
    ) -> np.ndarray:
        """
        Compute a moisture index from terrain-derived inputs.

        :param flow: Flow accumulation or drainage intensity array.
        :type flow: numpy.ndarray
        :param slope: Terrain slope array.
        :type slope: numpy.ndarray
        :param aspect: Terrain aspect array.
        :type aspect: numpy.ndarray
        :return: Moisture index array.
        :rtype: numpy.ndarray
        """
        pass
