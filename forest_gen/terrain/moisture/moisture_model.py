import numpy as np
from abc import ABC, abstractmethod


class MoistureModel(ABC):
    """Interface for moisture calculation based on flow, slope, and aspect."""

    @abstractmethod
    def compute(
        self, flow: np.ndarray, slope: np.ndarray, aspect: np.ndarray
    ) -> np.ndarray:
        """Compute moisture index from flow, slope, and aspect arrays."""
        pass
