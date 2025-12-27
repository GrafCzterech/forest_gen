from typing import Tuple

import numpy as np


class SlopeAspectCalculator:
    """
    Compute slope and aspect from a terrain heightmap.

    Slope is returned in degrees. Aspect is returned in degrees
    clockwise from north (``[0, 360)``).
    """

    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution

    def compute(self, heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute slope and aspect for each cell of a heightmap.

        Central differences are used to estimate height gradients.

        :param heightmap: 2D terrain heightmap.
        :type heightmap: numpy.ndarray
        :return: Tuple ``(slope, aspect)`` in degrees.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        rows, cols = heightmap.shape
        slope = np.zeros((rows, cols), dtype=np.float32)
        aspect = np.zeros((rows, cols), dtype=np.float32)
        for y in range(rows):
            for x in range(cols):
                xm, xp = max(x - 1, 0), min(x + 1, cols - 1)
                ym, yp = max(y - 1, 0), min(y + 1, rows - 1)
                dzdx = (heightmap[y, xp] - heightmap[y, xm]) / (2 * self.resolution)
                dzdy = (heightmap[yp, x] - heightmap[ym, x]) / (2 * self.resolution)
                # slope is the steepest descent angle
                slope[y, x] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
                # aspect: compass direction the slope faces
                aspect[y, x] = (np.degrees(np.arctan2(dzdy, -dzdx)) + 360) % 360
        return slope, aspect
