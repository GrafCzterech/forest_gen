from typing import Tuple
import numpy as np


class SlopeAspectCalculator:
    """Calculates slope (in degrees) and aspect (in degrees) for each cell of a heightmap."""

    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution

    def compute(self, heightmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute slope and aspect for each cell of the heightmap.

        Args:
        - heightmap: 2D numpy array of heights

        Returns:
        - slope: 2D numpy array of slopes in degrees
        - aspect: 2D numpy array of aspects in degrees (compass direction the slope faces)
        """
        rows, cols = heightmap.shape
        slope = np.zeros((rows, cols), dtype=np.float32)
        aspect = np.zeros((rows, cols), dtype=np.float32)
        for y in range(rows):
            for x in range(cols):
                xm, xp = max(x - 1, 0), min(x + 1, cols - 1)
                ym, yp = max(y - 1, 0), min(y + 1, rows - 1)
                dzdx = (heightmap[y, xp] - heightmap[y, xm]) / (
                    2 * self.resolution
                )
                dzdy = (heightmap[yp, x] - heightmap[ym, x]) / (
                    2 * self.resolution
                )
                # slope is the steepest descent angle
                slope[y, x] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
                # aspect: compass direction the slope faces
                aspect[y, x] = (np.degrees(np.arctan2(dzdy, -dzdx)) + 360) % 360
        return slope, aspect
