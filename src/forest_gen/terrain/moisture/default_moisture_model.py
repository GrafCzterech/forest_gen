from .moisture_model import MoistureModel
from typing import Dict
import numpy as np


class DefaultMoistureModel(MoistureModel):
    """
    Default moisture model combining normalized flow, slope penalty, and aspect factor.
    """

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = (
            weights
            if weights is not None
            else {"flow": 0.5, "slope": 0.3, "aspect": 0.2}
        )

    def compute(
        self, flow: np.ndarray, slope: np.ndarray, aspect: np.ndarray
    ) -> np.ndarray:
        flow_norm = (flow - flow.min()) / (np.ptp(flow) + 1e-8)
        slope_penalty = 1.0 - (slope / 90.0)
        aspect_rad = np.deg2rad(aspect)
        aspect_factor = (np.cos(aspect_rad) + 1) / 2
        w_flow = self.weights.get("flow", 0.0)
        w_slope = self.weights.get("slope", 0.0)
        w_aspect = self.weights.get("aspect", 0.0)
        moisture = (
            w_flow * flow_norm
            + w_slope * slope_penalty
            + w_aspect * aspect_factor
        )
        moisture -= moisture.min()
        moisture /= moisture.max() + 1e-8
        return moisture
