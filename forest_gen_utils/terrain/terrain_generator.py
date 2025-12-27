from __future__ import annotations

from dataclasses import dataclass

from .microrelief import MicroreliefStrategy
from .moisture import MoistureModel
from .noise import NoiseStrategy
from .terrain import Terrain
from .terrain_config import TerrainConfig
from .utils import DrainageCarver, FlowAccumulator, SlopeAspectCalculator


@dataclass
class TerrainGenerator:
    """
    High-level terrain generation orchestrator.

    Coordinates noise generation, optional microrelief, hydrology,
    slope/aspect analysis, and moisture computation to produce
    a complete :class:`Terrain`.
    """

    noise: NoiseStrategy
    micro: MicroreliefStrategy
    moisture_model: MoistureModel

    def generate(self, config: TerrainConfig) -> Terrain:
        """
        Generate a terrain from the given configuration.

        :param config: Terrain generation configuration.
        :type config: TerrainConfig
        :return: Generated terrain data container.
        :rtype: Terrain
        """
        hm = self.noise.generate(config)
        hm = self.micro.apply(hm) if config.apply_microrelief else hm
        flow = FlowAccumulator().compute(hm)
        hm = DrainageCarver().apply(hm, flow)
        hm *= config.height_scale
        slope, aspect = SlopeAspectCalculator(config.resolution).compute(hm)
        moisture = self.moisture_model.compute(flow, slope, aspect)
        return Terrain(config, hm, flow, slope, aspect, moisture)
