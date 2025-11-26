from __future__ import annotations
from dataclasses import dataclass

from .utils import DrainageCarver, FlowAccumulator, SlopeAspectCalculator

from .microrelief import MicroreliefStrategy
from .moisture import MoistureModel
from .noise import NoiseStrategy

from .terrain_config import TerrainConfig
from .terrain import Terrain


@dataclass
class TerrainGenerator:
    """
    Orchestrates terrain creation using noise, microrelief,
    hydrology, and moisture models, then exports via an ExportStrategy.
    """

    noise: NoiseStrategy
    micro: MicroreliefStrategy
    moisture_model: MoistureModel

    def generate(self, config: TerrainConfig) -> Terrain:
        hm = self.noise.generate(config)
        hm = self.micro.apply(hm) if config.apply_microrelief else hm
        flow = FlowAccumulator().compute(hm)
        hm = DrainageCarver().apply(hm, flow)
        slope, aspect = SlopeAspectCalculator(config.resolution).compute(hm)
        moisture = self.moisture_model.compute(flow, slope, aspect)
        return Terrain(config, hm, flow, slope, aspect, moisture)
