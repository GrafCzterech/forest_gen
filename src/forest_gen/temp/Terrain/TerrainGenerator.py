from __future__ import annotations

from ..Utils.DrainageCarver import DrainageCarver
from ..Utils.FlowAccumulator import FlowAccumulator
from ..Utils.SlopeAspectCalculator import SlopeAspectCalculator

from ..Export.ExportStrategy import ExportStrategy
from ..Microrelief.MicroreliefStrategy import MicroreliefStrategy
from ..Moisture.MoistureModel import MoistureModel
from ..Noise.NoiseStrategy import NoiseStrategy

from .TerrainConfig import TerrainConfig

from ..Visualization import (
    HeightmapVisualizer,
    FlowVisualizer,
    MoistureVisualizer,
    Visualizer,
)

import numpy as np


class TerrainGenerator:
    """
    Orchestrates terrain creation using noise, microrelief,
    hydrology, and moisture models, then exports via an ExportStrategy.
    """

    def __init__(
        self,
        noise: NoiseStrategy,
        micro: MicroreliefStrategy,
        moisture: MoistureModel,
        exporter: ExportStrategy,
        heightmap_viz: Visualizer | None = None,
        flow_viz: Visualizer | None = None,
        moist_viz: Visualizer | None = None,
    ):
        self.noise = noise
        self.micro = micro
        self.moisture_model = moisture
        self.exporter = exporter

        self.heightmap: np.ndarray | None = None
        self.flow: np.ndarray | None = None
        self.slope: np.ndarray | None = None
        self.aspect: np.ndarray | None = None
        self.moisture: np.ndarray | None = None
        self.resolution: float | None = None

        self._viz_height = heightmap_viz or HeightmapVisualizer()
        self._viz_flow = flow_viz or FlowVisualizer()
        self._viz_moist = moist_viz or MoistureVisualizer()

    def generate(self, config: TerrainConfig) -> None:
        self.resolution = config.resolution
        hm = self.noise.generate(config)
        hm = self.micro.apply(hm) if config.apply_microrelief else hm
        self.flow = FlowAccumulator().compute(hm)
        hm = DrainageCarver().apply(hm, self.flow)
        self.slope, self.aspect = SlopeAspectCalculator(
            config.resolution
        ).compute(hm)
        self.moisture = self.moisture_model.compute(
            self.flow, self.slope, self.aspect
        )
        self.heightmap = hm

    def export(self, path: str) -> None:
        """Export the generated heightmap using the configured ExportStrategy."""
        if self.heightmap is None:
            raise RuntimeError("Terrain not generated: call generate() first.")
        self.exporter.export(self.heightmap, path)

    def show_flow(self) -> None:
        if self.flow is None:
            raise RuntimeError(
                "Flow accumulation not computed. Call generate() first."
            )
        self._viz_flow.visualize(self.flow, "Flow Accumulation")

    def show_moisture(self) -> None:
        if self.moisture is None:
            raise RuntimeError("Moisture not computed. Call generate() first.")
        self._viz_moist.visualize(self.moisture, "Moisture Index")

    def show_heightmap(self) -> None:
        if self.heightmap is None:
            raise RuntimeError(
                "Heightmap not generated. Call generate() first."
            )
        self._viz_height.visualize(self.heightmap, "Heightmap")

    def visualize_all(self):
        self.show_heightmap()
        self.show_flow()
        self.show_moisture()
