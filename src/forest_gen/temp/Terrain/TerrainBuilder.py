
from typing import Dict

from .TerrainGenerator import TerrainGenerator

from ..Export.PNGExporter import PNGExporter
from ..Export.ExportFactory import ExportFactory
from ..Export.ExportStrategy import ExportStrategy

from ..Microrelief import BasicMicrorelief, NoneMicrorelief
from ..Microrelief.MicroreliefStrategy import MicroreliefStrategy

from ..Moisture import DefaultMoistureModel
from ..Moisture.MoistureModel import MoistureModel

from ..Noise import FractalNoise, NoiseFactory
from ..Noise.NoiseStrategy import NoiseStrategy

from ..Visualization import Visualizer


class TerrainBuilder:
    """
    Fluent builder for configuring and constructing a TerrainGenerator.
    """
    def __init__(self):
        self._noise: NoiseStrategy | None = None
        self._micro: MicroreliefStrategy | None = None
        self._moisture: MoistureModel | None = None
        self._exporter: ExportStrategy | None = None

    def with_noise(self, name: str) -> 'TerrainBuilder':
        self._noise = NoiseFactory.create(name)
        return self

    def with_microrelief(self, enable: bool) -> 'TerrainBuilder':
        self._micro = BasicMicrorelief() if enable else NoneMicrorelief()
        return self

    def with_exporter(self, fmt: str, **kwargs) -> 'TerrainBuilder':
        self._exporter = ExportFactory.create(fmt, **kwargs)
        return self

    def with_moisture_model(self, weights: Dict[str, float] | None = None) -> 'TerrainBuilder':
        self._moisture = DefaultMoistureModel(weights)
        return self

    def with_visualizers(
        self,
        heightmap_viz: Visualizer,
        flow_viz: Visualizer,
        moist_viz: Visualizer,
    ) -> 'TerrainBuilder':
        self._heightmap_viz = heightmap_viz
        self._flow_viz = flow_viz
        self._moist_viz = moist_viz
        return self

    def build(self) -> TerrainGenerator:
        # Apply defaults if not set
        noise = self._noise or FractalNoise()
        micro = self._micro or NoneMicrorelief()
        moisture = self._moisture or DefaultMoistureModel()
        exporter = self._exporter or PNGExporter()
        return TerrainGenerator(noise, micro, moisture, exporter)

