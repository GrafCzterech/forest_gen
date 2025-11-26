from typing import Literal

from .terrain_generator import TerrainGenerator
from .microrelief import BasicMicrorelief, NoneMicrorelief, MicroreliefStrategy
from .moisture import DefaultMoistureModel, MoistureModel
from .noise import FractalNoise, NoiseFactory, NoiseStrategy


class TerrainBuilder:
    """
    Fluent builder for configuring and constructing a TerrainGenerator.
    """

    def __init__(self):
        self._noise: NoiseStrategy | None = None
        self._micro: MicroreliefStrategy | None = None
        self._moisture: MoistureModel | None = None

    def with_noise(
        self, name: Literal["fractal", "simplex"]
    ) -> "TerrainBuilder":
        self._noise = NoiseFactory.create(name)
        return self

    def with_microrelief(self, enable: bool) -> "TerrainBuilder":
        self._micro = BasicMicrorelief() if enable else NoneMicrorelief()
        return self

    def with_moisture_model(
        self, weights: dict[str, float] | None = None
    ) -> "TerrainBuilder":
        self._moisture = DefaultMoistureModel(weights)
        return self

    def build(self) -> TerrainGenerator:
        # Apply defaults if not set
        noise = self._noise or FractalNoise()
        micro = self._micro or NoneMicrorelief()
        moisture = self._moisture or DefaultMoistureModel()
        return TerrainGenerator(noise, micro, moisture)
