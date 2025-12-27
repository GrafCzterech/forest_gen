from typing import Literal

from .microrelief import BasicMicrorelief, MicroreliefStrategy, NoneMicrorelief
from .moisture import DefaultMoistureModel, MoistureModel
from .noise import FractalNoise, NoiseFactory, NoiseStrategy
from .terrain_generator import TerrainGenerator


class TerrainBuilder:
    """
    Fluent builder for constructing a :class:`TerrainGenerator`.

    Provides a chainable API for selecting noise, microrelief,
    and moisture strategies before final assembly.
    """

    def __init__(self):
        self._noise: NoiseStrategy | None = None
        self._micro: MicroreliefStrategy | None = None
        self._moisture: MoistureModel | None = None

    def with_noise(self, name: Literal["fractal", "simplex"]) -> "TerrainBuilder":
        """
        Select the noise strategy.

        :param name: Noise strategy identifier.
        :type name: Literal["fractal", "simplex"]
        :return: Builder instance.
        :rtype: TerrainBuilder
        """
        self._noise = NoiseFactory.create(name)
        return self

    def with_microrelief(self, enable: bool) -> "TerrainBuilder":
        """
        Enable or disable microrelief.

        :param enable: Whether to apply microrelief.
        :type enable: bool
        :return: Builder instance.
        :rtype: TerrainBuilder
        """
        self._micro = BasicMicrorelief() if enable else NoneMicrorelief()
        return self

    def with_moisture_model(
        self, weights: dict[str, float] | None = None
    ) -> "TerrainBuilder":
        """
        Configure the moisture model.

        :param weights: Optional moisture weighting factors.
        :type weights: dict[str, float] or None
        :return: Builder instance.
        :rtype: TerrainBuilder
        """
        self._moisture = DefaultMoistureModel(weights)
        return self

    def build(self) -> TerrainGenerator:
        """
        Construct the configured :class:`TerrainGenerator`.

        Missing components are filled with default implementations.

        :return: Terrain generator instance.
        :rtype: TerrainGenerator
        """
        # Apply defaults if not set
        noise = self._noise or FractalNoise()
        micro = self._micro or NoneMicrorelief()
        moisture = self._moisture or DefaultMoistureModel()
        return TerrainGenerator(noise, micro, moisture)
