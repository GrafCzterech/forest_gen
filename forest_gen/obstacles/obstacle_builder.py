from __future__ import annotations

import random
from collections.abc import Iterable

from .obstacle_config import ObstacleSpec
from .obstacle_generator import ObstacleGenerator


class ObstacleBuilder:
    """Fluent builder for configuring obstacle generation."""

    def __init__(self):
        self._specs: list[ObstacleSpec] | None = None
        self._seed: int | None = None

    def add_spec(self, spec: ObstacleSpec) -> "ObstacleBuilder":
        self._specs = (self._specs or []) + [spec]
        return self

    def with_specs(self, specs: Iterable[ObstacleSpec]) -> "ObstacleBuilder":
        self._specs = list(specs)
        return self

    def with_seed(self, seed: int | None) -> "ObstacleBuilder":
        self._seed = seed
        return self

    def build(self) -> ObstacleGenerator:
        rng = random.Random(self._seed)
        specs = tuple(self._specs) if self._specs is not None else None
        return ObstacleGenerator(specs, rng)