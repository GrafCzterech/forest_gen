from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

from .obstacle import Obstacle
from .obstacle_config import ObstacleConfig, ObstacleSpec, default_obstacle_specs


@dataclass
class ObstacleGenerator:
    """
    Generate random navigational obstacles within a bounded area.

    Obstacles are sampled according to obstacle specifications and
    placed with minimum distance constraints.
    """

    specs: tuple[ObstacleSpec, ...] | None = None
    """Optional obstacle specifications overriding defaults."""
    rng: random.Random = field(default_factory=random.Random)
    """Base random number generator."""

    def _resolve_specs(self, config: ObstacleConfig) -> tuple[ObstacleSpec, ...]:
        if config.specs:
            return config.specs
        if self.specs:
            return self.specs
        return default_obstacle_specs()

    def _ensure_rng(self, seed: int | None) -> random.Random:
        if seed is None:
            return self.rng
        return random.Random(seed)

    def _is_far_enough(
        self,
        candidate: tuple[float, float],
        radius: float,
        obstacles: Iterable[Obstacle],
        min_distance: float,
    ) -> bool:
        for obstacle in obstacles:
            required_gap = max(min_distance, obstacle.radius + radius)
            if math.dist(candidate, obstacle.coords) < required_gap:
                return False
        return True

    def generate(self, config: ObstacleConfig) -> list[Obstacle]:
        """
        Generate obstacles according to the given configuration.

        :param config: Obstacle generation configuration.
        :type config: ObstacleConfig
        :return: Generated obstacles.
        :rtype: list[Obstacle]
        """
        ...
        rng = self._ensure_rng(config.seed)
        specs = self._resolve_specs(config)
        weights = [spec.weight for spec in specs]
        target_count = config.expected_obstacle_count()

        obstacles: list[Obstacle] = []
        max_attempts = max(target_count * 20, target_count + 10)

        attempts = 0
        while len(obstacles) < target_count and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(0, config.size[0])
            y = rng.uniform(0, config.size[1])
            spec = rng.choices(specs, weights=weights, k=1)[0]

            if self._is_far_enough((x, y), spec.radius, obstacles, config.min_distance):
                obstacles.append(Obstacle(spec.name, (x, y), spec.radius))

        return obstacles
