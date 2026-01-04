import math
import pytest


def _resolve(sym, name: str, *paths: str):
    errs = []
    for p in paths:
        try:
            return sym(p, name)
        except Exception as e:
            errs.append(f"{p}.{name}: {type(e).__name__}({e})")
    raise RuntimeError(
        f"Could not resolve '{name}' from any of: {paths}\n" + "\n".join(errs)
    )


def _snapshot(obstacles):
    # Stable comparison for determinism assertions
    return tuple(
        (o.kind, round(o.coords[0], 10), round(o.coords[1], 10), round(o.radius, 10))
        for o in obstacles
    )


def _pairwise_spacing_ok(obstacles, min_distance: float) -> bool:
    obstacles = list(obstacles)
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            a, b = obstacles[i], obstacles[j]
            required = max(min_distance, a.radius + b.radius)
            if math.dist(a.coords, b.coords) < required - 1e-12:
                return False
    return True


@pytest.mark.integration
def test_obstacle_builder_pipeline_places_in_bounds_and_respects_spacing(sym):
    ObstacleBuilder = _resolve(
        sym,
        "ObstacleBuilder",
        "obstacles.obstacle_builder",
        "obstacle_builder",
        "asset_dist.obstacles.obstacle_builder",
        "asset_dist.obstacle_builder",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "asset_dist.obstacles.obstacle_config",
        "asset_dist.obstacle_config",
        "obstacles",
    )

    cfg = ObstacleConfig(
        size=(200.0, 200.0),
        density=0.0005,      # expected ~ 20 obstacles; stable & packable with defaults
        min_distance=2.0,
        seed=123,            # config seed makes this fully deterministic too
    )

    gen = ObstacleBuilder().build()
    obstacles = gen.generate(cfg)

    assert len(obstacles) >= 1
    assert len(obstacles) <= cfg.expected_obstacle_count()

    for o in obstacles:
        assert 0.0 <= o.coords[0] <= cfg.size[0]
        assert 0.0 <= o.coords[1] <= cfg.size[1]
        assert o.radius > 0.0

    assert _pairwise_spacing_ok(obstacles, cfg.min_distance)


@pytest.mark.integration
def test_obstacle_builder_seed_makes_deterministic_when_config_seed_none(sym):
    ObstacleBuilder = _resolve(
        sym,
        "ObstacleBuilder",
        "obstacles.obstacle_builder",
        "obstacle_builder",
        "asset_dist.obstacles.obstacle_builder",
        "asset_dist.obstacle_builder",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "asset_dist.obstacles.obstacle_config",
        "asset_dist.obstacle_config",
        "obstacles",
    )

    cfg = ObstacleConfig(
        size=(150.0, 150.0),
        density=0.0006,      # ~ 14
        min_distance=1.5,
        seed=None,           # IMPORTANT: uses generator RNG, so builder seed must drive determinism
    )

    g1 = ObstacleBuilder().with_seed(777).build()
    g2 = ObstacleBuilder().with_seed(777).build()

    a = g1.generate(cfg)
    b = g2.generate(cfg)

    assert _snapshot(a) == _snapshot(b)


@pytest.mark.integration
def test_config_seed_overrides_builder_seed(sym):
    ObstacleBuilder = _resolve(
        sym,
        "ObstacleBuilder",
        "obstacles.obstacle_builder",
        "obstacle_builder",
        "asset_dist.obstacles.obstacle_builder",
        "asset_dist.obstacle_builder",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "asset_dist.obstacles.obstacle_config",
        "asset_dist.obstacle_config",
        "obstacles",
    )

    cfg = ObstacleConfig(
        size=(150.0, 150.0),
        density=0.0006,
        min_distance=1.5,
        seed=999,            # IMPORTANT: config seed should override generator RNG
    )

    g1 = ObstacleBuilder().with_seed(1).build()
    g2 = ObstacleBuilder().with_seed(2).build()

    a = g1.generate(cfg)
    b = g2.generate(cfg)

    assert _snapshot(a) == _snapshot(b)
