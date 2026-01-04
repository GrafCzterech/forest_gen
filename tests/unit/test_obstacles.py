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
    # Stable comparison: floats rounded, order preserved (generation is deterministic with seed)
    return tuple(
        (o.kind, round(o.coords[0], 10), round(o.coords[1], 10), round(o.radius, 10))
        for o in obstacles
    )


def _pairwise_spacing_ok(obstacles, min_distance: float) -> bool:
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            a, b = obstacles[i], obstacles[j]
            required = max(min_distance, a.radius + b.radius)
            if math.dist(a.coords, b.coords) < required - 1e-12:
                return False
    return True


@pytest.mark.unit
def test_default_obstacle_specs_have_expected_content(sym):
    ObstacleSpec = _resolve(
        sym,
        "ObstacleSpec",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )
    default_obstacle_specs = _resolve(
        sym,
        "default_obstacle_specs",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    specs = default_obstacle_specs()
    assert isinstance(specs, tuple)
    assert len(specs) >= 1
    assert all(isinstance(s, ObstacleSpec) for s in specs)

    names = {s.name for s in specs}
    # Your defaults (from snippets): fallen_tree, rock, log
    assert {"fallen_tree", "rock", "log"}.issubset(names)

    for s in specs:
        assert s.radius > 0.0
        assert s.weight > 0.0


@pytest.mark.unit
def test_obstacle_config_area_and_expected_count(sym):
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    cfg = ObstacleConfig(size=(10.0, 20.0), density=0.01)
    assert cfg.area == pytest.approx(200.0)
    # expected = max(1, round(area*density)) => round(2.0)=2
    assert cfg.expected_obstacle_count() == 2

    cfg0 = ObstacleConfig(size=(10.0, 20.0), density=0.0)
    # Behavior is explicit in code: at least 1
    assert cfg0.expected_obstacle_count() == 1


@pytest.mark.unit
def test_obstacle_config_with_specs_returns_copy(sym):
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )
    ObstacleSpec = _resolve(
        sym,
        "ObstacleSpec",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    cfg = ObstacleConfig(size=(10.0, 10.0), density=0.01, min_distance=2.0, seed=7)
    custom = [ObstacleSpec("a", radius=1.0, weight=1.0), ObstacleSpec("b", radius=2.0, weight=2.0)]

    cfg2 = cfg.with_specs(custom)

    assert cfg2 is not cfg
    assert cfg2.size == cfg.size
    assert cfg2.density == cfg.density
    assert cfg2.min_distance == cfg.min_distance
    assert cfg2.seed == cfg.seed
    assert cfg2.specs == tuple(custom)


@pytest.mark.unit
def test_obstacle_generator_deterministic_with_config_seed(sym):
    ObstacleGenerator = _resolve(
        sym,
        "ObstacleGenerator",
        "obstacles.obstacle_generator",
        "obstacle_generator",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    gen = ObstacleGenerator()
    cfg = ObstacleConfig(size=(50.0, 50.0), density=0.004, min_distance=2.0, seed=123)

    a = gen.generate(cfg)
    b = gen.generate(cfg)

    assert _snapshot(a) == _snapshot(b)


@pytest.mark.unit
def test_obstacle_generator_config_seed_overrides_generator_rng(sym):
    ObstacleGenerator = _resolve(
        sym,
        "ObstacleGenerator",
        "obstacles.obstacle_generator",
        "obstacle_generator",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    cfg = ObstacleConfig(size=(50.0, 50.0), density=0.004, min_distance=2.0, seed=999)

    g1 = ObstacleGenerator()
    g2 = ObstacleGenerator()

    # Even if internal RNG differs, config.seed forces identical output.
    assert _snapshot(g1.generate(cfg)) == _snapshot(g2.generate(cfg))


@pytest.mark.unit
def test_obstacle_generator_places_within_bounds_and_respects_spacing(sym):
    ObstacleGenerator = _resolve(
        sym,
        "ObstacleGenerator",
        "obstacles.obstacle_generator",
        "obstacle_generator",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    cfg = ObstacleConfig(size=(60.0, 40.0), density=0.01, min_distance=2.0, seed=42)
    obstacles = ObstacleGenerator().generate(cfg)

    assert len(obstacles) >= 1

    for o in obstacles:
        assert 0.0 <= o.coords[0] <= cfg.size[0]
        assert 0.0 <= o.coords[1] <= cfg.size[1]
        assert o.radius > 0.0

    assert _pairwise_spacing_ok(obstacles, cfg.min_distance)


@pytest.mark.unit
def test_obstacle_generator_respects_spec_priority_config_over_generator_over_default(sym):
    ObstacleGenerator = _resolve(
        sym,
        "ObstacleGenerator",
        "obstacles.obstacle_generator",
        "obstacle_generator",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )
    ObstacleSpec = _resolve(
        sym,
        "ObstacleSpec",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    spec_gen = (ObstacleSpec("gen_only", radius=1.0, weight=1.0),)
    spec_cfg = (ObstacleSpec("cfg_only", radius=1.0, weight=1.0),)

    gen = ObstacleGenerator(specs=spec_gen)
    cfg = ObstacleConfig(size=(30.0, 30.0), density=0.01, min_distance=1.0, seed=7, specs=spec_cfg)

    obs = gen.generate(cfg)
    assert {o.kind for o in obs} == {"cfg_only"}

    cfg2 = ObstacleConfig(size=(30.0, 30.0), density=0.01, min_distance=1.0, seed=7, specs=None)
    obs2 = gen.generate(cfg2)
    assert {o.kind for o in obs2} == {"gen_only"}


@pytest.mark.unit
def test_obstacle_generator_sampling_weights_bias_output(sym):
    ObstacleGenerator = _resolve(
        sym,
        "ObstacleGenerator",
        "obstacles.obstacle_generator",
        "obstacle_generator",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )
    ObstacleSpec = _resolve(
        sym,
        "ObstacleSpec",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    # Same radius to avoid spacing-driven bias; low min_distance so rejection is rare.
    specs = (
        ObstacleSpec("A", radius=0.2, weight=0.95),
        ObstacleSpec("B", radius=0.2, weight=0.05),
    )

    cfg = ObstacleConfig(
        size=(100.0, 100.0),
        density=0.03,        # ~300 targets
        min_distance=0.0,
        seed=123,
        specs=specs,
    )

    obs = ObstacleGenerator().generate(cfg)
    counts = {"A": 0, "B": 0}
    for o in obs:
        counts[o.kind] += 1

    # Strong bias should show up clearly with this many samples.
    assert counts["A"] > counts["B"] * 5
    assert counts["A"] / max(1, (counts["A"] + counts["B"])) > 0.8


@pytest.mark.unit
def test_obstacle_builder_seed_makes_generator_deterministic_when_config_seed_none(sym):
    ObstacleBuilder = _resolve(
        sym,
        "ObstacleBuilder",
        "obstacles.obstacle_builder",
        "obstacle_builder",
        "obstacles",
    )
    ObstacleConfig = _resolve(
        sym,
        "ObstacleConfig",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )
    ObstacleSpec = _resolve(
        sym,
        "ObstacleSpec",
        "obstacles.obstacle_config",
        "obstacle_config",
        "obstacles",
    )

    specs = [ObstacleSpec("X", radius=1.0, weight=1.0)]
    cfg = ObstacleConfig(size=(40.0, 40.0), density=0.01, min_distance=0.0, seed=None, specs=None)

    g1 = ObstacleBuilder().with_specs(specs).with_seed(123).build()
    g2 = ObstacleBuilder().with_specs(specs).with_seed(123).build()

    # config.seed is None -> generators use their internal RNG, so builder seed matters.
    assert _snapshot(g1.generate(cfg)) == _snapshot(g2.generate(cfg))


@pytest.mark.unit
def test_obstacle_generator_rng_is_not_shared_between_instances(sym):
    ObstacleGenerator = _resolve(
        sym,
        "ObstacleGenerator",
        "obstacles.obstacle_generator",
        "obstacle_generator",
        "obstacles",
    )

    a = ObstacleGenerator()
    b = ObstacleGenerator()

    # Ideally these should be different RNG instances.
    assert a.rng is not b.rng
