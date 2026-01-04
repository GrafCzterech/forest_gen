import os
import time
import random
import pytest
import numpy as np


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


def _nearest_idx(vec: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(vec - value)))


@pytest.mark.load
def test_one_button_world_load_profile(sym):
    """
    Load test: end-to-end "world generation" profile.

    Pipeline:
      Terrain -> Forest canopy -> Understory -> Grass -> Obstacles -> Traversability (+ obstacles)

    Optional perf gate:
      LOAD_ONE_BUTTON_BUDGET_SEC (float seconds)
    """
    pytest.importorskip("scipy")
    pytest.importorskip("trimesh")
    pytest.importorskip("opensimplex")

    # ---- knobs (tunable via env vars) ----
    size = int(os.getenv("LOAD_WORLD_SIZE", "128"))  # meters
    resolution_factor = int(os.getenv("LOAD_TRAV_RES_FACTOR", "2"))
    canopy_density = float(os.getenv("LOAD_CANOPY_DENSITY", "0.0025"))
    understory_density = float(os.getenv("LOAD_UNDERSTORY_DENSITY", "0.01"))
    grass_density = float(os.getenv("LOAD_GRASS_DENSITY", "0.02"))

    obstacle_density = float(os.getenv("LOAD_OBS_DENSITY", "0.005"))  
    obstacle_influence_radius = float(os.getenv("LOAD_OBS_INFLUENCE", "7.0"))
    obstacle_penalty = float(os.getenv("LOAD_OBS_PENALTY", "0.45"))

    budget_env = os.getenv("LOAD_ONE_BUTTON_BUDGET_SEC")
    budget = float(budget_env) if budget_env is not None else None

    random.seed(123)
    np.random.seed(123)

    TerrainConfig = _resolve(sym, "TerrainConfig", "terrain.terrain_config", "terrain_config", "terrain")
    TerrainGenerator = _resolve(sym, "TerrainGenerator", "terrain.terrain_generator", "terrain_generator", "terrain")

    # Prefer FractalNoise for heavier workload
    try:
        NoiseCls = _resolve(sym, "FractalNoise", "terrain.noise.fractal_noise", "noise.fractal_noise", "noise", "terrain")
        noise_kwargs = {"seed": 123}
    except Exception:
        NoiseCls = _resolve(sym, "SimplexNoise", "terrain.noise.simplex_noise", "noise.simplex_noise", "noise", "terrain")
        noise_kwargs = {"seed": 123}

    NoneMicrorelief = _resolve(sym, "NoneMicrorelief", "terrain.microrelief.none_microrelief", "microrelief.none_microrelief", "microrelief", "terrain")
    DefaultMoistureModel = _resolve(sym, "DefaultMoistureModel", "terrain.moisture.default_moisture_model", "moisture.default_moisture_model", "moisture", "terrain")

    ForestBuilder = _resolve(sym, "ForestBuilder", "forest.forest_builder", "forest_builder", "forest")
    ForestConfig = _resolve(sym, "ForestConfig", "forest.forest_config", "forest_config", "forest")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    GrassDistributor = _resolve(sym, "GrassDistributor", "asset_dist.grass", "asset_dist.grass_distributor", "grass")
    UnderstoryDistributor = _resolve(sym, "UnderstoryDistributor", "asset_dist.understory", "asset_dist.understory_distributor", "understory")

    ObstacleBuilder = _resolve(sym, "ObstacleBuilder", "obstacles.obstacle_builder", "obstacle_builder", "obstacles")
    ObstacleConfig = _resolve(sym, "ObstacleConfig", "obstacles.obstacle_config", "obstacle_config", "obstacles")

    TraversabilityMapBuilder = _resolve(
        sym,
        "TraversabilityMapBuilder",
        "traversability.traversability_map",
        "traversability.traversability",
        "navigation.traversability",
        "traversability",
    )

    # ---- run pipeline ----
    t0 = time.perf_counter()

    cfg = TerrainConfig(
        size=size,
        resolution=1.0,
        scale=float(os.getenv("LOAD_TERRAIN_SCALE", "35.0")),
        octaves=int(os.getenv("LOAD_TERRAIN_OCTAVES", "4")),
        height_scale=1.0,
        apply_microrelief=False,
    )

    terrain_gen = TerrainGenerator(
        noise=NoiseCls(**noise_kwargs),
        micro=NoneMicrorelief(),
        moisture_model=DefaultMoistureModel(),
    )
    terrain = terrain_gen.generate(cfg)

    # Terrain sanity
    assert np.isfinite(terrain.heightmap).all()
    assert terrain.heightmap.shape == (cfg.rows, cfg.cols)

    # Canopy (years=0 for stability; reproduction disabled anyway)
    canopy = Species(
        name="Canopy",
        max_age=60,
        radius=float(os.getenv("LOAD_CANOPY_RADIUS", "1.8")),
        species_density=canopy_density,
        reproduction_rate=0,
        reproduction_radius=4.0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    forest = (
        ForestBuilder()
        .with_size((float(size), float(size)))
        .with_terrain(terrain)
        .add_species("canopy", canopy)
        .build()
    )
    canopy_state = forest.generate(ForestConfig(scene_density=1.0, years=0))
    tree_positions = [p.coords for p in canopy_state]
    assert len(tree_positions) >= 1

    understory = UnderstoryDistributor(
        terrain,
        canopy_positions=tree_positions,
        species_density=understory_density,
        reproduction_rate=0,
        reproduction_radius=4.5,
        radius=float(os.getenv("LOAD_UNDERSTORY_RADIUS", "1.4")),
        max_age=35,
    )
    understory_state = understory.generate(ForestConfig(scene_density=1.0, years=0))
    understory_plants = list(understory_state)
    for p in understory_plants:
        assert 0.0 <= p.coords[0] <= size
        assert 0.0 <= p.coords[1] <= size

    grass = GrassDistributor(
        terrain,
        tree_positions=tree_positions,
        species_density=grass_density,
        reproduction_rate=0,
        reproduction_radius=2.5,
        radius=float(os.getenv("LOAD_GRASS_RADIUS", "0.6")),
        max_age=6,
    )
    grass_state = grass.generate(ForestConfig(scene_density=1.0, years=0))
    assert len(list(grass_state)) >= 1

    obs_cfg = ObstacleConfig(
        size=(float(size), float(size)),
        density=obstacle_density,
        min_distance=float(os.getenv("LOAD_OBS_MIN_DIST", "1.0")),
        seed=999,
    )
    obstacles = ObstacleBuilder().build().generate(obs_cfg)
    assert len(obstacles) >= 1
    obstacle_points = [o.coords for o in obstacles]

    tb0 = TraversabilityMapBuilder(terrain, resolution_factor=resolution_factor, max_slope_deg=30.0)
    score0 = tb0.get_score()

    tb1 = TraversabilityMapBuilder(terrain, resolution_factor=resolution_factor, max_slope_deg=30.0)
    tb1.add_obstacle_score(
        obstacles=obstacle_points,
        obstacle_influence_radius=obstacle_influence_radius,
        obstacle_penalty=obstacle_penalty,
    )
    score1 = tb1.get_score()

    for arr in (score0, score1):
        assert np.isfinite(arr).all()
        assert 0.0 <= float(arr.min()) <= float(arr.max()) <= 1.0

    # Obstacles must not increase average traversability
    assert float(score1.mean()) <= float(score0.mean()) + 1e-12

    ox, oy = obstacle_points[0]
    x_axis = tb1.X[0, :]
    y_axis = tb1.Y[:, 0]
    j = _nearest_idx(x_axis, ox)
    i = _nearest_idx(y_axis, oy)
    assert score1[i, j] < score0[i, j]

    for (x, y) in tree_positions:
        assert 0.0 <= x <= size and 0.0 <= y <= size
    for p in list(understory_state) + list(grass_state):
        assert 0.0 <= p.coords[0] <= size and 0.0 <= p.coords[1] <= size
    for (x, y) in obstacle_points:
        assert 0.0 <= x <= size and 0.0 <= y <= size

    dt = time.perf_counter() - t0

    if budget is not None:
        assert dt <= budget, f"one-button pipeline took {dt:.2f}s, budget={budget:.2f}s"

    assert dt > 0.0
