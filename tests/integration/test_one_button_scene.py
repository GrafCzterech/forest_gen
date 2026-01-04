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


@pytest.mark.integration
def test_one_button_scene_pipeline_traversability_penalized_by_obstacles(sym):
    # This test uses: PoissonDisk, RegularGridInterpolator, gaussian_filter
    pytest.importorskip("scipy")
    pytest.importorskip("trimesh")
    pytest.importorskip("opensimplex")

    # -------- Resolve symbols (with multiple possible module layouts) --------
    TerrainConfig = _resolve(sym, "TerrainConfig", "terrain.terrain_config", "terrain_config", "terrain")
    TerrainGenerator = _resolve(sym, "TerrainGenerator", "terrain.terrain_generator", "terrain_generator", "terrain")
    SimplexNoise = _resolve(sym, "SimplexNoise", "terrain.noise.simplex_noise", "noise.simplex_noise", "noise", "terrain")
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

    # -------- 1) Terrain --------
    size = 30  # keep runtime small but meaningful
    cfg = TerrainConfig(
        size=size,
        resolution=1.0,
        scale=25.0,
        octaves=2,
        height_scale=1.0,
        apply_microrelief=False,
    )

    terrain_gen = TerrainGenerator(
        noise=SimplexNoise(seed=123),
        micro=NoneMicrorelief(),
        moisture_model=DefaultMoistureModel(),
    )
    terrain = terrain_gen.generate(cfg)

    assert terrain.config.size == size
    assert terrain.heightmap.shape == (cfg.rows, cfg.cols)
    assert np.isfinite(terrain.heightmap).all()

    # -------- 2) Forest canopy (baseline tree positions) --------
    canopy = Species(
        name="Canopy",
        max_age=50,
        radius=1.6,
        species_density=0.003,      # ~3 trees on 900 m2
        reproduction_rate=0,        # keep deterministic-ish; no growth loop effects
        reproduction_radius=3.0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )

    forest = (
        ForestBuilder()
        .with_size((size, size))
        .with_terrain(terrain)
        .add_species("canopy", canopy)
        .build()
    )

    canopy_state = forest.generate(ForestConfig(scene_density=1.0, years=0))
    tree_positions = [p.coords for p in canopy_state]

    assert len(tree_positions) >= 1
    for (x, y) in tree_positions:
        assert 0.0 <= x <= size
        assert 0.0 <= y <= size

    # -------- 3) Understory + 4) Grass (same forest pipeline, different viability maps) --------
    understory = UnderstoryDistributor(
        terrain,
        canopy_positions=tree_positions,
        species_density=0.01,       # ~9 plants
        reproduction_rate=0,
        reproduction_radius=3.0,
        radius=1.2,
        max_age=20,
    )
    understory_state = understory.generate(ForestConfig(scene_density=1.0, years=0))
    understory_plants = list(understory_state)
    for p in understory_plants:
        assert 0.0 <= p.coords[0] <= size
        assert 0.0 <= p.coords[1] <= size
        
    grass = GrassDistributor(
        terrain,
        tree_positions=tree_positions,
        species_density=0.02,       # ~18 plants
        reproduction_rate=0,
        reproduction_radius=2.0,
        radius=0.6,
        max_age=6,
    )
    grass_state = grass.generate(ForestConfig(scene_density=1.0, years=0))
    assert len(list(grass_state)) >= 1

    # All plants must be in-bounds (common coordinate system check)
    for st in (canopy_state, understory_state, grass_state):
        for p in st:
            assert 0.0 <= p.coords[0] <= size
            assert 0.0 <= p.coords[1] <= size

    # -------- 5) Obstacles --------
    obs_cfg = ObstacleConfig(
        size=(float(size), float(size)),
        density=0.003,      # expected ~3 (and never below 1)
        min_distance=1.0,
        seed=999,
    )
    obstacles = ObstacleBuilder().build().generate(obs_cfg)
    assert len(obstacles) >= 1

    obstacle_points = [o.coords for o in obstacles]
    for (x, y) in obstacle_points:
        assert 0.0 <= x <= size
        assert 0.0 <= y <= size

    # -------- 6) Traversability map: compare no-obstacles vs obstacles --------
    tb0 = TraversabilityMapBuilder(terrain, resolution_factor=2, max_slope_deg=30.0)
    score0 = tb0.get_score()
    assert np.isfinite(score0).all()
    assert 0.0 <= float(score0.min()) <= float(score0.max()) <= 1.0

    tb1 = TraversabilityMapBuilder(terrain, resolution_factor=2, max_slope_deg=30.0)
    tb1.add_obstacle_score(
        obstacles=obstacle_points,
        obstacle_influence_radius=7.0,
        obstacle_penalty=0.45,
    )
    score1 = tb1.get_score()
    assert np.isfinite(score1).all()
    assert 0.0 <= float(score1.min()) <= float(score1.max()) <= 1.0

    # Global effect: adding obstacles cannot increase average traversability.
    assert float(score1.mean()) <= float(score0.mean()) + 1e-12

    # Local effect: at (or nearest to) an obstacle location, score must strictly decrease.
    # Use tb1's grid for indexing.
    ox, oy = obstacle_points[0]
    x_axis = tb1.X[0, :]   # shape (N,)
    y_axis = tb1.Y[:, 0]   # shape (N,)
    j = _nearest_idx(x_axis, ox)
    i = _nearest_idx(y_axis, oy)

    assert score1[i, j] < score0[i, j]
