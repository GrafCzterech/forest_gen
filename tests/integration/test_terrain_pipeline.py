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


def _assert_raster_ok(arr: np.ndarray, shape: tuple[int, int], *, name: str):
    assert isinstance(arr, np.ndarray), f"{name} must be numpy array"
    assert arr.shape == shape, f"{name} shape {arr.shape} != {shape}"
    assert np.isfinite(arr).all(), f"{name} contains non-finite values"


# ------------- (1) end-to-end smoke ------------- #

@pytest.mark.integration
@pytest.mark.parametrize("apply_microrelief", [False, True])
def test_terrain_pipeline_smoke_fractal(sym, apply_microrelief):
    """
    End-to-end integration smoke:
    TerrainBuilder -> TerrainGenerator.generate -> Terrain.
    Asserts shapes, finiteness, and basic ranges.
    """
    pytest.importorskip("scipy")  # FractalNoise + carving/filtering need it.

    TerrainBuilder = _resolve(sym, "TerrainBuilder", "terrain.terrain_builder", "terrain")
    TerrainConfig = _resolve(sym, "TerrainConfig", "terrain.terrain_config", "terrain")
    Terrain = _resolve(sym, "Terrain", "terrain.terrain", "terrain")

    builder = TerrainBuilder().with_noise("fractal").with_microrelief(apply_microrelief).with_moisture_model()
    gen = builder.build()

    cfg = TerrainConfig(
        size=20,
        resolution=1.0,
        scale=30.0,
        octaves=2,
        height_scale=1.0,
        apply_microrelief=apply_microrelief,
    )

    terrain = gen.generate(cfg)
    assert isinstance(terrain, Terrain)

    shape = (cfg.rows, cfg.cols)

    _assert_raster_ok(terrain.heightmap, shape, name="heightmap")
    _assert_raster_ok(terrain.flow, shape, name="flow")
    _assert_raster_ok(terrain.slope, shape, name="slope")
    _assert_raster_ok(terrain.aspect, shape, name="aspect")
    _assert_raster_ok(terrain.moisture, shape, name="moisture")

    # Heightmap should have variation (not constant).
    assert float(np.ptp(terrain.heightmap)) > 0.0

    # Moisture model normalizes to [0,1]
    assert float(terrain.moisture.min()) >= -1e-6
    assert float(terrain.moisture.max()) <= 1.0 + 1e-6

    # Slope is in degrees by contract; should be within [0, 90] for this estimator.
    assert float(terrain.slope.min()) >= -1e-6
    assert float(terrain.slope.max()) <= 90.0 + 1e-3

    # Aspect is in [0,360)
    assert float(terrain.aspect.min()) >= -1e-6
    assert float(terrain.aspect.max()) < 360.0 + 1e-6


# ------------- (2) determinism with seeds ------------- #

@pytest.mark.integration
def test_terrain_pipeline_deterministic_with_seeds(sym):
    """
    Determinism integration:
    Using explicit seeds (SimplexNoise seed + numpy seed for microrelief) must reproduce
    identical terrain rasters (hm/flow/slope/aspect/moisture).
    """
    pytest.importorskip("opensimplex")
    pytest.importorskip("scipy")

    TerrainGenerator = _resolve(sym, "TerrainGenerator", "terrain.terrain_generator", "terrain")
    TerrainConfig = _resolve(sym, "TerrainConfig", "terrain.terrain_config", "terrain")

    SimplexNoise = _resolve(sym, "SimplexNoise", "terrain.noise.simplex_noise", "terrain.noise", "terrain")
    BasicMicrorelief = _resolve(sym, "BasicMicrorelief", "terrain.microrelief.basic_microrelief", "terrain.microrelief", "terrain")
    DefaultMoistureModel = _resolve(sym, "DefaultMoistureModel", "terrain.moisture.default_moisture_model", "terrain.moisture", "terrain")

    noise_seed = 123
    np_seed = 123

    gen = TerrainGenerator(
        noise=SimplexNoise(seed=noise_seed),
        micro=BasicMicrorelief(strength=0.01, sigma=0.8),
        moisture_model=DefaultMoistureModel(),
    )

    cfg = TerrainConfig(
        size=15,
        resolution=1.0,
        scale=25.0,
        octaves=2,
        height_scale=1.0,
        apply_microrelief=True,
    )

    np.random.seed(np_seed)
    t1 = gen.generate(cfg)

    np.random.seed(np_seed)
    t2 = gen.generate(cfg)

    # Exact reproducibility (same seeds -> same arrays).
    assert np.array_equal(t1.heightmap, t2.heightmap)
    assert np.array_equal(t1.flow, t2.flow)
    assert np.array_equal(t1.slope, t2.slope)
    assert np.array_equal(t1.aspect, t2.aspect)
    assert np.array_equal(t1.moisture, t2.moisture)

    # Non-vacuous: changing numpy seed (microrelief) should change the result.
    np.random.seed(np_seed + 1)
    t3 = gen.generate(cfg)

    assert not np.array_equal(t1.heightmap, t3.heightmap)
