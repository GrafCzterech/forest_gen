import pytest
import numpy as np


def _assert_heightmap_valid(hm: np.ndarray, rows: int, cols: int):
    assert hm.shape == (rows, cols)
    assert np.isfinite(hm).all()
    assert hm.dtype == np.float32 or np.issubdtype(hm.dtype, np.floating)
    assert float(hm.min()) >= 0.0
    assert float(hm.max()) <= 1.0


@pytest.mark.unit
def test_simplex_noise_deterministic_and_normalized(sym):
    pytest.importorskip("opensimplex")

    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")
    SimplexNoise = sym("terrain.noise.simplex_noise", "SimplexNoise")

    cfg = TerrainConfig(size=32, resolution=1.0, scale=50.0)

    a = SimplexNoise(seed=123).generate(cfg)
    b = SimplexNoise(seed=123).generate(cfg)

    _assert_heightmap_valid(a, cfg.rows, cfg.cols)
    assert np.allclose(a, b)


@pytest.mark.unit
def test_simplex_noise_none_seed_matches_zero_seed(sym):
    pytest.importorskip("opensimplex")

    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")
    SimplexNoise = sym("terrain.noise.simplex_noise", "SimplexNoise")

    cfg = TerrainConfig(size=16, resolution=1.0, scale=25.0)

    a = SimplexNoise(seed=None).generate(cfg)
    b = SimplexNoise(seed=0).generate(cfg)

    assert np.allclose(a, b)


@pytest.mark.unit
def test_simplex_noise_scale_zero_raises(sym):
    pytest.importorskip("opensimplex")

    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")
    SimplexNoise = sym("terrain.noise.simplex_noise", "SimplexNoise")

    cfg = TerrainConfig(size=8, resolution=1.0, scale=0.0)
    with pytest.raises(ZeroDivisionError):
        SimplexNoise(seed=1).generate(cfg)


@pytest.mark.unit
def test_simplex_noise_size_zero_should_not_produce_nan(sym):
    pytest.importorskip("opensimplex")

    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")
    SimplexNoise = sym("terrain.noise.simplex_noise", "SimplexNoise")

    cfg = TerrainConfig(size=0.0, resolution=1.0, scale=10.0)
    hm = SimplexNoise(seed=1).generate(cfg)

    assert np.isfinite(hm).all()
    assert float(hm.min()) >= 0.0 and float(hm.max()) <= 1.0


@pytest.mark.unit
def test_fractal_noise_deterministic_and_normalized(sym):
    pytest.importorskip("scipy")

    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")
    FractalNoise = sym("terrain.noise.fractal_noise", "FractalNoise")

    cfg = TerrainConfig(size=32, resolution=1.0, scale=50.0, octaves=3)

    # Preserve  RNG state 
    state = np.random.get_state()
    try:
        a = FractalNoise(seed=123).generate(cfg)
        b = FractalNoise(seed=123).generate(cfg)
    finally:
        np.random.set_state(state)

    _assert_heightmap_valid(a, cfg.rows, cfg.cols)
    assert np.allclose(a, b)


@pytest.mark.unit
def test_fractal_noise_should_not_mutate_global_numpy_rng(sym):
    pytest.importorskip("scipy")

    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")
    FractalNoise = sym("terrain.noise.fractal_noise", "FractalNoise")

    cfg = TerrainConfig(size=16, resolution=1.0, scale=30.0, octaves=2)

    np.random.seed(999)
    before = np.random.random(5)

    _ = FractalNoise(seed=123).generate(cfg)

    after = np.random.random(5)

    #  should not reset/affect global RNG stream.
    np.random.seed(999)
    expected_after = np.random.random(10)[5:]

    assert np.allclose(after, expected_after)


@pytest.mark.unit
def test_fractal_noise_octaves_zero_should_fail_cleanly(sym):
    pytest.importorskip("scipy")

    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")
    FractalNoise = sym("terrain.noise.fractal_noise", "FractalNoise")

    cfg = TerrainConfig(size=16, resolution=1.0, scale=30.0, octaves=0)
    hm = FractalNoise(seed=1).generate(cfg)

    assert np.isfinite(hm).all()
    assert float(hm.min()) >= 0.0 and float(hm.max()) <= 1.0
