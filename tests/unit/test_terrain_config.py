import pytest
import numpy as np


@pytest.mark.unit
@pytest.mark.parametrize(
    "size,resolution",
    [
        (100.0, 1.0),
        (100.0, 3.0),
        (7.5, 0.2),
        (0.0, 1.0),
    ],
)
def test_terrain_config_transform_bounds_and_extent(sym, size, resolution):
    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")

    cfg = TerrainConfig(size=size, resolution=resolution)

    assert cfg.cols == cfg.rows
    assert cfg.transform(0.0) == 0
    assert cfg.transform(cfg.size) == cfg.rows - 1

    xs = np.linspace(0.0, float(cfg.size), 50)
    idx = np.array([cfg.transform(float(x)) for x in xs], dtype=int)
    assert idx.min() >= 0
    assert idx.max() <= cfg.rows - 1

    grid_extent = (cfg.rows - 1) * cfg.resolution
    assert abs(grid_extent - cfg.size) <= (cfg.resolution * 0.5 + 1e-9)


@pytest.mark.unit
def test_terrain_config_moisture_weights_is_not_shared(sym):
    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")

    a = TerrainConfig(size=10)
    b = TerrainConfig(size=10)

    assert a.moisture_weights == {"flow": 0.5, "slope": 0.3, "aspect": 0.2}
    assert a.moisture_weights is not b.moisture_weights

    a.moisture_weights["flow"] = 0.123
    assert b.moisture_weights["flow"] == 0.5


@pytest.mark.unit
def test_terrain_config_resolution_zero_is_invalid(sym):
    TerrainConfig = sym("terrain.terrain_config", "TerrainConfig")

    cfg = TerrainConfig(size=10, resolution=0.0)

    #transform will raise ZeroDivisionError.
    with pytest.raises(ZeroDivisionError):
        _ = cfg.transform(1.0)
