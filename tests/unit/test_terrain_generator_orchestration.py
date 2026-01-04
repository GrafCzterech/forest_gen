import os
import importlib
import pytest
import numpy as np


PKG = os.environ.get("FOREST_PKG", "forest_gen_utils")


def _mod(path: str):
    return importlib.import_module(f"{PKG}.{path}")


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


class _NoiseStub:
    def __init__(self, hm: np.ndarray):
        self.hm = hm
        self.called = False
        self.last_config = None

    def generate(self, config):
        self.called = True
        self.last_config = config
        return self.hm.copy()


class _MicroStub:
    def __init__(self, out: np.ndarray | None = None, *, forbid: bool = False):
        self.out = out
        self.forbid = forbid
        self.called = False

    def apply(self, heightmap: np.ndarray) -> np.ndarray:
        if self.forbid:
            raise AssertionError("microrelief.apply() should not be called")
        self.called = True
        return self.out.copy() if self.out is not None else heightmap


class _MoistureStub:
    def __init__(self, moisture: np.ndarray):
        self.moisture = moisture
        self.called = False
        self.args = None

    def compute(self, flow, slope, aspect):
        self.called = True
        self.args = (flow, slope, aspect)
        return self.moisture.copy()


@pytest.mark.unit
def test_terrain_generator_skips_microrelief_when_disabled(sym, monkeypatch):
    TerrainGenerator = _resolve(sym, "TerrainGenerator", "terrain.terrain_generator", "terrain")
    TerrainConfig = _resolve(sym, "TerrainConfig", "terrain.terrain_config", "terrain")
    Terrain = _resolve(sym, "Terrain", "terrain.terrain", "terrain")

    tg_mod = _mod("terrain.terrain_generator")

    # Stub for unit-level.
    flow = np.ones((3, 3), dtype=np.float32)
    slope = np.zeros((3, 3), dtype=np.float32)
    aspect = np.zeros((3, 3), dtype=np.float32)

    class _FA:
        def compute(self, hm):  # noqa: ARG002
            return flow

    class _DC:
        def apply(self, hm, f):  # noqa: ARG002
            return hm

    class _SAC:
        def __init__(self, resolution):  # noqa: ARG002
            pass

        def compute(self, hm):  # noqa: ARG002
            return slope, aspect

    monkeypatch.setattr(tg_mod, "FlowAccumulator", _FA)
    monkeypatch.setattr(tg_mod, "DrainageCarver", _DC)
    monkeypatch.setattr(tg_mod, "SlopeAspectCalculator", _SAC)

    base_hm = np.full((3, 3), 0.5, dtype=np.float32)
    noise = _NoiseStub(base_hm)
    micro = _MicroStub(forbid=True)
    moisture_stub = _MoistureStub(np.zeros((3, 3), dtype=np.float32))

    gen = TerrainGenerator(noise=noise, micro=micro, moisture_model=moisture_stub)

    cfg = TerrainConfig(size=2.0, resolution=1.0, scale=10.0, apply_microrelief=False)
    t = gen.generate(cfg)

    assert isinstance(t, Terrain)
    assert noise.called is True
    assert micro.called is False
    assert moisture_stub.called is True
    assert t.heightmap.shape == (cfg.rows, cfg.cols)


@pytest.mark.unit
def test_terrain_generator_applies_height_scale_and_wires_arrays(sym, monkeypatch):
    TerrainGenerator = _resolve(sym, "TerrainGenerator", "terrain.terrain_generator", "terrain")
    TerrainConfig = _resolve(sym, "TerrainConfig", "terrain.terrain_config", "terrain")
    Terrain = _resolve(sym, "Terrain", "terrain.terrain", "terrain")

    tg_mod = _mod("terrain.terrain_generator")

    # fixwd outputs for the patched pipeline steps.
    flow = np.full((3, 3), 7.0, dtype=np.float32)
    slope = np.full((3, 3), 11.0, dtype=np.float32)
    aspect = np.full((3, 3), 22.0, dtype=np.float32)

    class _FA:
        def compute(self, hm):  # noqa: ARG002
            return flow

    class _DC:
        def apply(self, hm, f):
            assert f is flow
            return hm

    class _SAC:
        def __init__(self, resolution):  # noqa: ARG002
            pass

        def compute(self, hm):
            # slope/aspect  computed after height scaling
            assert float(hm[0, 0]) == pytest.approx(1.0)
            return slope, aspect

    monkeypatch.setattr(tg_mod, "FlowAccumulator", _FA)
    monkeypatch.setattr(tg_mod, "DrainageCarver", _DC)
    monkeypatch.setattr(tg_mod, "SlopeAspectCalculator", _SAC)

    base_hm = np.full((3, 3), 0.5, dtype=np.float32)
    micro_out = np.full((3, 3), 0.5, dtype=np.float32) 
    noise = _NoiseStub(base_hm)
    micro = _MicroStub(out=micro_out)
    moisture_expected = np.full((3, 3), 0.33, dtype=np.float32)
    moisture_stub = _MoistureStub(moisture_expected)

    gen = TerrainGenerator(noise=noise, micro=micro, moisture_model=moisture_stub)

    cfg = TerrainConfig(
        size=2.0,
        resolution=1.0,
        scale=10.0,
        apply_microrelief=True,
        height_scale=2.0,
    )

    t = gen.generate(cfg)

    assert isinstance(t, Terrain)
    assert noise.called is True
    assert micro.called is True
    assert moisture_stub.called is True

    # height scaling  applied after carving
    assert float(t.heightmap[0, 0]) == pytest.approx(1.0)
    assert t.heightmap.shape == (cfg.rows, cfg.cols)

    #  model must be called with the outputs  pipeline
    f_arg, s_arg, a_arg = moisture_stub.args
    assert f_arg is flow
    assert s_arg is slope
    assert a_arg is aspect

    assert np.allclose(t.moisture, moisture_expected)
