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


# ---------- FlowAccumulator (D8 accumulation) ----------

@pytest.mark.unit
def test_flow_accumulator_flat_heightmap_all_ones(sym):
    FlowAccumulator = _resolve(
        sym,
        "FlowAccumulator",
        "terrain.utils",
        "terrain.utils.flow_accumulator",
        "terrain.flow_accumulator",
        "terrain.hydrology",
    )

    hm = np.zeros((5, 5), dtype=np.float32)
    acc = FlowAccumulator().compute(hm)

    assert acc.shape == hm.shape
    assert np.isfinite(acc).all()
    # On  flat surface all cell is sink
    assert np.allclose(acc, 1.0)


@pytest.mark.unit
def test_flow_accumulator_monotone_slope_outlet_collects_all_cells(sym):
    FlowAccumulator = _resolve(
        sym,
        "FlowAccumulator",
        "terrain.utils",
        "terrain.utils.flow_accumulator",
        "terrain.flow_accumulator",
        "terrain.hydrology",
    )

    #  toward bottom-right
    hm = np.array(
        [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1],
        ],
        dtype=np.float32,
    )

    acc = FlowAccumulator().compute(hm)

    assert acc[2, 2] == pytest.approx(9.0)
    assert acc[2, 2] == float(acc.max())


# ---------- SlopeAspectCalculator ----------

@pytest.mark.unit
def test_slope_aspect_flat_surface_slope_zero(sym):
    SlopeAspectCalculator = _resolve(
        sym,
        "SlopeAspectCalculator",
        "terrain.utils",
        "terrain.utils.slope_aspect_calculator",
        "terrain.slope_aspect_calculator",
        "terrain.analysis",
    )

    hm = np.zeros((7, 7), dtype=np.float32)
    slope, aspect = SlopeAspectCalculator(resolution=1.0).compute(hm)

    assert slope.shape == hm.shape
    assert aspect.shape == hm.shape
    assert np.isfinite(slope).all()
    assert np.isfinite(aspect).all()

    # Flat 
    assert np.allclose(slope, 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("resolution,expected_deg", [(1.0, 45.0), (2.0, np.degrees(np.arctan(0.5)))])
def test_slope_matches_known_ramp(sym, resolution, expected_deg):
    SlopeAspectCalculator = _resolve(
        sym,
        "SlopeAspectCalculator",
        "terrain.utils",
        "terrain.utils.slope_aspect_calculator",
        "terrain.slope_aspect_calculator",
        "terrain.analysis",
    )

    hm = np.tile(np.arange(0, 7, dtype=np.float32), (7, 1))
    slope, aspect = SlopeAspectCalculator(resolution=resolution).compute(hm)

    assert slope[3, 3] == pytest.approx(expected_deg, abs=1e-3)
    assert 0.0 <= float(aspect[3, 3]) < 360.0


@pytest.mark.unit
def test_aspect_flips_by_180_degrees_when_ramp_reverses(sym):
    SlopeAspectCalculator = _resolve(
        sym,
        "SlopeAspectCalculator",
        "terrain.utils",
        "terrain.utils.slope_aspect_calculator",
        "terrain.slope_aspect_calculator",
        "terrain.analysis",
    )

    calc = SlopeAspectCalculator(resolution=1.0)

    hm_inc = np.tile(np.arange(0, 7, dtype=np.float32), (7, 1))          # increasing x
    hm_dec = np.tile(np.arange(6, -1, -1, dtype=np.float32), (7, 1))     #  decreasing x

    _, aspect_inc = calc.compute(hm_inc)
    _, aspect_dec = calc.compute(hm_dec)

    a = float(aspect_inc[3, 3])
    b = float(aspect_dec[3, 3])

    diff = abs(((a - b) + 180.0) % 360.0 - 180.0)
    assert diff == pytest.approx(180.0, abs=1e-6)


# ---------- DrainageCarver ----------

@pytest.mark.unit
def test_drainage_carver_strength_zero_sigma_zero_is_noop(sym):
    pytest.importorskip("scipy")

    DrainageCarver = _resolve(
        sym,
        "DrainageCarver",
        "terrain.utils",
        "terrain.utils.drainage_carver",
        "terrain.drainage_carver",
        "terrain.hydrology",
    )

    hm = np.linspace(0.0, 1.0, 49, dtype=np.float32).reshape(7, 7)
    flow = np.random.default_rng(0).random((7, 7)).astype(np.float32)

    carver = DrainageCarver(strength=0.0, sigma=0.0)
    out = carver.apply(hm, flow)

    assert np.allclose(out, hm)
    assert out.shape == hm.shape


@pytest.mark.unit
def test_drainage_carver_reduces_height_where_flow_is_high(sym):
    pytest.importorskip("scipy")

    DrainageCarver = _resolve(
        sym,
        "DrainageCarver",
        "terrain.utils",
        "terrain.utils.drainage_carver",
        "terrain.drainage_carver",
        "terrain.hydrology",
    )

    hm = np.ones((5, 5), dtype=np.float32)
    flow = np.zeros((5, 5), dtype=np.float32)
    flow[2, 2] = 10.0  # peak flow

    strength = 0.3
    carver = DrainageCarver(strength=strength, sigma=0.0)
    out = carver.apply(hm, flow)

    # sigma=0 carving should be exact at the peak.
    assert out[2, 2] == pytest.approx(1.0 - strength, abs=1e-6)
    assert out[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


@pytest.mark.unit
def test_drainage_carver_clips_to_unit_interval(sym):
    pytest.importorskip("scipy")

    DrainageCarver = _resolve(
        sym,
        "DrainageCarver",
        "terrain.utils",
        "terrain.utils.drainage_carver",
        "terrain.drainage_carver",
        "terrain.hydrology",
    )

    hm = np.full((4, 4), 2.0, dtype=np.float32)  # out-of-range
    flow = np.zeros((4, 4), dtype=np.float32)

    out = DrainageCarver(strength=0.5, sigma=0.0).apply(hm, flow)

    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert np.allclose(out, 1.0)  # clipping dominates no carving  flow_norm=0
