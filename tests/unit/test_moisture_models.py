import pytest
import numpy as np


def _resolve(sym, name: str, *paths: str):
    errors = []
    for p in paths:
        try:
            return sym(p, name)
        except Exception as e:
            errors.append(f"{p}.{name}: {type(e).__name__}({e})")
    raise RuntimeError(
        f"Could not resolve symbol '{name}' from any of: {paths}\n" + "\n".join(errors)
    )


@pytest.mark.unit
def test_default_moisture_constant_inputs_are_finite_and_zero(sym):
    DefaultMoistureModel = _resolve(
        sym,
        "DefaultMoistureModel",
        "terrain.moisture",
        "terrain.moisture.default_moisture_model",
    )

    model = DefaultMoistureModel()

    flow = np.full((4, 4), 10.0, dtype=np.float32)
    slope = np.full((4, 4), 30.0, dtype=np.float32)
    aspect = np.full((4, 4), 90.0, dtype=np.float32)

    m = model.compute(flow, slope, aspect)

    assert m.shape == (4, 4)
    assert np.isfinite(m).all()
    assert float(m.min()) == 0.0
    assert float(m.max()) == 0.0


@pytest.mark.unit
def test_default_moisture_flow_only_matches_flow_normalization(sym):
    DefaultMoistureModel = _resolve(
        sym,
        "DefaultMoistureModel",
        "terrain.moisture",
        "terrain.moisture.default_moisture_model",
    )

    model = DefaultMoistureModel(weights={"flow": 1.0})

    flow = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    slope = np.zeros_like(flow)
    aspect = np.zeros_like(flow)

    m = model.compute(flow, slope, aspect)

    # flow in [0,1]
    expected = (flow - flow.min()) / (np.ptp(flow) + 1e-8)
    expected -= expected.min()
    expected /= expected.max() + 1e-8

    assert np.isfinite(m).all()
    assert np.allclose(m, expected, atol=1e-6)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0


@pytest.mark.unit
def test_default_moisture_slope_only_penalizes_steep_slopes(sym):
    DefaultMoistureModel = _resolve(
        sym,
        "DefaultMoistureModel",
        "terrain.moisture",
        "terrain.moisture.default_moisture_model",
    )

    model = DefaultMoistureModel(weights={"slope": 1.0})

    #  both 0 and 90 to  span [1,0]
    slope = np.array([[0.0, 45.0], [90.0, 30.0]], dtype=np.float32)
    flow = np.zeros_like(slope)
    aspect = np.zeros_like(slope)

    m = model.compute(flow, slope, aspect)

    slope_penalty = 1.0 - (slope / 90.0)
    expected = slope_penalty.copy()
    expected -= expected.min()
    expected /= expected.max() + 1e-8

    assert np.isfinite(m).all()
    assert np.allclose(m, expected, atol=1e-6)
    # Steepest slope should be least moist
    assert m[1, 0] <= m[0, 0]


@pytest.mark.unit
def test_default_moisture_aspect_only_prefers_aspect_0_over_180(sym):
    DefaultMoistureModel = _resolve(
        sym,
        "DefaultMoistureModel",
        "terrain.moisture",
        "terrain.moisture.default_moisture_model",
    )

    model = DefaultMoistureModel(weights={"aspect": 1.0})

    aspect = np.array([[0.0, 180.0], [90.0, 270.0]], dtype=np.float32)
    flow = np.zeros_like(aspect)
    slope = np.zeros_like(aspect)

    m = model.compute(flow, slope, aspect)

    assert np.isfinite(m).all()
    assert m[0, 0] > m[0, 1]


@pytest.mark.unit
def test_default_moisture_missing_weight_keys_are_treated_as_zero(sym):
    DefaultMoistureModel = _resolve(
        sym,
        "DefaultMoistureModel",
        "terrain.moisture",
        "terrain.moisture.default_moisture_model",
    )

    # Only "flow" provided; "slope"/"aspect" should  0 
    model = DefaultMoistureModel(weights={"flow": 1.0})

    flow = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    slope = np.array([[90.0, 90.0], [0.0, 0.0]], dtype=np.float32)
    aspect = np.array([[180.0, 180.0], [0.0, 0.0]], dtype=np.float32)

    m = model.compute(flow, slope, aspect)

    expected = (flow - flow.min()) / (np.ptp(flow) + 1e-8)
    expected -= expected.min()
    expected /= expected.max() + 1e-8

    assert np.allclose(m, expected, atol=1e-6)
