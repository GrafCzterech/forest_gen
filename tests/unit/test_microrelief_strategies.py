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
def test_none_microrelief_is_noop_and_returns_same_object(sym):
    NoneMicrorelief = _resolve(
        sym,
        "NoneMicrorelief",
        "terrain.microrelief",
        "terrain.microrelief.none_microrelief",
    )

    hm = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    strat = NoneMicrorelief()

    out = strat.apply(hm)

    # Real behavior: it returns the same array object.
    assert out is hm
    assert np.allclose(out, hm)


@pytest.mark.unit
def test_basic_microrelief_strength_zero_is_exact_noop(sym):
    pytest.importorskip("scipy")

    BasicMicrorelief = _resolve(
        sym,
        "BasicMicrorelief",
        "terrain.microrelief",
        "terrain.microrelief.basic_microrelief",
    )

    hm = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    strat = BasicMicrorelief(strength=0.0, sigma=0.8)

    # Seed to avoid accidental failures if implementation changes
    state = np.random.get_state()
    try:
        np.random.seed(0)
        out = strat.apply(hm.copy())
    finally:
        np.random.set_state(state)

    # With strength=0, output should equal input (already in [0,1], so clip doesn't change).
    assert np.allclose(out, hm, atol=0.0)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


@pytest.mark.unit
def test_basic_microrelief_adds_variation_and_clips(sym):
    pytest.importorskip("scipy")

    BasicMicrorelief = _resolve(
        sym,
        "BasicMicrorelief",
        "terrain.microrelief",
        "terrain.microrelief.basic_microrelief",
    )

    hm = np.full((32, 32), 0.5, dtype=np.float32)
    strat = BasicMicrorelief(strength=0.02, sigma=1.0)

    state = np.random.get_state()
    try:
        np.random.seed(123)
        out = strat.apply(hm.copy())
    finally:
        np.random.set_state(state)

    assert out.shape == hm.shape
    assert np.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0

    # Should introduce non-zero variation for non-zero strength.
    assert float(out.std()) > 0.0
    # Should not be identical to the input constant field.
    assert not np.allclose(out, hm)


@pytest.mark.unit
def test_basic_microrelief_is_deterministic_if_numpy_rng_is_seeded(sym):
    pytest.importorskip("scipy")

    BasicMicrorelief = _resolve(
        sym,
        "BasicMicrorelief",
        "terrain.microrelief",
        "terrain.microrelief.basic_microrelief",
    )

    hm = np.full((16, 16), 0.25, dtype=np.float32)
    strat = BasicMicrorelief(strength=0.01, sigma=0.9)

    state = np.random.get_state()
    try:
        np.random.seed(999)
        a = strat.apply(hm.copy())
        np.random.seed(999)
        b = strat.apply(hm.copy())
    finally:
        np.random.set_state(state)

    assert np.allclose(a, b)
