import pytest


@pytest.mark.unit
def test_noise_factory_selects_strategy_case_insensitive(sym):
    # Keep runtime robust: if scipy/opensimplex missing, just skip relevant parts.
    NoiseFactory = sym("terrain.noise.noise_factory", "NoiseFactory")

    # Case-insensitivity is implemented via name.lower()
    strat = NoiseFactory.create("FRACTAL")  # type: ignore[arg-type]
    assert strat is not None
    assert strat.__class__.__name__.lower().startswith("fractal")


@pytest.mark.unit
def test_noise_factory_unknown_raises(sym):
    NoiseFactory = sym("terrain.noise.noise_factory", "NoiseFactory")

    with pytest.raises(ValueError) as e:
        NoiseFactory.create("nope")  # type: ignore[arg-type]

    assert "Unknown" in str(e.value)