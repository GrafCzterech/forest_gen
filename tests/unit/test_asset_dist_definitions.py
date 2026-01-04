import math
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


@pytest.mark.unit
def test_viability_map_is_bounded_and_deterministic_when_seed_is_forced(sym, monkeypatch):
    pytest.importorskip("opensimplex")

    monkeypatch.setattr(random, "randint", lambda a, b: 1234)

    ViabilityMap = _resolve(sym, "ViabilityMap", "asset_dist.definitions", "definitions", "asset_dist")

    vm1 = ViabilityMap(eps=0.2)
    vm2 = ViabilityMap(eps=0.2)

    v1 = vm1(10.0, 20.0)
    v2 = vm2(10.0, 20.0)

    assert 0.0 <= v1 <= 1.0
    assert 0.0 <= v2 <= 1.0
    assert v1 == pytest.approx(v2, abs=1e-12)


@pytest.mark.unit
def test_viability_map_eps_zero_raises(sym, monkeypatch):
    pytest.importorskip("opensimplex")

    monkeypatch.setattr(random, "randint", lambda a, b: 1)
    ViabilityMap = _resolve(sym, "ViabilityMap", "asset_dist.definitions", "definitions", "asset_dist")

    vm = ViabilityMap(eps=0.0)
    with pytest.raises(ZeroDivisionError):
        _ = vm(1.0, 1.0)


@pytest.mark.unit
def test_species_equality_and_hash_use_name_only(sym):
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")

    s1 = Species(name="Oak", max_age=100, reproduction_rate=1, radius=0.5)
    s2 = Species(name="Oak", max_age=10, reproduction_rate=999, radius=9.9)
    s3 = Species(name="Pine", max_age=100, reproduction_rate=1, radius=0.5)

    assert s1 == s2
    assert s1 != s3

    # Set/dict semantics must match equality.
    assert len({s1, s2, s3}) == 2
    d = {s1: "a"}
    d[s2] = "b"
    assert len(d) == 1
    assert d[s1] == "b"


@pytest.mark.unit
def test_plant_vt_returns_0_for_non_positive_max_age(sym):
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "definitions", "asset_dist")

    sp = Species(name="Bad", max_age=0)
    p = Plant(coords=(0.0, 0.0), species=sp, age=5)
    assert p.vt() == 0.0


@pytest.mark.unit
def test_plant_vt_age_curve_without_juvenile_spike(sym):
    """
    This checks the core shape of vt() using parameters that make it predictable:
    - juvenile mortality depth = 0 => pure growth to 1 by juvenile_recovery_age
    - senescence begins at 0.7 and trends to plateau
    """
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "definitions", "asset_dist")

    sp = Species(
        name="Test",
        max_age=100,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        senescence_start=0.7,
        senescence_plateau=0.5,
        senescence_plateau_span=0.15,
    )

    p0 = Plant((0.0, 0.0), sp, age=0)
    p10 = Plant((0.0, 0.0), sp, age=10)   # norm=0.1 => growth_phase=0.5
    p20 = Plant((0.0, 0.0), sp, age=20)   # norm=0.2 => full viability
    p70 = Plant((0.0, 0.0), sp, age=70)   # senescence start => still 1.0
    p80 = Plant((0.0, 0.0), sp, age=80)   # moving toward plateau
    p90 = Plant((0.0, 0.0), sp, age=90)   # plateau

    assert p0.vt() == pytest.approx(0.0, abs=1e-12)
    assert p10.vt() == pytest.approx(0.5, abs=1e-6)
    assert p20.vt() == pytest.approx(1.0, abs=1e-6)
    assert p70.vt() == pytest.approx(1.0, abs=1e-6)

    assert 0.5 < p80.vt() < 1.0
    assert p90.vt() == pytest.approx(0.5, abs=1e-6)

    for p in (p0, p10, p20, p70, p80, p90):
        assert 0.0 <= p.vt() <= 1.0


@pytest.mark.unit
def test_plant_vt_juvenile_spike_reduces_viability_near_peak(sym):
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "definitions", "asset_dist")

    sp = Species(
        name="Spike",
        max_age=100,
        juvenile_mortality_depth=0.4,
        juvenile_mortality_peak=0.05,
        juvenile_mortality_width=0.03,
        juvenile_recovery_age=0.2,
        senescence_start=0.7,
    )

    p5 = Plant((0.0, 0.0), sp, age=5)
    p10 = Plant((0.0, 0.0), sp, age=10)

    # At peak, vt should be lower than at age 10 for these params.
    assert p5.vt() < p10.vt()
    assert 0.0 <= p5.vt() <= 1.0
    assert 0.0 <= p10.vt() <= 1.0


@pytest.mark.unit
def test_plant_vt_prim_applies_spatial_viability_and_population_weight(sym):
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "definitions", "asset_dist")

    def vmap(x: float, y: float) -> float:  # noqa: ARG001
        return 0.25

    sp = Species(name="Pop", max_age=100, viability_map=vmap, juvenile_mortality_depth=0.0, juvenile_recovery_age=0.2)
    plant = Plant(coords=(1.0, 2.0), species=sp, age=20)  

    pop = {sp: 5}
    total = 20

    val = plant.vt_prim(pop, total)
    assert val == pytest.approx(0.25 * (5 / 20) * plant.vt(), abs=1e-8)
    assert 0.0 <= val <= 1.0


@pytest.mark.unit
def test_seed_returns_same_species_and_age_zero(sym):
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "definitions", "asset_dist")

    sp = Species(
        name="Seeder",
        max_age=10,
        reproduction_rate=5,
        reproduction_radius=3.0,
        species_density=0.02,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
    )
    p = Plant((10.0, 10.0), sp, age=5)

    random.seed(123)
    seeds = list(p.seed())

    assert len(seeds) <= sp.reproduction_rate
    for s in seeds:
        assert s.species is sp
        assert s.age == 0


@pytest.mark.unit
def test_seed_does_not_exceed_reproduction_radius(sym):
    Species = _resolve(sym, "Species", "asset_dist.definitions", "definitions", "asset_dist")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "definitions", "asset_dist")


    sp = Species(
        name="SeederBug",
        max_age=10,
        reproduction_rate=10,
        reproduction_radius=2.5,
        species_density=10.0, #Fix bug later, then test should pass
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
    )
    p = Plant((0.0, 0.0), sp, age=5)

    random.seed(0)
    seeds = list(p.seed())
    assert seeds

    for s in seeds:
        d = math.dist(p.coords, s.coords)
        assert d <= sp.reproduction_radius + 1e-9
