import os
import time
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


def _make_species(
    Species,
    *,
    name: str,
    radius: float,
    density: float,
):
    return Species(
        name=name,
        max_age=200,
        radius=radius,
        species_density=density,
        reproduction_rate=0,
        reproduction_radius=2.0,
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )


def _assert_bounds(plants, size: tuple[float, float]):
    w, h = size
    for p in plants:
        x, y = p.coords
        assert 0.0 <= x <= w
        assert 0.0 <= y <= h
        assert p.age == 0


def _assert_clearance_two_groups(big_pts, small_pts, big_r: float, small_r: float):
    """
    sim.py clearance rule is max(r1, r2), so for big/small it's:
      - big-big >= big_r
      - small-small >= small_r
      - big-small >= big_r
    KDTree O(N log N) checks.
    """
    from scipy.spatial import KDTree

    eps = 1e-9

    if len(big_pts) > 1:
        tree = KDTree(big_pts)
        d, _ = tree.query(big_pts, k=2) 
        assert float(np.min(d[:, 1])) >= big_r - eps

    if len(small_pts) > 1:
        tree = KDTree(small_pts)
        d, _ = tree.query(small_pts, k=2)
        assert float(np.min(d[:, 1])) >= small_r - eps

    if len(big_pts) and len(small_pts):
        big_tree = KDTree(big_pts)
        d_to_big, _ = big_tree.query(small_pts, k=1)
        assert float(np.min(d_to_big)) >= big_r - eps

        small_tree = KDTree(small_pts)
        d_to_small, _ = small_tree.query(big_pts, k=1)
        assert float(np.min(d_to_small)) >= big_r - eps


@pytest.mark.load
def test_new_state_load_high_density_multispecies(sym):
    """
    Load test: Simulation.new_state() at high density and mixed radii.

    Catches:
      - performance cliffs (O(N^2) conflict checks)
      - out-of-bounds placements
      - overlap regressions (clearance rule changes)
      - starvation of large-radius species if radius-sorting is removed

    Optional perf gate:
      LOAD_NEW_STATE_BUDGET_SEC (float seconds).
    """
    pytest.importorskip("scipy")

    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Simulation = _resolve(sym, "Simulation", "asset_dist.sim", "asset_dist", "sim")

    size = float(os.getenv("LOAD_SCENE_SIZE", "200"))
    scene_density = float(os.getenv("LOAD_SCENE_DENSITY", "1.0"))

    # area=40k at size=200.
    big_radius = float(os.getenv("LOAD_BIG_RADIUS", "4.0"))
    small_radius = float(os.getenv("LOAD_SMALL_RADIUS", "1.2"))

    big_density = float(os.getenv("LOAD_BIG_DENSITY", "0.0008"))      # ~32
    small_density = float(os.getenv("LOAD_SMALL_DENSITY", "0.02"))    # ~800

    budget_env = os.getenv("LOAD_NEW_STATE_BUDGET_SEC")
    budget = float(budget_env) if budget_env is not None else None

    big = _make_species(Species, name="Big", radius=big_radius, density=big_density)
    small = _make_species(Species, name="Small", radius=small_radius, density=small_density)

    sim = Simulation(size=(size, size), species={"canopy": {big}, "understory": {small}})

    t0 = time.perf_counter()
    state = sim.new_state(scene_density=scene_density)
    dt = time.perf_counter() - t0

    plants = list(state)
    assert len(plants) > 0

    _assert_bounds(plants, (size, size))

    # Species presence (guards against "big starved by small" regression)
    big_plants = [p for p in plants if p.species.name == "Big"]
    small_plants = [p for p in plants if p.species.name == "Small"]
    assert len(big_plants) >= 1
    assert len(small_plants) >= 1

    # Clearance checks using KDTree (fast enough for load, unlike O(N^2) pairwise)
    big_pts = np.array([p.coords for p in big_plants], dtype=float)
    small_pts = np.array([p.coords for p in small_plants], dtype=float)
    _assert_clearance_two_groups(big_pts, small_pts, big_radius, small_radius)

    if budget is not None:
        assert dt <= budget, f"new_state took {dt:.2f}s, budget={budget:.2f}s"

    assert dt > 0.0
