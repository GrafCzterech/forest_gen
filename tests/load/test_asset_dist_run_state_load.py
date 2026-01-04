import os
import time
import math
import random
import pytest


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


def _make_species(Species, *, name="LoadSpecies"):
    # Keep vt() predictable and avoid early-life penalty surprises.
    return Species(
        name=name,
        max_age=200,
        radius=0.35,               # small clearance => can pack many plants
        species_density=0.01,       # NOTE: seed() uses this as min distance (buggy units)
        reproduction_rate=2,        # mild growth pressure per year
        reproduction_radius=1.5,    # local dispersal
        juvenile_mortality_depth=0.0,
        juvenile_recovery_age=0.2,
        viability_map=lambda x, y: 1.0,  # noqa: E731
    )


def _grid_points(size: float, step: float, margin: float, n: int):
    """Deterministic dense-but-non-overlapping initial population."""
    pts = []
    x = margin
    while x <= size - margin and len(pts) < n:
        y = margin
        while y <= size - margin and len(pts) < n:
            pts.append((x, y))
            y += step
        x += step
    if len(pts) < n:
        raise RuntimeError(f"Could only place {len(pts)} points; need {n}. Increase size or reduce step/margin.")
    return pts


@pytest.mark.load
def test_run_state_load_large_population_bounded_growth(sym):
    """
    Load test: SimulationState.run_state on a large initial population.

    What this catches:
      - pathological slowdowns in neighbor queries / overlap handling
      - runaway spawning ignoring max_population
      - out-of-bounds plants
      - accidental non-termination

    Optional perf gate:
      - set LOAD_RUN_STATE_BUDGET_SEC (float) to enforce a wall-time budget.
        Example: LOAD_RUN_STATE_BUDGET_SEC=20
    """
    Species = _resolve(sym, "Species", "asset_dist.definitions", "asset_dist", "definitions")
    Plant = _resolve(sym, "Plant", "asset_dist.definitions", "asset_dist", "definitions")
    SimulationState = _resolve(sym, "SimulationState", "asset_dist.state", "asset_dist", "state")

    # --- Load parameters (tune via env vars if needed) ---
    size = float(os.getenv("LOAD_SCENE_SIZE", "140"))          # meters
    initial_n = int(os.getenv("LOAD_INIT_N", "5000"))          # starting population
    years = int(os.getenv("LOAD_YEARS", "8"))
    max_population = int(os.getenv("LOAD_MAX_POP", "6500"))
    div = int(os.getenv("LOAD_GRID_DIV", "70"))                # spatial binning; higher => less per-cell crowding

    # Optional performance budget (seconds). If unset, we don't enforce time.
    budget_env = os.getenv("LOAD_RUN_STATE_BUDGET_SEC")
    budget = float(budget_env) if budget_env is not None else None

    sp = _make_species(Species)

    # Ensure initial plants are safely in-bounds even with reproduction.
    margin = sp.reproduction_radius + 2.0
    # Choose step so initial plants don't overlap under your clearance rule (max radii).
    step = max(2.05 * sp.radius, 0.8)

    points = _grid_points(size=size, step=step, margin=margin, n=initial_n)
    plants = [Plant(coords=p, species=sp, age=5) for p in points]  # age>0 so vt() is nonzero

    state = SimulationState(plants, size=(size, size), div=div)

    # Deterministic randomness for seed() and run_state loop.
    random.seed(0)

    t0 = time.perf_counter()
    state.run_state(years, max_population=max_population)
    dt = time.perf_counter() - t0

    # --- Correctness invariants ---
    assert len(state) <= max_population
    assert len(state) >= 1  # should never go empty in this setup

    for plant in state:
        x, y = plant.coords
        assert 0.0 <= x <= size
        assert 0.0 <= y <= size
        assert plant.age >= 0

    # Optional perf assertion
    if budget is not None:
        assert dt <= budget, f"run_state took {dt:.2f}s, budget={budget:.2f}s"

    # Tiny sanity check that time wasn't 'zero' (guards against accidentally skipping run_state)
    assert dt > 0.0
