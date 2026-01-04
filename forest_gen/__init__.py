# forest_gen/__init__.py
"""forest_gen – procedural forest-generation toolkit"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ForestGenSpec",
    "HeightmapTerrain",
    "PlantSpec",
    "TraversabilityMapBuilder",
    "TraversabilityConfig",
    "assets",
]


def __getattr__(name: str):
    if name in {"ForestGenSpec", "HeightmapTerrain", "PlantSpec", "assets"}:
        # scene depends on stripe_kit / isaaclab -> import lazily
        mod = import_module(".scene", __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    if name in {"TraversabilityMapBuilder", "TraversabilityConfig"}:
        # keep utils import lazy too (avoids accidental init-time cycles)
        mod = import_module("forest_gen_utils.traversability")
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
