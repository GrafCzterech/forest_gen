"""forest_gen – procedural forest-generation toolkit"""

__all__ = ["ForestGenSpec", "TraversabilityMapBuilder"]

def __getattr__(name):
    if name == "ForestGenSpec":
        from .scene import ForestGenSpec

        return ForestGenSpec
    if name == "TraversabilityMapBuilder":
        from .travelsibilitymap import TraversabilityMapBuilder

        return TraversabilityMapBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name}")