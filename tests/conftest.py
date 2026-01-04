from __future__ import annotations

import os
import importlib
from typing import Any, Callable

import pytest
import numpy as np

PKG = os.environ.get("FOREST_PKG", "forest_gen_utils")
def _mod(path: str):
    full = f"{PKG}.{path}" if PKG else path
    return importlib.import_module(full)

def _sym(path: str, name: str) -> Any:
    return getattr(_mod(path), name)

@pytest.fixture
def sym() -> Callable[[str, str], Any]:
    return _sym

