"""Optional dependency helpers."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np


@cache
def import_cupy() -> tuple[ModuleType, ModuleType]:
    """Import CuPy and sparse-linalg helpers with a clear failure mode."""
    try:
        cupy = import_module("cupy")
        cupyx_sparse_linalg = import_module("cupyx.scipy.sparse.linalg")
    except Exception as exc:  # pragma: no cover - exercised via callers
        raise RuntimeError(
            "CuPy support was requested, but CuPy could not be initialized. "
            "Ensure the optional GPU dependency is installed and that the CUDA "
            "runtime/toolkit paths are configured correctly for this machine. "
            f"Original error: {exc}"
        ) from exc
    return cupy, cupyx_sparse_linalg


def is_cupy_array(x: Any) -> bool:
    """Return True when `x` is a CuPy array without importing CuPy eagerly."""
    mod = type(x).__module__
    return mod == "cupy" or mod.startswith("cupy.")


def asnumpy(x: Any) -> np.ndarray:
    """Convert NumPy/CuPy-like arrays to a NumPy ndarray."""
    if is_cupy_array(x):
        cupy, _ = import_cupy()
        return np.asarray(cupy.asnumpy(x))
    return np.asarray(x)


def coerce_array(x: Any, *, dtype: np.dtype, prefer_cupy: bool = False) -> Any:
    """Convert input to NumPy or CuPy according to the requested preference."""
    if prefer_cupy or is_cupy_array(x):
        cupy, _ = import_cupy()
        return cupy.asarray(x, dtype=dtype)
    return np.asarray(x, dtype=dtype)
