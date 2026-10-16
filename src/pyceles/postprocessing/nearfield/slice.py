from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NearFieldSlice:
    """Near-field payload on a 2D slice for plotting and diagnostics.

    `field_maps` stores complex vector fields `(E, H)` for each family
    (`initial`, `scattered`, `internal`, `total`) sampled on the same grid.
    Field arrays retain the evaluator's values and accumulation dtype.
    """

    axis_0: np.ndarray
    axis_1: np.ndarray
    inside: np.ndarray
    field_maps: dict[str, tuple[np.ndarray, np.ndarray]]
    plane: str
    plane_value: float
    axis_0_label: str
    axis_1_label: str


def slice_plane_metadata(plane: str) -> tuple[int, int, int, str, str]:
    """Map plane label to normal/tangential coordinate indices and axis labels."""
    p = str(plane).lower()
    if p == "x":
        return 0, 1, 2, "y", "z"
    if p == "y":
        return 1, 0, 2, "x", "z"
    if p == "z":
        return 2, 0, 1, "x", "y"
    raise ValueError("`plane` must be one of 'x', 'y', or 'z'.")


def _slice_grid(
    *,
    axis_0_min: float,
    axis_0_max: float,
    axis_1_min: float,
    axis_1_max: float,
    dx: float,
    plane: str,
    plane_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one Cartesian slice grid for finite and periodic field evaluators."""
    if float(dx) <= 0:
        raise ValueError(f"`dx` must be > 0. Got {dx!r}.")
    if float(axis_0_max) < float(axis_0_min):
        raise ValueError(
            f"`axis_0_max` must be >= `axis_0_min`. Got {axis_0_max!r} < {axis_0_min!r}."
        )
    if float(axis_1_max) < float(axis_1_min):
        raise ValueError(
            f"`axis_1_max` must be >= `axis_1_min`. Got {axis_1_max!r} < {axis_1_min!r}."
        )
    normal, axis_0, axis_1, _, _ = slice_plane_metadata(plane)
    values_0 = np.arange(float(axis_0_min), float(axis_0_max) + float(dx), float(dx))
    values_1 = np.arange(float(axis_1_min), float(axis_1_max) + float(dx), float(dx))
    grid_0, grid_1 = np.meshgrid(values_0, values_1, indexing="xy")
    points = np.empty((*grid_0.shape, 3), dtype=float)
    points[..., normal] = float(plane_value)
    points[..., axis_0] = grid_0
    points[..., axis_1] = grid_1
    return grid_0, grid_1, points


def reshape_field_points(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Normalize arbitrary point inputs to flat `(N,3)` plus original leading shape."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(
                f"1D `points` input must have exactly 3 entries (x,y,z). Got shape {arr.shape}."
            )
        return arr.reshape(1, 3), ()
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        lead_shape = tuple(arr.shape[:-1])
        return arr.reshape(-1, 3), lead_shape
    raise ValueError(f"`points` must be shaped (3,), (N,3), or (...,3). Got shape {arr.shape}.")


__all__ = [
    "NearFieldSlice",
    "reshape_field_points",
    "slice_plane_metadata",
]
