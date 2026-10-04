from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NearFieldSlice:
    """Near-field payload on a 2D slice for plotting and diagnostics.

    `field_maps` stores complex vector fields `(E, H)` for each family
    (`initial`, `scattered`, `internal`, `total`) sampled on the same grid.
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
    raise ValueError("plane must be one of 'x', 'y', or 'z'.")


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


def _neighbor_mean_inplace(arr: np.ndarray, i0: int, i1: int) -> None:
    """Replace one sample with the mean of available 4-neighborhood samples."""
    n0, n1 = arr.shape[0], arr.shape[1]
    values = []
    for d0, d1 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        j0 = i0 + d0
        j1 = i1 + d1
        if 0 <= j0 < n0 and 0 <= j1 < n1:
            values.append(arr[j0, j1])
    if values:
        arr[i0, i1] = np.mean(np.asarray(values), axis=0)


def interpolate_center_pixels(
    *,
    run,
    axis_0_grid: np.ndarray,
    axis_1_grid: np.ndarray,
    field_maps: dict[str, tuple[np.ndarray, np.ndarray]],
    plane: str,
    plane_value: float,
) -> None:
    """Patch singular center pixels by local interpolation on exact center hits."""
    p = np.asarray(run.positions, dtype=float)
    normal_idx, axis_0_idx, axis_1_idx, _, _ = slice_plane_metadata(plane)
    on_plane = np.isclose(p[:, normal_idx], plane_value, rtol=0.0, atol=1e-12)
    u = p[on_plane, axis_0_idx]
    v = p[on_plane, axis_1_idx]

    if u.size == 0:
        return

    axis_0_values = np.asarray(axis_0_grid[0, :], dtype=float)
    axis_1_values = np.asarray(axis_1_grid[:, 0], dtype=float)
    for uu, vv in zip(u, v, strict=True):
        i0 = int(np.argmin(np.abs(axis_0_values - uu)))
        i1 = int(np.argmin(np.abs(axis_1_values - vv)))
        if not (
            np.isclose(axis_0_values[i0], uu, rtol=0.0, atol=1e-12)
            and np.isclose(axis_1_values[i1], vv, rtol=0.0, atol=1e-12)
        ):
            continue
        for key in field_maps:
            e_map, h_map = field_maps[key]
            _neighbor_mean_inplace(e_map, i1, i0)
            _neighbor_mean_inplace(h_map, i1, i0)


__all__ = [
    "NearFieldSlice",
    "interpolate_center_pixels",
    "reshape_field_points",
    "slice_plane_metadata",
]
