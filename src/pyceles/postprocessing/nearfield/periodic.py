from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .components import NearFieldComponents
from .periodic_exterior import compute_periodic_near_field_exterior
from .slice import NearFieldSlice, interpolate_center_pixels, slice_plane_metadata

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


def compute_periodic_near_field(
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    field_bmax: float | None = None,
    slab_tolerance: float = 1e-12,
) -> NearFieldComponents:
    """Evaluate periodic near fields with exterior-order evaluation.

    This production path intentionally does not use replicated real-space image
    sums. Points intersecting the particle slab are routed to the local
    periodic evaluator, which is not implemented yet in this slice.
    The exterior order basis is explicit: callers must provide `field_bmax`, or
    configure `run.config.periodic.options.output_bmax`.
    """
    return compute_periodic_near_field_exterior(
        run,
        points=points,
        channel=channel,
        field_bmax=field_bmax,
        slab_tolerance=slab_tolerance,
    )


def compute_periodic_near_field_slice(
    run: SimulationResult,
    *,
    axis_0_min: float = -4000.0,
    axis_0_max: float = 4000.0,
    axis_1_min: float = -3000.0,
    axis_1_max: float = 5000.0,
    dx: float = 40.0,
    plane: str = "y",
    plane_value: float = 0.0,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    field_bmax: float | None = None,
    slab_tolerance: float = 1e-12,
    center_pixel_policy: Literal["none", "interpolate"] = "interpolate",
) -> NearFieldSlice:
    """Evaluate periodic near fields on an axis-aligned planar slice."""
    if float(dx) <= 0.0:
        raise ValueError(f"`dx` must be > 0. Got {dx!r}.")
    if float(axis_0_max) < float(axis_0_min):
        raise ValueError(
            f"`axis_0_max` must be >= `axis_0_min`. Got axis_0_min={axis_0_min!r}, axis_0_max={axis_0_max!r}."
        )
    if float(axis_1_max) < float(axis_1_min):
        raise ValueError(
            f"`axis_1_max` must be >= `axis_1_min`. Got axis_1_min={axis_1_min!r}, axis_1_max={axis_1_max!r}."
        )

    plane_name = str(plane).lower()
    if plane_name not in {"x", "y", "z"}:
        raise ValueError(f"`plane` must be one of 'x', 'y', or 'z'. Got {plane!r}.")
    _, _, _, axis_0_label, axis_1_label = slice_plane_metadata(plane_name)
    policy = str(center_pixel_policy).lower()
    if policy not in {"none", "interpolate"}:
        raise ValueError(
            f"`center_pixel_policy` must be 'none' or 'interpolate'. Got {center_pixel_policy!r}."
        )

    axis_0_vals = np.arange(float(axis_0_min), float(axis_0_max) + float(dx), float(dx))
    axis_1_vals = np.arange(float(axis_1_min), float(axis_1_max) + float(dx), float(dx))
    axis_0_grid, axis_1_grid = np.meshgrid(axis_0_vals, axis_1_vals, indexing="xy")
    axis_const = np.zeros_like(axis_0_grid)
    if plane_name == "x":
        pts = np.stack(
            [np.full_like(axis_0_grid, float(plane_value)), axis_0_grid, axis_1_grid], axis=-1
        )
    elif plane_name == "z":
        pts = np.stack(
            [axis_0_grid, axis_1_grid, np.full_like(axis_0_grid, float(plane_value))], axis=-1
        )
    else:
        pts = np.stack([axis_0_grid, axis_const + float(plane_value), axis_1_grid], axis=-1)

    nf = compute_periodic_near_field(
        run,
        points=pts,
        channel=channel,
        field_bmax=field_bmax,
        slab_tolerance=slab_tolerance,
    )
    field_dtype = np.dtype(getattr(run.config, "compute_dtype", "complex128"))
    field_maps = {
        "initial": (
            nf.E_initial.astype(field_dtype, copy=False),
            nf.H_initial.astype(field_dtype, copy=False),
        ),
        "scattered": (
            nf.E_scattered.astype(field_dtype, copy=False),
            nf.H_scattered.astype(field_dtype, copy=False),
        ),
        "internal": (
            nf.E_internal.astype(field_dtype, copy=False),
            nf.H_internal.astype(field_dtype, copy=False),
        ),
        "total": (
            nf.E_total.astype(field_dtype, copy=False),
            nf.H_total.astype(field_dtype, copy=False),
        ),
    }
    if policy == "interpolate":
        interpolate_center_pixels(
            run=run,
            axis_0_grid=axis_0_grid,
            axis_1_grid=axis_1_grid,
            field_maps=field_maps,
            plane=plane_name,
            plane_value=float(plane_value),
        )
    return NearFieldSlice(
        axis_0=axis_0_grid,
        axis_1=axis_1_grid,
        inside=np.asarray(nf.inside_mask).reshape(axis_1_grid.shape),
        field_maps=field_maps,
        plane=plane_name,
        plane_value=float(plane_value),
        axis_0_label=axis_0_label,
        axis_1_label=axis_1_label,
    )


__all__ = ["compute_periodic_near_field", "compute_periodic_near_field_slice"]
