from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .components import NearFieldComponents
from .periodic_exterior import (
    _slab_classification_tolerance,
    _slab_z_bounds,
    compute_periodic_near_field_exterior,
)
from .periodic_interior import compute_periodic_near_field_interior
from .slice import (
    NearFieldSlice,
    interpolate_center_pixels,
    reshape_field_points,
    slice_plane_metadata,
)

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


def compute_periodic_near_field(
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
    field_bmax: float | None = None,
    slab_tolerance: float = 1e-12,
    show_progress: bool = False,
) -> NearFieldComponents:
    """Evaluate periodic near fields with slab-aware production dispatch.

    This production path intentionally does not use replicated real-space image
    sums. Points above/below the particle slab use the exterior Rayleigh-order
    evaluator, while points intersecting the slab use the local periodic
    evaluator. The exterior order basis is explicit: callers must provide
    `field_bmax`, or configure `run.config.periodic.options.output_bmax`.
    """
    pts_flat, lead_shape = reshape_field_points(points)
    n_points = int(pts_flat.shape[0])

    if run.n_particles == 0:
        outer_mask = np.ones((n_points,), dtype=bool)
        slab_mask = np.zeros((n_points,), dtype=bool)
    else:
        z_min, z_max = _slab_z_bounds(run)
        tol = _slab_classification_tolerance(
            points_z=np.asarray(pts_flat[:, 2], dtype=float),
            z_min=z_min,
            z_max=z_max,
            accum_dtype=str(run.accum_dtype),
            user_tol=float(slab_tolerance),
        )
        above_mask = np.asarray(pts_flat[:, 2] > (z_max + tol), dtype=bool)
        below_mask = np.asarray(pts_flat[:, 2] < (z_min - tol), dtype=bool)
        outer_mask = above_mask | below_mask
        slab_mask = ~outer_mask

    out_dtype = np.dtype(run.accum_dtype)
    e_initial = np.zeros((n_points, 3), dtype=out_dtype)
    h_initial = np.zeros((n_points, 3), dtype=out_dtype)
    e_scat = np.zeros((n_points, 3), dtype=out_dtype)
    h_scat = np.zeros((n_points, 3), dtype=out_dtype)
    e_internal = np.zeros((n_points, 3), dtype=out_dtype)
    h_internal = np.zeros((n_points, 3), dtype=out_dtype)
    e_total = np.zeros((n_points, 3), dtype=out_dtype)
    h_total = np.zeros((n_points, 3), dtype=out_dtype)
    inside_mask = np.zeros((n_points,), dtype=bool)

    if np.any(outer_mask):
        nf_ext = compute_periodic_near_field_exterior(
            run,
            points=pts_flat[outer_mask],
            channel=channel,
            field_bmax=field_bmax,
            slab_tolerance=slab_tolerance,
        )
        e_initial[outer_mask] = np.asarray(nf_ext.E_initial, dtype=out_dtype).reshape(-1, 3)
        h_initial[outer_mask] = np.asarray(nf_ext.H_initial, dtype=out_dtype).reshape(-1, 3)
        e_scat[outer_mask] = np.asarray(nf_ext.E_scattered, dtype=out_dtype).reshape(-1, 3)
        h_scat[outer_mask] = np.asarray(nf_ext.H_scattered, dtype=out_dtype).reshape(-1, 3)
        e_internal[outer_mask] = np.asarray(nf_ext.E_internal, dtype=out_dtype).reshape(-1, 3)
        h_internal[outer_mask] = np.asarray(nf_ext.H_internal, dtype=out_dtype).reshape(-1, 3)
        e_total[outer_mask] = np.asarray(nf_ext.E_total, dtype=out_dtype).reshape(-1, 3)
        h_total[outer_mask] = np.asarray(nf_ext.H_total, dtype=out_dtype).reshape(-1, 3)
        inside_mask[outer_mask] = np.asarray(nf_ext.inside_mask, dtype=bool).reshape(-1)

    if np.any(slab_mask):
        nf_int = compute_periodic_near_field_interior(
            run,
            points=pts_flat[slab_mask],
            channel=channel,
            show_progress=show_progress,
        )
        e_initial[slab_mask] = np.asarray(nf_int.E_initial, dtype=out_dtype).reshape(-1, 3)
        h_initial[slab_mask] = np.asarray(nf_int.H_initial, dtype=out_dtype).reshape(-1, 3)
        e_scat[slab_mask] = np.asarray(nf_int.E_scattered, dtype=out_dtype).reshape(-1, 3)
        h_scat[slab_mask] = np.asarray(nf_int.H_scattered, dtype=out_dtype).reshape(-1, 3)
        e_internal[slab_mask] = np.asarray(nf_int.E_internal, dtype=out_dtype).reshape(-1, 3)
        h_internal[slab_mask] = np.asarray(nf_int.H_internal, dtype=out_dtype).reshape(-1, 3)
        e_total[slab_mask] = np.asarray(nf_int.E_total, dtype=out_dtype).reshape(-1, 3)
        h_total[slab_mask] = np.asarray(nf_int.H_total, dtype=out_dtype).reshape(-1, 3)
        inside_mask[slab_mask] = np.asarray(nf_int.inside_mask, dtype=bool).reshape(-1)

    if lead_shape == ():
        vec_shape: tuple[int, ...] = (3,)
        mask_shape: tuple[int, ...] = ()
    else:
        vec_shape = (*lead_shape, 3)
        mask_shape = lead_shape
    return NearFieldComponents(
        E_initial=e_initial.reshape(vec_shape),
        H_initial=h_initial.reshape(vec_shape),
        E_scattered=e_scat.reshape(vec_shape),
        H_scattered=h_scat.reshape(vec_shape),
        E_internal=e_internal.reshape(vec_shape),
        H_internal=h_internal.reshape(vec_shape),
        E_total=e_total.reshape(vec_shape),
        H_total=h_total.reshape(vec_shape),
        inside_mask=inside_mask.reshape(mask_shape),
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
    show_progress: bool = False,
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
        show_progress=show_progress,
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
