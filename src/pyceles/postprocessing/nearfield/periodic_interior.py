from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyceles.core.periodic import PeriodicSpec, plane_wave_k_parallel
from pyceles.core.periodic.reception import apply_periodic_reception

from .classification import classify_periodic_internal_points
from .components import NearFieldComponents
from .internal import compute_internal_field
from .periodic_exterior import _initial_plane_wave_field, _resolve_periodic_channel_payload
from .slice import reshape_field_points

if TYPE_CHECKING:
    from pyceles.simulation import ChannelResult


def _local_regular_l1_fields_at_center(
    *,
    local_l1_coeffs: np.ndarray,
    n_medium: complex,
    out_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate point-local `E/H` from CELES-order local `l=1` coefficients.

    The column ordering is the CELES-mode order restricted to `l=1`:
    `tau=1,m=-1..1`, then `tau=2,m=-1..1`.

    Only the `l=1` regular sector contributes to the field exactly at the
    point-centered expansion origin; higher regular orders vanish there.
    """
    coeffs = np.asarray(local_l1_coeffs, dtype=np.complex128).reshape(-1, 6)
    n_points = int(coeffs.shape[0])
    e = np.zeros((n_points, 3), dtype=np.complex128)
    h = np.zeros((n_points, 3), dtype=np.complex128)

    a_m1 = coeffs[:, 0]
    a_0 = coeffs[:, 1]
    a_p1 = coeffs[:, 2]
    b_m1 = coeffs[:, 3]
    b_0 = coeffs[:, 4]
    b_p1 = coeffs[:, 5]

    c12 = 1.0 / np.sqrt(12.0)
    c6 = 1.0 / np.sqrt(6.0)
    medium = complex(n_medium)

    e[:, 0] = c12 * (b_m1 + b_p1)
    e[:, 1] = 1j * c12 * (b_p1 - b_m1)
    e[:, 2] = c6 * b_0

    h[:, 0] = -1j * medium * c12 * (a_m1 + a_p1)
    h[:, 1] = medium * c12 * (a_p1 - a_m1)
    h[:, 2] = -1j * medium * c6 * a_0
    return np.asarray(e, dtype=out_dtype), np.asarray(h, dtype=out_dtype)


def compute_periodic_near_field_interior(
    run: ChannelResult,
    *,
    points: np.ndarray,
    show_progress: bool = False,
) -> NearFieldComponents:
    """Evaluate periodic near fields for points inside the particle slab.

    Notes
    -----
    Points inside physical periodic particle images reuse the canonical finite-cluster
    internal-field machinery on wrapped reference-cell coordinates, with the
    corresponding Bloch phase applied to the result.
    """
    periodic = run.config.periodic
    if periodic is None:
        raise ValueError(
            "`compute_periodic_near_field` requires a periodic simulation result "
            "(`run.config.periodic` must be set)."
        )
    if not isinstance(periodic, PeriodicSpec):
        raise TypeError(
            "Periodic near-field evaluation requires `run.config.periodic` to be a PeriodicSpec."
        )

    coeffs, source = _resolve_periodic_channel_payload(run)
    k = complex(run.k)

    pts_flat, lead_shape = reshape_field_points(points)
    n_points = int(pts_flat.shape[0])
    out_dtype = np.dtype(run.accum_dtype)

    e_initial, h_initial = _initial_plane_wave_field(
        pts_flat,
        source=source,
        k=k,
        n_medium=run.config.n_medium,
    )
    e_scat = np.zeros((n_points, 3), dtype=np.complex128)
    h_scat = np.zeros((n_points, 3), dtype=np.complex128)
    e_internal = np.zeros((n_points, 3), dtype=np.complex128)
    h_internal = np.zeros((n_points, 3), dtype=np.complex128)
    inside_mask = np.zeros((n_points,), dtype=bool)

    if n_points > 0 and run.n_particles > 0:
        ax = float(periodic.lattice.ax)
        ay = float(periodic.lattice.ay)
        k_parallel = np.asarray(plane_wave_k_parallel(source), dtype=float)
        internal_classification, wrapped_points, internal_phase = classify_periodic_internal_points(
            points=pts_flat,
            particles=run.particles,
            lattice_ax=ax,
            lattice_ay=ay,
            k_parallel=k_parallel,
            n_medium=run.config.n_medium,
        )
        inside_mask = np.asarray(internal_classification.inside_any, dtype=bool)

        if np.any(inside_mask):
            e_int_ref, h_int_ref, _ = compute_internal_field(
                wrapped_points,
                coeffs,
                k=k,
                lmax=int(run.config.lmax),
                particles=run.particles,
                _point_classification=internal_classification,
                n_medium=run.config.n_medium,
                show_progress=False,
                backend=run.config.resolved_postprocessing_backend(),
                compute_dtype=np.dtype(run.config.compute_dtype),
                accum_dtype=np.dtype(run.accum_dtype),
            )
            phase = internal_phase[:, None]
            e_internal = np.asarray(phase * e_int_ref, dtype=np.complex128)
            h_internal = np.asarray(phase * h_int_ref, dtype=np.complex128)

        valid_idx = np.flatnonzero(~inside_mask)
        if valid_idx.size > 0:
            points_valid = np.asarray(pts_flat[valid_idx], dtype=float)
            local_l1 = apply_periodic_reception(
                points=points_valid,
                positions=run.positions,
                coefficients=coeffs,
                lmax_in=int(run.config.lmax),
                k=k,
                periodic=periodic,
                k_parallel=k_parallel,
                circumscribing_radii=run.circumscribing_radii,
                show_progress=show_progress,
                backend=run.config.resolved_postprocessing_backend(),
                compute_dtype=run.config.compute_dtype,
                accum_dtype=run.accum_dtype,
            )
            e_valid, h_valid = _local_regular_l1_fields_at_center(
                local_l1_coeffs=local_l1,
                n_medium=run.config.n_medium,
                out_dtype=np.dtype(np.complex128),
            )
            e_scat[valid_idx, :] = e_valid
            h_scat[valid_idx, :] = h_valid

    if np.any(inside_mask):
        e_scat[inside_mask, :] = 0.0 + 0.0j
        h_scat[inside_mask, :] = 0.0 + 0.0j
    e_total = e_initial + e_scat
    h_total = h_initial + h_scat
    if np.any(inside_mask):
        e_total[inside_mask, :] = e_internal[inside_mask, :]
        h_total[inside_mask, :] = h_internal[inside_mask, :]

    if lead_shape == ():
        vec_shape: tuple[int, ...] = (3,)
        mask_shape: tuple[int, ...] = ()
    else:
        vec_shape = (*lead_shape, 3)
        mask_shape = lead_shape
    return NearFieldComponents(
        E_initial=np.asarray(e_initial, dtype=out_dtype).reshape(vec_shape),
        H_initial=np.asarray(h_initial, dtype=out_dtype).reshape(vec_shape),
        E_scattered=np.asarray(e_scat, dtype=out_dtype).reshape(vec_shape),
        H_scattered=np.asarray(h_scat, dtype=out_dtype).reshape(vec_shape),
        E_internal=np.asarray(e_internal, dtype=out_dtype).reshape(vec_shape),
        H_internal=np.asarray(h_internal, dtype=out_dtype).reshape(vec_shape),
        E_total=np.asarray(e_total, dtype=out_dtype).reshape(vec_shape),
        H_total=np.asarray(h_total, dtype=out_dtype).reshape(vec_shape),
        inside_mask=np.asarray(inside_mask, dtype=bool).reshape(mask_shape),
    )


__all__ = ["compute_periodic_near_field_interior"]
