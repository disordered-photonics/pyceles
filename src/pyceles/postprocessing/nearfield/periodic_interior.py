from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pyceles.core.indexing import index_vswf, n_modes, n_scalar
from pyceles.core.periodic import PeriodicSpec, plane_wave_k_parallel
from pyceles.core.periodic.directsum import periodic_direct_sum_block
from pyceles.core.periodic.ewald import default_ewald_eta, periodic_ewald_block
from pyceles.core.translation import translation_ab5_table

from .components import NearFieldComponents
from .periodic_exterior import _initial_plane_wave_field, _resolve_periodic_channel_payload
from .slice import reshape_field_points

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


def _nearest_rectangular_image_distance(
    *,
    points: np.ndarray,
    center: np.ndarray,
    lattice_ax: float,
    lattice_ay: float,
) -> np.ndarray:
    """Return distance to the nearest rectangular-lattice image of one center."""
    dx = np.asarray(points[:, 0], dtype=float) - float(center[0])
    dy = np.asarray(points[:, 1], dtype=float) - float(center[1])
    dz = np.asarray(points[:, 2], dtype=float) - float(center[2])
    nx = np.rint(dx / float(lattice_ax))
    ny = np.rint(dy / float(lattice_ay))
    dx_wrap = dx - nx * float(lattice_ax)
    dy_wrap = dy - ny * float(lattice_ay)
    return np.sqrt(dx_wrap * dx_wrap + dy_wrap * dy_wrap + dz * dz)


def _inside_periodic_circumspheres(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    lattice_ax: float,
    lattice_ay: float,
    atol: float,
) -> np.ndarray:
    """Classify points that fall inside any periodic circumscribing sphere image."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return np.zeros((pts.shape[0],), dtype=bool)
    out = np.zeros((pts.shape[0],), dtype=bool)
    tol = float(max(0.0, atol))
    for j in range(pos.shape[0]):
        d = _nearest_rectangular_image_distance(
            points=pts,
            center=pos[j],
            lattice_ax=float(lattice_ax),
            lattice_ay=float(lattice_ay),
        )
        out |= d <= (float(rad[j]) + tol)
    return out


def _local_regular_fields_at_center(
    *,
    local_coeffs: np.ndarray,
    lmax: int,
    n_medium: complex,
    out_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `E/H` at local-center points from regular SVWF coefficients.

    Only the `l=1` regular sector contributes to the field at the expansion
    center. In CELES ordering this maps:
    - `tau=2, l=1, m=-1,0,1` -> electric field
    - `tau=1, l=1, m=-1,0,1` -> magnetic field
    """
    coeffs = np.asarray(local_coeffs, dtype=np.complex128).reshape(-1, n_modes(int(lmax)))
    n_points = int(coeffs.shape[0])
    e = np.zeros((n_points, 3), dtype=np.complex128)
    h = np.zeros((n_points, 3), dtype=np.complex128)
    if int(lmax) < 1:
        return np.asarray(e, dtype=out_dtype), np.asarray(h, dtype=out_dtype)

    a_m1 = coeffs[:, index_vswf(1, -1, 1, int(lmax))]
    a_0 = coeffs[:, index_vswf(1, 0, 1, int(lmax))]
    a_p1 = coeffs[:, index_vswf(1, 1, 1, int(lmax))]
    b_m1 = coeffs[:, index_vswf(1, -1, 2, int(lmax))]
    b_0 = coeffs[:, index_vswf(1, 0, 2, int(lmax))]
    b_p1 = coeffs[:, index_vswf(1, 1, 2, int(lmax))]

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
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
) -> NearFieldComponents:
    """Evaluate periodic near fields for points inside the particle slab.

    This is a local periodic evaluator: it maps solved outgoing multipoles to a
    point-centered regular SVWF expansion through periodic translation blocks,
    then evaluates the regular field at the local center.
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

    channel_key = str(channel).lower()
    if channel_key not in {"mixed", "te", "tm"}:
        raise ValueError("`channel` must be one of {'mixed', 'te', 'tm'}.")
    channel_literal: Literal["mixed", "te", "tm"] = (
        "mixed" if channel_key == "mixed" else ("te" if channel_key == "te" else "tm")
    )
    coeffs, source = _resolve_periodic_channel_payload(run, channel=channel_literal)

    pts_flat, lead_shape = reshape_field_points(points)
    n_points = int(pts_flat.shape[0])
    out_dtype = np.dtype(run.accum_dtype)

    e_initial, h_initial = _initial_plane_wave_field(
        pts_flat,
        source=source,
        k=float(run.k),
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
        sphere_atol = max(1.0e-12, 64.0 * np.finfo(np.dtype(run.accum_dtype)).eps)
        inside_mask = _inside_periodic_circumspheres(
            points=pts_flat,
            positions=run.positions,
            radii=run.circumscribing_radii,
            lattice_ax=ax,
            lattice_ay=ay,
            atol=sphere_atol,
        )

        valid_idx = np.flatnonzero(~inside_mask)
        if valid_idx.size > 0:
            lmax = int(run.config.lmax)
            nm = n_modes(lmax)
            ns = n_scalar(lmax)
            if ns <= 0 or nm <= 0:
                local_coeffs = np.zeros((valid_idx.size, nm), dtype=np.complex128)
            else:
                ab5 = translation_ab5_table(lmax, dtype=np.complex128)
                coeff_arr = np.asarray(coeffs, dtype=np.complex128).reshape(run.n_particles, nm)
                kp = plane_wave_k_parallel(source)
                local_coeffs = np.zeros((valid_idx.size, nm), dtype=np.complex128)
                method = periodic.options.method
                if method == "ewald":
                    eta = (
                        default_ewald_eta(periodic.lattice)
                        if periodic.options.eta is None
                        else float(periodic.options.eta)
                    )
                for row, idx in enumerate(valid_idx):
                    p = np.asarray(pts_flat[idx], dtype=float)
                    acc = np.zeros((nm,), dtype=np.complex128)
                    for j in range(run.n_particles):
                        if method == "directsum":
                            block = periodic_direct_sum_block(
                                lmax=lmax,
                                k=float(run.k),
                                destination=p,
                                source=np.asarray(run.positions[j], dtype=float),
                                lattice=periodic.lattice,
                                k_parallel=kp,
                                window=int(periodic.options.directsum_window),
                                ab5=ab5,
                                dtype=np.complex128,
                                exclude_zero_shift=False,
                            )
                        else:
                            block = periodic_ewald_block(
                                lmax=lmax,
                                k=float(run.k),
                                destination=p,
                                source=np.asarray(run.positions[j], dtype=float),
                                lattice=periodic.lattice,
                                k_parallel=kp,
                                eta=float(eta),
                                real_shells=int(periodic.options.real_shells),
                                reciprocal_shells=int(periodic.options.reciprocal_shells),
                                ab5=ab5,
                                dtype=np.complex128,
                                exclude_zero_shift=False,
                            )
                        acc += block @ coeff_arr[j]
                    local_coeffs[row, :] = acc
            e_valid, h_valid = _local_regular_fields_at_center(
                local_coeffs=local_coeffs,
                lmax=int(run.config.lmax),
                n_medium=run.config.n_medium,
                out_dtype=np.dtype(np.complex128),
            )
            e_scat[valid_idx, :] = e_valid
            h_scat[valid_idx, :] = h_valid

    if np.any(inside_mask):
        e_scat[inside_mask, :] = np.nan + 0.0j
        h_scat[inside_mask, :] = np.nan + 0.0j
    e_total = e_initial + e_scat
    h_total = h_initial + h_scat

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
