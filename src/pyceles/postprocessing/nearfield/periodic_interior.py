from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from pyceles.core.indexing import n_modes
from pyceles.core.periodic import PeriodicSpec, plane_wave_k_parallel
from pyceles.core.periodic.ewald import (
    EwaldShellWorkspace,
    ewald_structural_sums_2d_batch,
    resolve_ewald_eta,
)
from pyceles.core.periodic.rayleigh import (
    apply_rayleigh_far_to_points_numpy,
    near_point_source_csr,
    prepare_rayleigh_plan,
)

from .classification import classify_periodic_internal_points
from .components import NearFieldComponents
from .internal import compute_internal_field
from .periodic_exterior import _initial_plane_wave_field, _resolve_periodic_channel_payload
from .periodic_projection import (
    l1_compact_projection_data,
    reduce_compact_structural_sums_to_l1,
)
from .slice import reshape_field_points

if TYPE_CHECKING:
    from pyceles.simulation import ChannelResult


def _periodic_local_regular_l1_coeffs(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    periodic: PeriodicSpec,
    k_parallel: np.ndarray,
    circumscribing_radii: np.ndarray | None = None,
    point_batch_size: int = 128,
    show_progress: bool = False,
) -> np.ndarray:
    """Return point-local regular `l=1` coefficients for periodic in-slab points.

    The Ewald path contracts every source-point structural sum directly into the
    destination ``l=1`` sector. The hybrid Rayleigh path uses the same exact
    contraction inside its vertical band and z-sorted reciprocal scans outside.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    lmax_i = int(lmax)
    nm = n_modes(lmax_i)
    out = np.zeros((pts.shape[0], 6), dtype=np.complex128)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return out

    coeff_arr = np.asarray(coeffs, dtype=np.complex128).reshape(pos.shape[0], nm)
    (
        lmax_struct,
        structural_order,
        degree_indices,
        order_indices,
        kernel,
    ) = l1_compact_projection_data(lmax_i)
    method = str(periodic.options.method)
    if method not in {"ewald", "rayleigh"}:
        raise NotImplementedError(
            "Periodic in-slab near-field evaluation requires "
            "`periodic.options.method` to be 'ewald' or 'rayleigh'."
        )

    rayleigh_z_cut: float | None = None
    if method == "rayleigh":
        plan = prepare_rayleigh_plan(
            lmax=lmax_i,
            k=float(k),
            positions=pos,
            circumscribing_radii=circumscribing_radii,
            periodic=periodic,
            k_parallel=k_parallel,
            dtype=np.complex128,
        )
        out += np.asarray(
            apply_rayleigh_far_to_points_numpy(plan, coeff_arr, pts),
            dtype=np.complex128,
        )
        rayleigh_z_cut = float(plan.z_cut)

    batch = max(1, int(point_batch_size))
    eta = resolve_ewald_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=k_parallel,
        positions=pos,
        lmax=lmax_i,
        max_vertical_offset=rayleigh_z_cut,
    )
    workspace = EwaldShellWorkspace(
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        eta=float(eta),
    )
    if method == "ewald":
        n_batches = (pts.shape[0] + batch - 1) // batch
        total_work = int(n_batches * pos.shape[0])
        progress = (
            tqdm(total=total_work, desc="Periodic slab field", unit="source-batch", leave=True)
            if show_progress and total_work > 0
            else None
        )
        try:
            for start in range(0, pts.shape[0], batch):
                stop = min(pts.shape[0], start + batch)
                pts_batch = np.asarray(pts[start:stop], dtype=float)
                acc = np.zeros((pts_batch.shape[0], 6), dtype=np.complex128)
                for j in range(pos.shape[0]):
                    sums = ewald_structural_sums_2d_batch(
                        lmax_struct=lmax_struct,
                        structural_order=structural_order,
                        k=float(k),
                        destinations=pts_batch,
                        source=pos[j],
                        lattice=periodic.lattice,
                        k_parallel=k_parallel,
                        eta=float(eta),
                        real_shells=periodic.options.real_shells,
                        reciprocal_shells=periodic.options.reciprocal_shells,
                        shell_tolerance=float(periodic.options.shell_tolerance),
                        max_shells=int(periodic.options.max_shells),
                        workspace=workspace,
                    )
                    acc += reduce_compact_structural_sums_to_l1(
                        sums,
                        coeff_arr[j],
                        degree_indices=degree_indices,
                        order_indices=order_indices,
                        kernel=kernel,
                    )
                    if progress is not None:
                        progress.update(1)
                out[start:stop] = acc
        finally:
            if progress is not None:
                progress.close()
        return out

    if rayleigh_z_cut is None:
        raise RuntimeError("Rayleigh near-field initialization failed.")
    indptr, destination_indices, _source_indices = near_point_source_csr(
        pts,
        pos,
        rayleigh_z_cut,
    )
    if destination_indices.size == 0:
        return out
    total_work = sum(
        (int(indptr[j + 1] - indptr[j]) + batch - 1) // batch for j in range(pos.shape[0])
    )
    progress = (
        tqdm(total=total_work, desc="Periodic slab field", unit="source-batch", leave=True)
        if show_progress and total_work > 0
        else None
    )
    try:
        for j in range(pos.shape[0]):
            point_indices = destination_indices[indptr[j] : indptr[j + 1]]
            for start in range(0, point_indices.size, batch):
                batch_indices = point_indices[start : start + batch]
                pts_batch = np.asarray(pts[batch_indices], dtype=float)
                sums = ewald_structural_sums_2d_batch(
                    lmax_struct=lmax_struct,
                    structural_order=structural_order,
                    k=float(k),
                    destinations=pts_batch,
                    source=pos[j],
                    lattice=periodic.lattice,
                    k_parallel=k_parallel,
                    eta=float(eta),
                    real_shells=periodic.options.real_shells,
                    reciprocal_shells=periodic.options.reciprocal_shells,
                    shell_tolerance=float(periodic.options.shell_tolerance),
                    max_shells=int(periodic.options.max_shells),
                    workspace=workspace,
                )
                out[batch_indices] += reduce_compact_structural_sums_to_l1(
                    sums,
                    coeff_arr[j],
                    degree_indices=degree_indices,
                    order_indices=order_indices,
                    kernel=kernel,
                )
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return out


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
                k=float(run.k),
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
            if run.config.resolved_postprocessing_backend() == "cupy":
                from .periodic_interior_cupy import periodic_local_regular_l1_coeffs_cupy

                local_l1 = periodic_local_regular_l1_coeffs_cupy(
                    points=points_valid,
                    positions=run.positions,
                    coeffs=coeffs,
                    lmax=int(run.config.lmax),
                    k=float(run.k),
                    periodic=periodic,
                    k_parallel=k_parallel,
                    circumscribing_radii=run.circumscribing_radii,
                    show_progress=show_progress,
                    compute_dtype=np.dtype(run.config.compute_dtype),
                    accum_dtype=np.dtype(run.accum_dtype),
                )
            else:
                local_l1 = _periodic_local_regular_l1_coeffs(
                    points=points_valid,
                    positions=run.positions,
                    coeffs=coeffs,
                    lmax=int(run.config.lmax),
                    k=float(run.k),
                    periodic=periodic,
                    k_parallel=k_parallel,
                    circumscribing_radii=run.circumscribing_radii,
                    show_progress=show_progress,
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
