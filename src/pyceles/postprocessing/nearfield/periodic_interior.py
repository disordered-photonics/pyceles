from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

import numpy as np

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.particles import (
    LayeredSphere,
    Sphere,
    Spheroid,
    particle_contains_points,
)
from pyceles.core.periodic import PeriodicSpec, plane_wave_k_parallel
from pyceles.core.periodic.ewald import (
    EwaldShellWorkspace,
    ewald_structural_sums_2d_batch,
    select_ewald_eta,
)
from pyceles.core.translation import translation_ab5_table

from .classification import InternalPointClassification
from .components import NearFieldComponents
from .internal import compute_internal_field
from .periodic_exterior import _initial_plane_wave_field, _resolve_periodic_channel_payload
from .slice import reshape_field_points

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


def _wrap_points_to_nearest_rectangular_image(
    *,
    points: np.ndarray,
    center: np.ndarray,
    lattice_ax: float,
    lattice_ay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrap points to the nearest rectangular-lattice image of one center."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    ctr = np.asarray(center, dtype=float).reshape(3)
    dx = pts[:, 0] - float(ctr[0])
    dy = pts[:, 1] - float(ctr[1])
    nx = np.rint(dx / float(lattice_ax)).astype(np.int64, copy=False)
    ny = np.rint(dy / float(lattice_ay)).astype(np.int64, copy=False)
    wrapped = np.asarray(pts, dtype=float).copy()
    wrapped[:, 0] -= nx.astype(float) * float(lattice_ax)
    wrapped[:, 1] -= ny.astype(float) * float(lattice_ay)
    return wrapped, nx, ny


def _classify_periodic_internal_particle_points(
    *,
    points: np.ndarray,
    particles,
    lattice_ax: float,
    lattice_ay: float,
    k_parallel: np.ndarray,
    n_medium: complex,
) -> tuple[InternalPointClassification, np.ndarray, np.ndarray]:
    """Classify points inside periodic particle images and wrap them to the reference cell.

    Returns
    -------
    classification, wrapped_points, bloch_phase
        `classification` uses reference-cell particle ownership, `wrapped_points`
        stores the corresponding image-wrapped coordinates for owned points, and
        `bloch_phase` contains the phase factor of the selected image.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n_points = int(pts.shape[0])
    wrapped_points = np.asarray(pts, dtype=float).copy()
    bloch_phase = np.ones((n_points,), dtype=np.complex128)
    inside_any = np.zeros((n_points,), dtype=bool)
    point_indices_by_particle: list[np.ndarray] = []
    kpar = np.asarray(k_parallel, dtype=float).reshape(2)

    for _particle in particles:
        point_indices_by_particle.append(np.empty((0,), dtype=np.intp))

    n_medium_c = complex(n_medium)

    for j, particle in enumerate(particles):
        if (
            (isinstance(particle, Sphere) and complex(particle.refractive_index) == n_medium_c)
            or (isinstance(particle, Spheroid) and complex(particle.refractive_index) == n_medium_c)
            or (
                isinstance(particle, LayeredSphere)
                and all(
                    complex(n_layer) == n_medium_c for n_layer in particle.layer_refractive_indices
                )
            )
        ):
            continue
        remaining = np.flatnonzero(~inside_any)
        if remaining.size == 0:
            break
        wrapped_remain, nx, ny = _wrap_points_to_nearest_rectangular_image(
            points=pts[remaining],
            center=np.asarray(particle.position, dtype=float),
            lattice_ax=float(lattice_ax),
            lattice_ay=float(lattice_ay),
        )
        mask = np.asarray(particle_contains_points(particle, wrapped_remain), dtype=bool)
        if not np.any(mask):
            continue
        owned = remaining[mask]
        point_indices_by_particle[j] = owned.astype(np.intp, copy=False)
        inside_any[owned] = True
        wrapped_points[owned] = wrapped_remain[mask]
        phase_arg = kpar[0] * nx[mask].astype(float) * float(lattice_ax) + kpar[1] * ny[
            mask
        ].astype(float) * float(lattice_ay)
        bloch_phase[owned] = np.exp(1j * phase_arg)

    return (
        InternalPointClassification(
            inside_any=inside_any,
            point_indices_by_particle=tuple(point_indices_by_particle),
        ),
        wrapped_points,
        bloch_phase,
    )


@cache
def _l1_projection_data(lmax: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Precompute the minimal contraction tensor for local `l=1` fields.

    Returns
    -------
    lmax_struct, m_offset, kernel, row_idx
        `kernel` has shape `(6, nm, 2*order+1, p_count)` and contracts the
        structural scalar table directly into the destination `l=1` sector,
        avoiding construction of the full `(nm x nm)` translation block.
    """
    lmax_i = int(lmax)
    if lmax_i < 1:
        raise ValueError(f"`lmax` must be >= 1. Got {lmax!r}.")
    nm = n_modes(lmax_i)
    row_idx: list[int] = []
    m_dst: list[int] = []
    m_src = np.zeros((nm,), dtype=np.int32)
    for _tau, l, m, idx in iter_modes(lmax_i):
        m_src[idx] = int(m)
        if l == 1:
            row_idx.append(int(idx))
            m_dst.append(int(m))
    row_idx_arr = np.asarray(row_idx, dtype=np.int64)
    m_dst_arr = np.asarray(m_dst, dtype=np.int32)

    ab5 = np.asarray(translation_ab5_table(lmax_i, dtype=np.complex128), dtype=np.complex128)
    ab5_l1 = np.asarray(ab5[row_idx_arr, :, :], dtype=np.complex128)

    max_degree = int(lmax_i + 1)  # p_max = l_dst + l_src with l_dst = 1
    lmax_struct = int((max_degree + 1) // 2)  # ensure 2*lmax_struct >= max_degree
    order = 2 * lmax_struct
    p_count = max_degree + 1
    m_offset = order
    kernel = np.zeros((ab5_l1.shape[0], nm, 2 * order + 1, p_count), dtype=np.complex128)
    for row in range(ab5_l1.shape[0]):
        dm_idx = m_src - int(m_dst_arr[row]) + m_offset
        for col in range(nm):
            kernel[row, col, int(dm_idx[col]), :p_count] = ab5_l1[row, col, :p_count]
    return lmax_struct, m_offset, kernel, row_idx_arr


def _reduce_structural_sums_to_l1(
    structural_sums: np.ndarray,
    coeffs: np.ndarray,
    *,
    kernel: np.ndarray,
) -> np.ndarray:
    """Contract batched structural sums directly into local `l=1` coefficients."""
    sums = np.asarray(structural_sums, dtype=np.complex128)
    src_coeffs = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
    p_count = int(kernel.shape[3])
    # sums: (n_points, order+1, 2*order+1), kernel: (6, nm, 2*order+1, p_count)
    # Only `p<=l_dst+l_src` contributes; slice the structural table accordingly.
    return np.asarray(
        np.einsum("npm,rcmp,c->nr", sums[:, :p_count, :], kernel, src_coeffs, optimize=True)
    )


def _periodic_local_regular_l1_coeffs(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    periodic: PeriodicSpec,
    k_parallel: np.ndarray,
    point_batch_size: int = 128,
) -> np.ndarray:
    """Return point-local regular `l=1` coefficients for periodic in-slab points.

    The hot path contracts periodic structural sums directly into the destination
    `l=1` sector.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    lmax_i = int(lmax)
    nm = n_modes(lmax_i)
    out = np.zeros((pts.shape[0], 6), dtype=np.complex128)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return out

    coeff_arr = np.asarray(coeffs, dtype=np.complex128).reshape(pos.shape[0], nm)
    lmax_struct, _m_offset, kernel, _row_idx = _l1_projection_data(lmax_i)
    method = str(periodic.options.method)
    if method != "ewald":
        raise NotImplementedError(
            "Periodic in-slab near-field evaluation currently requires "
            "`periodic.options.method='ewald'`."
        )

    batch = max(1, int(point_batch_size))
    eta = (
        select_ewald_eta(
            lattice=periodic.lattice,
            k=float(k),
            k_parallel=k_parallel,
            positions=pos,
            lmax=lmax_i,
            shell_tolerance=float(periodic.options.shell_tolerance),
            max_shells=int(periodic.options.max_shells),
            real_shells=periodic.options.real_shells,
            reciprocal_shells=periodic.options.reciprocal_shells,
        )
        if periodic.options.eta is None
        else float(periodic.options.eta)
    )
    workspace = EwaldShellWorkspace(
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        eta=float(eta),
    )
    for s in range(0, pts.shape[0], batch):
        e = min(pts.shape[0], s + batch)
        pts_batch = np.asarray(pts[s:e], dtype=float)
        acc = np.zeros((pts_batch.shape[0], 6), dtype=np.complex128)
        for j in range(pos.shape[0]):
            sums = ewald_structural_sums_2d_batch(
                lmax_struct=lmax_struct,
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
            acc += _reduce_structural_sums_to_l1(sums, coeff_arr[j], kernel=kernel)
        out[s:e, :] = acc
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
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
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
        k_parallel = np.asarray(plane_wave_k_parallel(source), dtype=float)
        internal_classification, wrapped_points, internal_phase = (
            _classify_periodic_internal_particle_points(
                points=pts_flat,
                particles=run.particles,
                lattice_ax=ax,
                lattice_ay=ay,
                k_parallel=k_parallel,
                n_medium=run.config.n_medium,
            )
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
            local_l1 = _periodic_local_regular_l1_coeffs(
                points=np.asarray(pts_flat[valid_idx], dtype=float),
                positions=run.positions,
                coeffs=coeffs,
                lmax=int(run.config.lmax),
                k=float(run.k),
                periodic=periodic,
                k_parallel=k_parallel,
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
