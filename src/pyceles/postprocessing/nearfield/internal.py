from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import spherical_jn, spherical_yn
from tqdm.auto import tqdm

from pyceles._cupy_memory import cupy_allocator_snapshot
from pyceles._optional import asnumpy, import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.particles import (
    LayeredSphere,
    Particle,
    ParticleCollection,
    PECSphere,
    Sphere,
    Spheroid,
    TMatrixParticle,
    particle_contains_points,
)
from pyceles.core.spherical import spherical_functions_trigon
from pyceles.core.tmatrix import (
    _spheroid_internal_block,
    layered_internal_ab_ratios,
    sphere_internal_ratios,
)

from .classification import InternalPointClassification
from .common import build_internal_mode_tensors, contract_modes, mode_indices_by_l


def compute_internal_field(
    field_points: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    *,
    particles: Sequence[Particle],
    _point_classification: InternalPointClassification | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the physical total field inside particles."""
    return _compute_internal_field_particles(
        field_points,
        particles=particles,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        classification=_point_classification,
        n_medium=n_medium,
        show_progress=show_progress,
        backend=backend,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )


def _compute_internal_field_homogeneous_spheres(
    field_points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_particle: np.ndarray,
    classification: InternalPointClassification | None = None,
    particle_indices: np.ndarray | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical homogeneous-sphere internal-field kernel."""
    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    n_spheres = pos.shape[0]
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)

    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != n_spheres:
        raise ValueError(f"radii must have length Ns={n_spheres}, got {rad.shape}")
    if particle_indices is not None and np.asarray(particle_indices).size != n_spheres:
        raise ValueError(
            "`particle_indices` length must match number of spheres "
            f"({n_spheres}). Got {np.asarray(particle_indices).size}."
        )

    if np.ndim(n_particle) == 0:
        n_particle_arr = np.full(n_spheres, complex(n_particle), dtype=complex)
    else:
        n_particle_arr = np.asarray(n_particle, dtype=complex).reshape(-1)
        if n_particle_arr.shape[0] != n_spheres:
            raise ValueError(
                f"n_particle must have length Ns={n_spheres}, got {n_particle_arr.shape}"
            )

    n_medium_c = complex(n_medium)
    if str(backend).lower() == "cupy":
        return _compute_internal_field_homogeneous_spheres_cupy(
            pts,
            pos,
            rad,
            np.asarray(coeffs, dtype=compute_dtype),
            k=k,
            lmax=lmax,
            n_particle=n_particle_arr,
            classification=classification,
            particle_indices=particle_indices,
            n_medium=n_medium_c,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )

    return _compute_internal_field_homogeneous_spheres_numpy_batched(
        pts,
        pos,
        rad,
        np.asarray(coeffs, dtype=compute_dtype),
        k=k,
        lmax=lmax,
        n_particle=n_particle_arr,
        classification=classification,
        particle_indices=particle_indices,
        n_medium=n_medium_c,
        show_progress=show_progress,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )


_INTERNAL_PAIR_BATCH_SIZE = 65_536
_INTERNAL_CUPY_WORKSPACE_HEADROOM_FRACTION = 0.5


def _internal_cupy_workspace_bytes_per_pair(
    *,
    lmax: int,
    compute_dtype: np.dtype,
) -> int:
    """Conservatively estimate CuPy allocator footprint per radial pair."""
    lmax_i = max(1, int(lmax))
    complex_dtype = np.dtype(compute_dtype)
    real_itemsize = np.dtype(
        np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
    ).itemsize
    complex_itemsize = complex_dtype.itemsize

    # pi/tau/P grids plus geometry/angular scratch.
    angular_real_values = 4 * (lmax_i + 1) ** 2
    # Cover the layered path (two radial bases), mode-tensor temporaries,
    # einsum inputs/outputs, and CuPy pool retention across degrees. This is
    # intentionally a safe allocator-footprint estimate, not a live tensor
    # byte count.
    mode_complex_values = 28 * lmax_i * (lmax_i + 2)
    fixed_real_values = 24
    fixed_complex_values = 24
    return int(
        (angular_real_values + fixed_real_values) * real_itemsize
        + (mode_complex_values + fixed_complex_values) * complex_itemsize
    )


def _internal_pair_batch_size_for_workspace(
    *,
    total_pairs: int,
    lmax: int,
    compute_dtype: np.dtype,
    workspace_bytes: int,
) -> int:
    """Resolve a nonzero radial-pair batch within one workspace budget."""
    total = max(1, int(total_pairs))
    fast_cap = min(total, _INTERNAL_PAIR_BATCH_SIZE)
    bytes_per_pair = _internal_cupy_workspace_bytes_per_pair(
        lmax=int(lmax), compute_dtype=np.dtype(compute_dtype)
    )
    memory_cap = max(1, int(workspace_bytes) // max(1, int(bytes_per_pair)))
    return max(1, min(fast_cap, memory_cap))


def _cupy_internal_pair_batch_size(
    *,
    cupy: object,
    total_pairs: int,
    lmax: int,
    compute_dtype: np.dtype,
) -> int:
    """Choose a scalable CuPy pair batch from current guarded headroom.

    Keep half of usable device headroom outside this postprocessing workspace
    for the solver state, allocator fragmentation, and other temporaries. The
    fixed 65,536-pair path remains the maximum and therefore stays intact
    whenever the estimated workspace fits.
    """
    snapshot = cupy_allocator_snapshot(cupy, apply_pool_limit=False)
    reusable_or_fresh = int(snapshot.raw_free_bytes) + int(snapshot.pool_free_bytes)
    usable_headroom = max(
        1,
        min(int(snapshot.active_headroom_bytes), int(reusable_or_fresh)),
    )
    workspace_bytes = max(
        1,
        int(float(usable_headroom) * _INTERNAL_CUPY_WORKSPACE_HEADROOM_FRACTION),
    )
    return _internal_pair_batch_size_for_workspace(
        total_pairs=int(total_pairs),
        lmax=int(lmax),
        compute_dtype=np.dtype(compute_dtype),
        workspace_bytes=int(workspace_bytes),
    )


def _radial_internal_point_pairs(
    field_points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    classification: InternalPointClassification | None,
    particle_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat ``(point, local-sphere)`` ownership pairs."""
    n_points = int(field_points.shape[0])
    n_spheres = int(positions.shape[0])
    if n_points == 0 or n_spheres == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty.copy()

    if classification is not None:
        counts = np.diff(classification.point_offsets).astype(np.int64, copy=False)
        if counts.size != classification.active_particle_indices.size:
            raise ValueError("Invalid internal-point classification offsets.")
        active_global = classification.active_particle_indices.astype(np.int64, copy=False)
        if particle_indices is None:
            if classification.n_particles != n_spheres:
                raise ValueError(
                    "Classification particle count must match the homogeneous-sphere collection."
                )
            return (
                classification.point_indices.astype(np.int64, copy=False),
                np.repeat(active_global, counts),
            )

        selected = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        is_sorted = selected.size < 2 or bool(np.all(selected[1:] > selected[:-1]))
        if is_sorted:
            order = None
            selected_sorted = selected
        else:
            order = np.argsort(selected, kind="stable")
            selected_sorted = selected[order]
        sorted_positions = np.searchsorted(selected_sorted, active_global)
        valid_entry = sorted_positions < selected_sorted.size
        if np.any(valid_entry):
            valid_positions = np.flatnonzero(valid_entry)
            valid_entry[valid_positions] &= (
                selected_sorted[sorted_positions[valid_positions]] == active_global[valid_positions]
            )
        if not np.any(valid_entry):
            empty = np.zeros((0,), dtype=np.int64)
            return empty, empty.copy()
        local_by_entry = np.full(active_global.shape, -1, dtype=np.int64)
        if order is None:
            local_by_entry[valid_entry] = sorted_positions[valid_entry]
        else:
            local_by_entry[valid_entry] = order[sorted_positions[valid_entry]]
        entry_ids = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        pair_mask = valid_entry[entry_ids]
        return (
            classification.point_indices[pair_mask].astype(np.int64, copy=False),
            local_by_entry[entry_ids[pair_mask]].astype(np.int64, copy=False),
        )

    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    candidate_lists = tree.query_ball_point(
        field_points,
        r=float(np.max(radii)),
        return_sorted=True,
    )
    point_chunks: list[np.ndarray] = []
    sphere_chunks: list[np.ndarray] = []
    for point_index, candidates_raw in enumerate(candidate_lists):
        candidates = np.asarray(candidates_raw, dtype=np.int64)
        if candidates.size == 0:
            continue
        delta = field_points[point_index] - positions[candidates]
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        candidates = candidates[distance_squared < radii[candidates] ** 2]
        if candidates.size == 0:
            continue
        point_chunks.append(np.full(candidates.size, point_index, dtype=np.int64))
        sphere_chunks.append(candidates)
    if not point_chunks:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty.copy()
    return np.concatenate(point_chunks), np.concatenate(sphere_chunks)


def _sphere_internal_material_tables(
    *,
    pair_spheres: np.ndarray,
    radii: np.ndarray,
    n_particle: np.ndarray,
    lmax: int,
    k: float,
    n_medium: complex,
    compute_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build material-ratio tables for the active sphere-point pairs."""
    active_spheres = np.unique(pair_spheres)
    parameters = np.column_stack(
        (
            radii[active_spheres],
            n_particle[active_spheres].real,
            n_particle[active_spheres].imag,
        )
    )
    unique_parameters, active_group = np.unique(parameters, axis=0, return_inverse=True)
    active_position = np.searchsorted(active_spheres, pair_spheres)
    pair_groups = active_group[active_position].astype(np.int32, copy=False)

    ratio_m = np.zeros((unique_parameters.shape[0], lmax + 1), dtype=compute_dtype)
    ratio_n = np.zeros_like(ratio_m)
    group_refractive_indices = np.empty((unique_parameters.shape[0],), dtype=np.complex128)
    for group_index, row in enumerate(unique_parameters):
        radius = float(row[0])
        index = complex(float(row[1]), float(row[2]))
        ratios = sphere_internal_ratios(lmax, k, radius, index, n_medium)
        ratio_m[group_index] = np.asarray(ratios[1], dtype=compute_dtype)
        ratio_n[group_index] = np.asarray(ratios[2], dtype=compute_dtype)
        group_refractive_indices[group_index] = index
    return pair_groups, ratio_m, ratio_n, group_refractive_indices


def _compute_internal_field_homogeneous_spheres_numpy_batched(
    field_points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_particle: np.ndarray,
    classification: InternalPointClassification | None,
    particle_indices: np.ndarray | None,
    n_medium: complex,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute homogeneous-sphere internal fields with batched NumPy work."""
    real_dtype = np.dtype(np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64)
    n_points = int(field_points.shape[0])
    pair_points, pair_spheres = _radial_internal_point_pairs(
        field_points,
        positions,
        radii,
        classification=classification,
        particle_indices=particle_indices,
    )
    inside = np.zeros((n_points,), dtype=bool)
    if pair_points.size == 0:
        empty = np.zeros((n_points, 3), dtype=accum_dtype)
        return empty, empty.copy(), inside

    unique_pair_points = np.unique(pair_points)
    inside[unique_pair_points] = True
    (
        pair_groups,
        ratio_m_host,
        ratio_n_host,
        group_refractive_indices,
    ) = _sphere_internal_material_tables(
        pair_spheres=pair_spheres,
        radii=radii,
        n_particle=n_particle,
        lmax=lmax,
        k=k,
        n_medium=n_medium,
        compute_dtype=compute_dtype,
    )
    coeffs_host = np.asarray(coeffs, dtype=compute_dtype)
    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    has_overlap = unique_pair_points.size != pair_points.size
    mode_by_l = mode_indices_by_l(lmax)
    progress = (
        tqdm(total=pair_points.size, desc="Internal field (sphere-point pairs)", unit="pair")
        if show_progress
        else None
    )

    for start in range(0, pair_points.size, _INTERNAL_PAIR_BATCH_SIZE):
        stop = min(pair_points.size, start + _INTERNAL_PAIR_BATCH_SIZE)
        point_batch = pair_points[start:stop]
        sphere_batch = pair_spheres[start:stop]
        group_batch = pair_groups[start:stop]

        rvec_full = field_points[point_batch] - positions[sphere_batch]
        r = np.linalg.norm(rvec_full, axis=1)
        r_safe = np.where(r < 1.0e-12, 1.0e-12, r)
        rvec = np.asarray(rvec_full, dtype=real_dtype)
        r_safe_angular = np.asarray(r_safe, dtype=real_dtype)
        k_group = k * (group_refractive_indices[group_batch] / n_medium)
        kr = np.asarray(k_group * r_safe, dtype=compute_dtype)
        x = rvec[:, 0]
        y = rvec[:, 1]
        z = rvec[:, 2]
        rho = np.sqrt(x * x + y * y)
        ct = z / r_safe_angular
        st = rho / r_safe_angular
        phi = np.arctan2(y, x)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        e_r = np.stack([st * cos_phi, st * sin_phi, ct], axis=1)
        e_theta = np.stack([ct * cos_phi, ct * sin_phi, -st], axis=1)
        e_phi = np.stack([-sin_phi, cos_phi, np.zeros_like(phi)], axis=1)
        pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)
        e_batch = np.zeros((stop - start, 3), dtype=accum_dtype)
        h_batch = np.zeros_like(e_batch)
        h_factor = np.asarray(-1j * group_refractive_indices[group_batch], dtype=compute_dtype)

        for l in range(1, lmax + 1):
            z_l = np.asarray(spherical_jn(l, kr), dtype=compute_dtype)
            dz_l = np.asarray(spherical_jn(l, kr, derivative=True), dtype=compute_dtype)
            dxxz = np.asarray(z_l + kr * dz_l, dtype=compute_dtype)
            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            m_all, n_all = build_internal_mode_tensors(
                l=l,
                m_vals=m_vals,
                abs_m=abs_m,
                phi=phi,
                e_r=e_r,
                e_theta=e_theta,
                e_phi=e_phi,
                pi_all=pi_all,
                tau_all=tau_all,
                p_all=p_all,
                z_l=z_l,
                dxxz=dxxz,
                kr=kr,
                compute_dtype=compute_dtype,
            )
            a_int = (
                coeffs_host[sphere_batch[:, None], n1_idx] * ratio_m_host[group_batch, l][:, None]
            )
            b_int = (
                coeffs_host[sphere_batch[:, None], n2_idx] * ratio_n_host[group_batch, l][:, None]
            )
            e_batch += np.einsum("bm,bmc->bc", a_int, m_all)
            e_batch += np.einsum("bm,bmc->bc", b_int, n_all)
            h_batch += h_factor[:, None] * np.einsum("bm,bmc->bc", a_int, n_all)
            h_batch += h_factor[:, None] * np.einsum("bm,bmc->bc", b_int, m_all)

        if has_overlap:
            np.add.at(e, point_batch, e_batch)
            np.add.at(h, point_batch, h_batch)
        else:
            e[point_batch] = e_batch
            h[point_batch] = h_batch
        if progress is not None:
            progress.update(stop - start)

    if progress is not None:
        progress.close()
    return e, h, inside


def _compute_internal_field_homogeneous_spheres_cupy(
    field_points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_particle: np.ndarray,
    classification: InternalPointClassification | None,
    particle_indices: np.ndarray | None,
    n_medium: complex,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute homogeneous-sphere internal fields with batched CuPy work."""
    cupy, _ = import_cupy()
    compute_dtype_cp = (
        cupy.complex64 if compute_dtype == np.dtype(np.complex64) else cupy.complex128
    )
    accum_dtype_cp = cupy.complex64 if accum_dtype == np.dtype(np.complex64) else cupy.complex128
    real_dtype = np.dtype(np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64)
    real_dtype_cp = cupy.float32 if real_dtype == np.dtype(np.float32) else cupy.float64

    n_points = int(field_points.shape[0])
    pair_points, pair_spheres = _radial_internal_point_pairs(
        field_points,
        positions,
        radii,
        classification=classification,
        particle_indices=particle_indices,
    )
    inside = np.zeros((n_points,), dtype=bool)
    if pair_points.size == 0:
        empty = np.zeros((n_points, 3), dtype=accum_dtype)
        return empty, empty.copy(), inside
    unique_pair_points = np.unique(pair_points)
    inside[unique_pair_points] = True

    (
        pair_groups,
        ratio_m_host,
        ratio_n_host,
        group_refractive_indices,
    ) = _sphere_internal_material_tables(
        pair_spheres=pair_spheres,
        radii=radii,
        n_particle=n_particle,
        lmax=lmax,
        k=k,
        n_medium=n_medium,
        compute_dtype=compute_dtype,
    )
    coeffs_host = np.asarray(coeffs, dtype=compute_dtype)
    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    has_overlap = unique_pair_points.size != pair_points.size
    mode_by_l = mode_indices_by_l(lmax)
    pair_batch_size = _cupy_internal_pair_batch_size(
        cupy=cupy,
        total_pairs=int(pair_points.size),
        lmax=lmax,
        compute_dtype=compute_dtype,
    )
    progress = (
        tqdm(total=pair_points.size, desc="Internal field (sphere-point pairs)", unit="pair")
        if show_progress
        else None
    )

    for start in range(0, pair_points.size, pair_batch_size):
        stop = min(pair_points.size, start + pair_batch_size)
        point_batch = pair_points[start:stop]
        sphere_batch = pair_spheres[start:stop]
        group_batch = pair_groups[start:stop]

        rvec_full = field_points[point_batch] - positions[sphere_batch]
        r_host = np.linalg.norm(rvec_full, axis=1)
        r_safe_host = np.where(r_host < 1.0e-12, 1.0e-12, r_host)
        rvec_host = np.asarray(rvec_full, dtype=real_dtype)
        k_group = k * (group_refractive_indices[group_batch] / n_medium)
        kr_host = np.asarray(k_group * r_safe_host, dtype=compute_dtype)

        rvec = cupy.asarray(rvec_host, dtype=real_dtype_cp)
        r_safe = cupy.asarray(r_safe_host, dtype=real_dtype_cp)
        kr = cupy.asarray(kr_host, dtype=compute_dtype_cp)
        x = rvec[:, 0]
        y = rvec[:, 1]
        z = rvec[:, 2]
        rho = cupy.sqrt(x * x + y * y)
        ct = z / r_safe
        st = rho / r_safe
        phi = cupy.arctan2(y, x)
        cos_phi = cupy.cos(phi)
        sin_phi = cupy.sin(phi)
        e_r = cupy.stack([st * cos_phi, st * sin_phi, ct], axis=1)
        e_theta = cupy.stack([ct * cos_phi, ct * sin_phi, -st], axis=1)
        e_phi = cupy.stack([-sin_phi, cos_phi, cupy.zeros_like(phi)], axis=1)
        pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=cupy, return_plm=True)

        e_batch = cupy.zeros((stop - start, 3), dtype=accum_dtype_cp)
        h_batch = cupy.zeros_like(e_batch)
        n_s_batch = cupy.asarray(group_refractive_indices[group_batch], dtype=compute_dtype_cp)
        minus_i = cupy.asarray(-1j, dtype=compute_dtype_cp)
        h_factor = minus_i * n_s_batch
        for l in range(1, lmax + 1):
            z_l = cupy.asarray(
                np.asarray(spherical_jn(l, kr_host), dtype=compute_dtype),
                dtype=compute_dtype_cp,
            )
            dz_l = cupy.asarray(
                np.asarray(spherical_jn(l, kr_host, derivative=True), dtype=compute_dtype),
                dtype=compute_dtype_cp,
            )
            dxxz = z_l + kr * dz_l
            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            m_all, n_all = build_internal_mode_tensors(
                l=l,
                m_vals=m_vals,
                abs_m=abs_m,
                phi=phi,
                e_r=e_r,
                e_theta=e_theta,
                e_phi=e_phi,
                pi_all=pi_all,
                tau_all=tau_all,
                p_all=p_all,
                z_l=z_l,
                dxxz=dxxz,
                kr=kr,
                compute_dtype=compute_dtype,
            )
            m_all = m_all.astype(compute_dtype_cp, copy=False)
            n_all = n_all.astype(compute_dtype_cp, copy=False)

            # Keep sparse coefficient/material gathers on the host. Some CuPy
            # stacks can stall on the corresponding tiny advanced-index
            # kernels; the dense mode algebra and einsum contractions are the
            # GPU work.
            a_int = cupy.asarray(
                coeffs_host[sphere_batch[:, None], n1_idx] * ratio_m_host[group_batch, l][:, None],
                dtype=compute_dtype_cp,
            )
            b_int = cupy.asarray(
                coeffs_host[sphere_batch[:, None], n2_idx] * ratio_n_host[group_batch, l][:, None],
                dtype=compute_dtype_cp,
            )
            am = cupy.einsum("bm,bmc->bc", a_int, m_all)
            bn = cupy.einsum("bm,bmc->bc", b_int, n_all)
            an = cupy.einsum("bm,bmc->bc", a_int, n_all)
            bm = cupy.einsum("bm,bmc->bc", b_int, m_all)
            e_batch += am.astype(accum_dtype_cp, copy=False)
            e_batch += bn.astype(accum_dtype_cp, copy=False)
            h_batch += (h_factor[:, None] * an).astype(accum_dtype_cp, copy=False)
            h_batch += (h_factor[:, None] * bm).astype(accum_dtype_cp, copy=False)

        e_batch_host = asnumpy(e_batch).astype(accum_dtype, copy=False)
        h_batch_host = asnumpy(h_batch).astype(accum_dtype, copy=False)
        if has_overlap:
            np.add.at(e, point_batch, e_batch_host)
            np.add.at(h, point_batch, h_batch_host)
        else:
            e[point_batch] = e_batch_host
            h[point_batch] = h_batch_host
        if progress is not None:
            progress.update(stop - start)

    if progress is not None:
        progress.close()
    return e, h, inside


@dataclass(frozen=True, slots=True)
class _LayeredRadialPlan:
    """Host-owned radial tables shared by NumPy and CuPy layered batches."""

    archetype_ids: np.ndarray
    layer_radii_by_archetype: dict[int, np.ndarray]
    layer_offsets: np.ndarray
    refractive_indices: np.ndarray
    m_regular: np.ndarray
    m_outgoing: np.ndarray
    n_regular: np.ndarray
    n_outgoing: np.ndarray


def _prepare_layered_radial_plan(
    particles: ParticleCollection,
    pair_archetypes: np.ndarray,
    *,
    lmax: int,
    k: float,
    n_medium: complex,
    compute_dtype: np.dtype,
) -> _LayeredRadialPlan:
    """Normalize active layered archetypes into compact flat radial tables."""
    active_archetypes = np.unique(np.asarray(pair_archetypes, dtype=np.int64))
    layer_radii_by_archetype: dict[int, np.ndarray] = {}
    layer_offsets = np.empty(active_archetypes.shape, dtype=np.int64)
    refractive_chunks: list[np.ndarray] = []
    m_regular_chunks: list[np.ndarray] = []
    m_outgoing_chunks: list[np.ndarray] = []
    n_regular_chunks: list[np.ndarray] = []
    n_outgoing_chunks: list[np.ndarray] = []
    offset = 0

    for active_index, archetype_id in enumerate(active_archetypes):
        archetype = particles.archetypes[int(archetype_id)]
        if not isinstance(archetype, LayeredSphere):
            raise TypeError("Layered evaluation received a non-layered archetype.")
        layer_radii = np.asarray(archetype.layer_radii, dtype=np.float64)
        layer_indices = np.asarray(archetype.layer_refractive_indices, dtype=np.complex128)
        ratios = layered_internal_ab_ratios(
            lmax=lmax,
            k_medium=float(k),
            layer_radii=archetype.layer_radii,
            layer_refractive_indices=archetype.layer_refractive_indices,
            n_medium=n_medium,
        )
        archetype_index = int(archetype_id)
        layer_radii_by_archetype[archetype_index] = layer_radii
        layer_offsets[active_index] = offset
        refractive_chunks.append(layer_indices)
        m_regular_chunks.append(np.asarray(ratios[1]["A"], dtype=compute_dtype))
        m_outgoing_chunks.append(np.asarray(ratios[1]["B"], dtype=compute_dtype))
        n_regular_chunks.append(np.asarray(ratios[2]["A"], dtype=compute_dtype))
        n_outgoing_chunks.append(np.asarray(ratios[2]["B"], dtype=compute_dtype))
        offset += int(layer_radii.size)

    return _LayeredRadialPlan(
        archetype_ids=active_archetypes,
        layer_radii_by_archetype=layer_radii_by_archetype,
        layer_offsets=layer_offsets,
        refractive_indices=np.concatenate(refractive_chunks),
        m_regular=np.concatenate(m_regular_chunks, axis=0),
        m_outgoing=np.concatenate(m_outgoing_chunks, axis=0),
        n_regular=np.concatenate(n_regular_chunks, axis=0),
        n_outgoing=np.concatenate(n_outgoing_chunks, axis=0),
    )


def _layered_flat_layer_ids(
    archetype_batch: np.ndarray,
    radii: np.ndarray,
    plan: _LayeredRadialPlan,
) -> np.ndarray:
    """Map each layered particle-point pair to one row of the radial plan."""
    local_layers = np.empty(archetype_batch.shape, dtype=np.int64)
    for archetype_id in np.unique(archetype_batch):
        mask = archetype_batch == archetype_id
        layer_radii = plan.layer_radii_by_archetype[int(archetype_id)]
        local_layers[mask] = np.clip(
            np.searchsorted(layer_radii, radii[mask], side="right"),
            0,
            layer_radii.size - 1,
        )
    plan_positions = np.searchsorted(plan.archetype_ids, archetype_batch)
    valid = plan_positions < plan.archetype_ids.size
    if np.any(valid):
        valid_positions = np.flatnonzero(valid)
        valid[valid_positions] &= (
            plan.archetype_ids[plan_positions[valid_positions]] == archetype_batch[valid_positions]
        )
    if not np.all(valid):
        raise RuntimeError("Layered radial plan is missing an active archetype.")
    return plan.layer_offsets[plan_positions] + local_layers


def _compute_internal_field_layered_spheres_numpy_batched(
    field_points: np.ndarray,
    particles: ParticleCollection,
    coeffs: np.ndarray,
    layered_indices: np.ndarray,
    *,
    k: float,
    lmax: int,
    classification: InternalPointClassification | None,
    n_medium: complex,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute layered-sphere internal fields with batched NumPy work."""
    real_dtype = np.dtype(np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64)
    n_points = int(field_points.shape[0])
    selected = np.asarray(layered_indices, dtype=np.int64).reshape(-1)
    positions = particles.positions[selected]
    outer_radii = particles.circumscribing_radii[selected]
    pair_points, pair_spheres = _radial_internal_point_pairs(
        field_points,
        positions,
        outer_radii,
        classification=classification,
        particle_indices=selected,
    )
    inside = np.zeros((n_points,), dtype=bool)
    if pair_points.size == 0:
        empty = np.zeros((n_points, 3), dtype=accum_dtype)
        return empty, empty.copy(), inside

    unique_pair_points = np.unique(pair_points)
    inside[unique_pair_points] = True
    global_particles = selected[pair_spheres]
    pair_archetypes = particles.archetype_indices[global_particles].astype(np.int64, copy=False)
    radial_plan = _prepare_layered_radial_plan(
        particles,
        pair_archetypes,
        lmax=lmax,
        k=k,
        n_medium=n_medium,
        compute_dtype=compute_dtype,
    )

    coeffs_host = np.asarray(coeffs, dtype=compute_dtype)
    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    has_overlap = unique_pair_points.size != pair_points.size
    mode_by_l = mode_indices_by_l(lmax)
    progress = (
        tqdm(total=pair_points.size, desc="Internal field (layered pairs)", unit="pair")
        if show_progress
        else None
    )

    for start in range(0, pair_points.size, _INTERNAL_PAIR_BATCH_SIZE):
        stop = min(pair_points.size, start + _INTERNAL_PAIR_BATCH_SIZE)
        point_batch = pair_points[start:stop]
        global_particle_batch = global_particles[start:stop]
        archetype_batch = pair_archetypes[start:stop]
        rvec_full = field_points[point_batch] - particles.positions[global_particle_batch]
        r = np.linalg.norm(rvec_full, axis=1)
        r_safe = np.where(r < 1.0e-12, 1.0e-12, r)
        rvec = np.asarray(rvec_full, dtype=real_dtype)
        r_safe_angular = np.asarray(r_safe, dtype=real_dtype)
        flat_layer_ids = _layered_flat_layer_ids(archetype_batch, r, radial_plan)
        n_layer = radial_plan.refractive_indices[flat_layer_ids]
        kr = np.asarray(k * (n_layer / n_medium) * r_safe, dtype=compute_dtype)

        x = rvec[:, 0]
        y = rvec[:, 1]
        z = rvec[:, 2]
        rho = np.sqrt(x * x + y * y)
        ct = z / r_safe_angular
        st = rho / r_safe_angular
        phi = np.arctan2(y, x)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        e_r = np.stack([st * cos_phi, st * sin_phi, ct], axis=1)
        e_theta = np.stack([ct * cos_phi, ct * sin_phi, -st], axis=1)
        e_phi = np.stack([-sin_phi, cos_phi, np.zeros_like(phi)], axis=1)
        pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)
        e_batch = np.zeros((stop - start, 3), dtype=accum_dtype)
        h_batch = np.zeros_like(e_batch)
        h_factor = np.asarray(-1j * n_layer, dtype=compute_dtype)

        for l in range(1, lmax + 1):
            a_m = radial_plan.m_regular[flat_layer_ids, l]
            b_m = radial_plan.m_outgoing[flat_layer_ids, l]
            a_n = radial_plan.n_regular[flat_layer_ids, l]
            b_n = radial_plan.n_outgoing[flat_layer_ids, l]

            jl = np.asarray(spherical_jn(l, kr), dtype=compute_dtype)
            djl = np.asarray(spherical_jn(l, kr, derivative=True), dtype=compute_dtype)
            use_h = (b_m != 0) | (b_n != 0)
            yl = np.zeros_like(jl)
            dyl = np.zeros_like(djl)
            if np.any(use_h):
                yl[use_h] = np.asarray(spherical_yn(l, kr[use_h]), dtype=compute_dtype)
                dyl[use_h] = np.asarray(
                    spherical_yn(l, kr[use_h], derivative=True), dtype=compute_dtype
                )
            hl = np.asarray(jl + 1j * yl, dtype=compute_dtype)
            dhl = np.asarray(djl + 1j * dyl, dtype=compute_dtype)
            z_m = np.asarray(a_m * jl + b_m * hl, dtype=compute_dtype)
            dxxz_m = np.asarray(a_m * (jl + kr * djl) + b_m * (hl + kr * dhl), dtype=compute_dtype)
            z_n = np.asarray(a_n * jl + b_n * hl, dtype=compute_dtype)
            dxxz_n = np.asarray(a_n * (jl + kr * djl) + b_n * (hl + kr * dhl), dtype=compute_dtype)
            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            m_m, n_m = build_internal_mode_tensors(
                l=l,
                m_vals=m_vals,
                abs_m=abs_m,
                phi=phi,
                e_r=e_r,
                e_theta=e_theta,
                e_phi=e_phi,
                pi_all=pi_all,
                tau_all=tau_all,
                p_all=p_all,
                z_l=z_m,
                dxxz=dxxz_m,
                kr=kr,
                compute_dtype=compute_dtype,
            )
            m_n, n_n = build_internal_mode_tensors(
                l=l,
                m_vals=m_vals,
                abs_m=abs_m,
                phi=phi,
                e_r=e_r,
                e_theta=e_theta,
                e_phi=e_phi,
                pi_all=pi_all,
                tau_all=tau_all,
                p_all=p_all,
                z_l=z_n,
                dxxz=dxxz_n,
                kr=kr,
                compute_dtype=compute_dtype,
            )
            a_out = coeffs_host[global_particle_batch[:, None], n1_idx]
            b_out = coeffs_host[global_particle_batch[:, None], n2_idx]
            e_batch += np.einsum("bm,bmc->bc", a_out, m_m)
            e_batch += np.einsum("bm,bmc->bc", b_out, n_n)
            h_batch += h_factor[:, None] * np.einsum("bm,bmc->bc", a_out, n_m)
            h_batch += h_factor[:, None] * np.einsum("bm,bmc->bc", b_out, m_n)

        if has_overlap:
            np.add.at(e, point_batch, e_batch)
            np.add.at(h, point_batch, h_batch)
        else:
            e[point_batch] = e_batch
            h[point_batch] = h_batch
        if progress is not None:
            progress.update(stop - start)

    if progress is not None:
        progress.close()
    return e, h, inside


def _compute_internal_field_layered_spheres_cupy(
    field_points: np.ndarray,
    particles: ParticleCollection,
    coeffs: np.ndarray,
    layered_indices: np.ndarray,
    *,
    k: float,
    lmax: int,
    classification: InternalPointClassification | None,
    n_medium: complex,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute layered-sphere internal fields with batched CuPy contractions."""
    cupy, _ = import_cupy()
    compute_dtype_cp = (
        cupy.complex64 if compute_dtype == np.dtype(np.complex64) else cupy.complex128
    )
    accum_dtype_cp = cupy.complex64 if accum_dtype == np.dtype(np.complex64) else cupy.complex128
    real_dtype = np.dtype(np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64)
    real_dtype_cp = cupy.float32 if real_dtype == np.dtype(np.float32) else cupy.float64

    n_points = int(field_points.shape[0])
    selected = np.asarray(layered_indices, dtype=np.int64).reshape(-1)
    positions = particles.positions[selected]
    outer_radii = particles.circumscribing_radii[selected]
    pair_points, pair_spheres = _radial_internal_point_pairs(
        field_points,
        positions,
        outer_radii,
        classification=classification,
        particle_indices=selected,
    )
    inside = np.zeros((n_points,), dtype=bool)
    if pair_points.size == 0:
        empty = np.zeros((n_points, 3), dtype=accum_dtype)
        return empty, empty.copy(), inside

    unique_pair_points = np.unique(pair_points)
    inside[unique_pair_points] = True
    global_particles = selected[pair_spheres]
    pair_archetypes = particles.archetype_indices[global_particles].astype(np.int64, copy=False)
    radial_plan = _prepare_layered_radial_plan(
        particles,
        pair_archetypes,
        lmax=lmax,
        k=k,
        n_medium=n_medium,
        compute_dtype=compute_dtype,
    )

    coeffs_host = np.asarray(coeffs, dtype=compute_dtype)
    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    has_overlap = unique_pair_points.size != pair_points.size
    mode_by_l = mode_indices_by_l(lmax)
    pair_batch_size = _cupy_internal_pair_batch_size(
        cupy=cupy,
        total_pairs=int(pair_points.size),
        lmax=lmax,
        compute_dtype=compute_dtype,
    )
    progress = (
        tqdm(total=pair_points.size, desc="Internal field (layered pairs)", unit="pair")
        if show_progress
        else None
    )

    for start in range(0, pair_points.size, pair_batch_size):
        stop = min(pair_points.size, start + pair_batch_size)
        point_batch = pair_points[start:stop]
        global_particle_batch = global_particles[start:stop]
        archetype_batch = pair_archetypes[start:stop]
        rvec_full = field_points[point_batch] - particles.positions[global_particle_batch]
        r_host = np.linalg.norm(rvec_full, axis=1)
        r_safe_host = np.where(r_host < 1.0e-12, 1.0e-12, r_host)
        rvec_host = np.asarray(rvec_full, dtype=real_dtype)
        flat_layer_ids = _layered_flat_layer_ids(archetype_batch, r_host, radial_plan)
        n_layer_host = radial_plan.refractive_indices[flat_layer_ids]
        kr_host = np.asarray(
            k * (n_layer_host / n_medium) * r_safe_host,
            dtype=compute_dtype,
        )

        rvec = cupy.asarray(rvec_host, dtype=real_dtype_cp)
        r_safe = cupy.asarray(r_safe_host, dtype=real_dtype_cp)
        kr = cupy.asarray(kr_host, dtype=compute_dtype_cp)
        x = rvec[:, 0]
        y = rvec[:, 1]
        z = rvec[:, 2]
        rho = cupy.sqrt(x * x + y * y)
        ct = z / r_safe
        st = rho / r_safe
        phi = cupy.arctan2(y, x)
        cos_phi = cupy.cos(phi)
        sin_phi = cupy.sin(phi)
        e_r = cupy.stack([st * cos_phi, st * sin_phi, ct], axis=1)
        e_theta = cupy.stack([ct * cos_phi, ct * sin_phi, -st], axis=1)
        e_phi = cupy.stack([-sin_phi, cos_phi, cupy.zeros_like(phi)], axis=1)
        pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=cupy, return_plm=True)
        e_batch = cupy.zeros((stop - start, 3), dtype=accum_dtype_cp)
        h_batch = cupy.zeros_like(e_batch)
        n_layer = cupy.asarray(n_layer_host, dtype=compute_dtype_cp)
        minus_i = cupy.asarray(-1j, dtype=compute_dtype_cp)
        h_factor = minus_i * n_layer

        for l in range(1, lmax + 1):
            a_m_host = radial_plan.m_regular[flat_layer_ids, l]
            b_m_host = radial_plan.m_outgoing[flat_layer_ids, l]
            a_n_host = radial_plan.n_regular[flat_layer_ids, l]
            b_n_host = radial_plan.n_outgoing[flat_layer_ids, l]

            jl_host = np.asarray(spherical_jn(l, kr_host), dtype=compute_dtype)
            djl_host = np.asarray(spherical_jn(l, kr_host, derivative=True), dtype=compute_dtype)
            use_h = (b_m_host != 0) | (b_n_host != 0)
            yl_host = np.zeros_like(jl_host)
            dyl_host = np.zeros_like(djl_host)
            if np.any(use_h):
                yl_host[use_h] = np.asarray(spherical_yn(l, kr_host[use_h]), dtype=compute_dtype)
                dyl_host[use_h] = np.asarray(
                    spherical_yn(l, kr_host[use_h], derivative=True), dtype=compute_dtype
                )
            hl_host = jl_host + 1j * yl_host
            dhl_host = djl_host + 1j * dyl_host
            z_m = a_m_host * jl_host + b_m_host * hl_host
            dxxz_m = a_m_host * (jl_host + kr_host * djl_host) + b_m_host * (
                hl_host + kr_host * dhl_host
            )
            z_n = a_n_host * jl_host + b_n_host * hl_host
            dxxz_n = a_n_host * (jl_host + kr_host * djl_host) + b_n_host * (
                hl_host + kr_host * dhl_host
            )
            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            m_m, n_m = build_internal_mode_tensors(
                l=l,
                m_vals=m_vals,
                abs_m=abs_m,
                phi=phi,
                e_r=e_r,
                e_theta=e_theta,
                e_phi=e_phi,
                pi_all=pi_all,
                tau_all=tau_all,
                p_all=p_all,
                z_l=cupy.asarray(z_m, dtype=compute_dtype_cp),
                dxxz=cupy.asarray(dxxz_m, dtype=compute_dtype_cp),
                kr=kr,
                compute_dtype=compute_dtype,
            )
            m_n, n_n = build_internal_mode_tensors(
                l=l,
                m_vals=m_vals,
                abs_m=abs_m,
                phi=phi,
                e_r=e_r,
                e_theta=e_theta,
                e_phi=e_phi,
                pi_all=pi_all,
                tau_all=tau_all,
                p_all=p_all,
                z_l=cupy.asarray(z_n, dtype=compute_dtype_cp),
                dxxz=cupy.asarray(dxxz_n, dtype=compute_dtype_cp),
                kr=kr,
                compute_dtype=compute_dtype,
            )
            m_m = m_m.astype(compute_dtype_cp, copy=False)
            n_m = n_m.astype(compute_dtype_cp, copy=False)
            m_n = m_n.astype(compute_dtype_cp, copy=False)
            n_n = n_n.astype(compute_dtype_cp, copy=False)
            a_out = cupy.asarray(
                coeffs_host[global_particle_batch[:, None], n1_idx],
                dtype=compute_dtype_cp,
            )
            b_out = cupy.asarray(
                coeffs_host[global_particle_batch[:, None], n2_idx],
                dtype=compute_dtype_cp,
            )
            e_batch += cupy.einsum("bm,bmc->bc", a_out, m_m).astype(accum_dtype_cp, copy=False)
            e_batch += cupy.einsum("bm,bmc->bc", b_out, n_n).astype(accum_dtype_cp, copy=False)
            h_batch += (h_factor[:, None] * cupy.einsum("bm,bmc->bc", a_out, n_m)).astype(
                accum_dtype_cp, copy=False
            )
            h_batch += (h_factor[:, None] * cupy.einsum("bm,bmc->bc", b_out, m_n)).astype(
                accum_dtype_cp, copy=False
            )

        e_batch_host = asnumpy(e_batch).astype(accum_dtype, copy=False)
        h_batch_host = asnumpy(h_batch).astype(accum_dtype, copy=False)
        if has_overlap:
            np.add.at(e, point_batch, e_batch_host)
            np.add.at(h, point_batch, h_batch_host)
        else:
            e[point_batch] = e_batch_host
            h[point_batch] = h_batch_host
        if progress is not None:
            progress.update(stop - start)

    if progress is not None:
        progress.close()
    return e, h, inside


def _compute_internal_field_particles(
    field_points: np.ndarray,
    particles: Sequence[Particle],
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    classification: InternalPointClassification | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute internal fields for a canonical particle collection."""
    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    part = ParticleCollection.from_particles(particles)
    n_particles = len(part)
    n_points = pts.shape[0]
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)
    n_medium_c = complex(n_medium)
    lmax = int(lmax)
    n_modes_total = n_modes(lmax)
    real_dtype = np.dtype(np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64)

    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros((n_points, 3), dtype=accum_dtype)
    inside = np.zeros(n_points, dtype=bool)
    if n_particles == 0:
        return e, h, inside

    if classification is not None:
        if classification.inside_any.shape != (n_points,):
            raise ValueError(
                "`classification.inside_any` must have shape "
                f"({n_points},). Got {classification.inside_any.shape}."
            )
        if classification.n_particles != n_particles:
            raise ValueError(
                "`classification.n_particles` must match "
                f"particle count ({n_particles}). Got {classification.n_particles}."
            )

    c = np.asarray(coeffs, dtype=compute_dtype)
    if c.size != n_particles * n_modes_total:
        raise ValueError(
            f"`coeffs` must have {n_particles * n_modes_total} entries for {n_particles} particles and lmax={lmax}. Got {c.size}."
        )
    c = c.reshape(n_particles, n_modes_total)

    pec_idx = part.indices_of_type(PECSphere)
    if pec_idx.size:
        for j in pec_idx:
            if classification is None:
                idx = np.flatnonzero(particle_contains_points(part[j], pts)).astype(
                    np.intp, copy=False
                )
            else:
                idx = classification.points_for_particle(j)
            inside[idx] = True
        if pec_idx.size == n_particles:
            return e, h, inside

    imported_idx = part.indices_of_type(TMatrixParticle)
    if imported_idx.size:
        # An imported T matrix intentionally carries no interior material
        # model. Preserve the ownership mask but make those samples explicit
        # NaNs rather than silently presenting the host or a zero field.
        for index in imported_idx:
            if classification is None:
                idx = np.flatnonzero(particle_contains_points(part[int(index)], pts)).astype(
                    np.intp, copy=False
                )
            else:
                idx = classification.points_for_particle(int(index))
            inside[idx] = True
            e[idx] = np.nan + 0j
            h[idx] = np.nan + 0j
        if imported_idx.size == n_particles or (imported_idx.size + pec_idx.size == n_particles):
            return e, h, inside

    sphere_arrays = part.homogeneous_sphere_arrays()
    if sphere_arrays is not None:
        positions, radii, n_particle = sphere_arrays
        return _compute_internal_field_homogeneous_spheres(
            pts,
            positions,
            radii,
            c,
            k=float(k),
            lmax=lmax,
            n_particle=n_particle,
            classification=classification,
            n_medium=n_medium_c,
            show_progress=show_progress,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )

    supported = (Sphere, PECSphere, LayeredSphere, Spheroid, TMatrixParticle)
    bad = [type(p).__name__ for p in part.archetypes if not isinstance(p, supported)]
    if bad:
        raise TypeError(
            "compute_internal_field currently supports Sphere, PECSphere, LayeredSphere, Spheroid, "
            "and circumscribing-mask handling for TMatrixParticle in "
            f"particle-dispatch mode. Got {bad}."
        )

    sphere_idx = part.indices_of_type(Sphere)
    layered_idx = part.indices_of_type(LayeredSphere)
    spheroid_idx = part.indices_of_type(Spheroid)

    if sphere_idx.size:
        sphere_arrays = part.homogeneous_sphere_arrays(sphere_idx)
        if sphere_arrays is None:
            raise RuntimeError("Sphere dispatch selected a non-sphere archetype.")
        positions, radii, n_particle = sphere_arrays
        c_sphere = c[sphere_idx, :]
        e_s, h_s, inside_s = _compute_internal_field_homogeneous_spheres(
            pts,
            positions,
            radii,
            c_sphere,
            k=float(k),
            lmax=lmax,
            n_particle=n_particle,
            classification=classification,
            particle_indices=sphere_idx,
            n_medium=n_medium_c,
            show_progress=show_progress,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        e += e_s
        h += h_s
        inside |= inside_s

    if spheroid_idx.size:
        eps = 1e-12
        mode_by_l = mode_indices_by_l(lmax)
        sph_iter: Iterable[int] = (int(index) for index in spheroid_idx)
        if show_progress:
            sph_iter = tqdm(
                (int(index) for index in spheroid_idx),
                total=int(spheroid_idx.size),
                desc="Internal field (spheroids)",
                leave=True,
            )
        internal_block_memo: dict[int, np.ndarray] = {}

        for j_sphere in sph_iter:
            archetype_id = int(part.archetype_indices[j_sphere])
            archetype = part.archetypes[archetype_id]
            if not isinstance(archetype, Spheroid):
                continue
            if classification is None:
                idx = np.flatnonzero(particle_contains_points(part[j_sphere], pts))
            else:
                idx = classification.points_for_particle(j_sphere)
            if idx.size == 0:
                continue

            inside[idx] = True
            internal_map = internal_block_memo.get(archetype_id)
            if internal_map is None:
                internal_map = _spheroid_internal_block(
                    lmax=lmax,
                    k_medium=float(k),
                    particle=archetype,
                    n_medium=n_medium_c,
                )
                internal_block_memo[archetype_id] = internal_map

            c_internal = np.asarray(internal_map @ c[j_sphere], dtype=compute_dtype)
            center = part.positions[j_sphere]
            rvec_full = pts[idx] - center[None, :]
            r2 = np.sum(rvec_full * rvec_full, axis=1)
            r = np.sqrt(r2)
            r_safe = np.where(r < eps, eps, r)
            rvec = np.asarray(rvec_full, dtype=real_dtype)
            r_safe_angular = np.asarray(r_safe, dtype=real_dtype)

            x = rvec[:, 0]
            y = rvec[:, 1]
            z = rvec[:, 2]
            rho = np.sqrt(x * x + y * y)
            ct = z / r_safe_angular
            st = rho / r_safe_angular
            phi = np.arctan2(y, x)

            e_r = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
            e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
            e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
            pi_all, tau_all, p_all = spherical_functions_trigon(
                ct, st, lmax, xp=np, return_plm=True
            )

            n_s = complex(archetype.refractive_index)
            kr_full = float(k) * (n_s / n_medium_c) * r_safe

            for l in range(1, lmax + 1):
                m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
                a_int = c_internal[n1_idx].astype(compute_dtype, copy=False)
                b_int = c_internal[n2_idx].astype(compute_dtype, copy=False)
                jl = spherical_jn(l, kr_full)
                djl = spherical_jn(l, kr_full, derivative=True)
                m_reg, n_reg = build_internal_mode_tensors(
                    l=l,
                    m_vals=m_vals,
                    abs_m=abs_m,
                    phi=phi,
                    e_r=e_r,
                    e_theta=e_theta,
                    e_phi=e_phi,
                    pi_all=pi_all,
                    tau_all=tau_all,
                    p_all=p_all,
                    z_l=np.asarray(jl, dtype=compute_dtype),
                    dxxz=np.asarray(jl + kr_full * djl, dtype=compute_dtype),
                    kr=np.asarray(kr_full, dtype=compute_dtype),
                    compute_dtype=compute_dtype,
                )
                e[idx] += contract_modes(a_int, m_reg)
                e[idx] += contract_modes(b_int, n_reg)
                h[idx] += (-1j * n_s) * contract_modes(a_int, n_reg)
                h[idx] += (-1j * n_s) * contract_modes(b_int, m_reg)

    if layered_idx.size:
        if str(backend).lower() == "cupy":
            layered_compute = _compute_internal_field_layered_spheres_cupy
        else:
            layered_compute = _compute_internal_field_layered_spheres_numpy_batched
        e_layered, h_layered, inside_layered = layered_compute(
            pts,
            part,
            c,
            layered_idx,
            k=float(k),
            lmax=lmax,
            classification=classification,
            n_medium=n_medium_c,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        e += e_layered
        h += h_layered
        inside |= inside_layered
        return e, h, inside

    return e, h, inside


__all__ = ["compute_internal_field"]
