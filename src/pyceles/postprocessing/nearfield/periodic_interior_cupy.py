"""CuPy owner for periodic in-slab near-field coefficients.

Exact Ewald batches evaluate all pairs for ``method="ewald"``.  The Rayleigh
path instead uses reciprocal scans for vertically separated sources and retains
exact Ewald only inside the configured z band.  Both paths contract directly
into the local regular ``l=1`` sector needed by field reconstruction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from pyceles._cupy_memory import cupy_allocator_snapshot
from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._optional import import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.periodic.ewald import resolve_ewald_eta, resolve_ewald_shell_counts
from pyceles.core.periodic.ewald_cupy import (
    CupyEwaldShellWorkspace,
    ewald_structural_sums_2d_fixed_cupy,
)
from pyceles.core.periodic.rayleigh import near_point_source_csr, prepare_rayleigh_plan
from pyceles.core.periodic.rayleigh_cupy import (
    apply_rayleigh_far_to_points_cupy,
    scatter_add_complex,
)

from .periodic_projection import l1_compact_projection_data


def _build_source_projection_kernels_cupy(
    *,
    lmax: int,
    coeffs: np.ndarray,
    compute_dtype: np.dtype,
    cupy: Any,
) -> tuple[int, int, Any, Any, Any]:
    """Return compact source-specific ``l=1`` projection data on the GPU."""
    (
        lmax_struct,
        structural_order,
        degree_indices,
        order_indices,
        kernel_np,
    ) = l1_compact_projection_data(int(lmax))
    dtype_cp = cupy.complex64 if compute_dtype == np.dtype(np.complex64) else cupy.complex128
    coeff_cp = cupy.asarray(np.asarray(coeffs, dtype=compute_dtype), dtype=dtype_cp)
    kernel_cp = cupy.asarray(np.asarray(kernel_np, dtype=compute_dtype), dtype=dtype_cp)
    source_kernel = cupy.einsum("rck,jc->jrk", kernel_cp, coeff_cp, optimize=True)
    return (
        int(lmax_struct),
        int(structural_order),
        cupy.asarray(degree_indices, dtype=cupy.int32),
        cupy.asarray(order_indices, dtype=cupy.int32),
        source_kernel,
    )


_PERIODIC_NEAR_PAIR_BATCH_CAP = 65_536
_PERIODIC_NEAR_WORKSPACE_HEADROOM_FRACTION = 0.5


def _periodic_near_pair_workspace_bytes_per_pair(
    *,
    lmax: int,
    compute_dtype: np.dtype,
) -> int:
    """Conservatively estimate live CuPy workspace for one exact-near pair."""
    lmax_i = max(1, int(lmax))
    structural_order = lmax_i + 1
    rectangular_channels = (structural_order + 1) * (2 * structural_order + 1)
    compact_channels = (structural_order + 1) ** 2
    compute_itemsize = int(np.dtype(compute_dtype).itemsize)
    # Structural sums are evaluated in complex128. The compact contraction also
    # materializes a source-specific (6, K) gather per pair. Keep extra room
    # for relative coordinates, masks, output rows, and allocator retention.
    return int(
        rectangular_channels * np.dtype(np.complex128).itemsize
        + compact_channels * compute_itemsize
        + 6 * compact_channels * compute_itemsize
        + 6 * compute_itemsize
        + 128
    )


def _periodic_near_pair_batch_size_for_workspace(
    *,
    total_pairs: int,
    lmax: int,
    compute_dtype: np.dtype,
    workspace_bytes: int,
) -> int:
    """Choose a nonzero exact-near pair batch within a workspace budget."""
    total = max(1, int(total_pairs))
    bytes_per_pair = _periodic_near_pair_workspace_bytes_per_pair(
        lmax=int(lmax),
        compute_dtype=np.dtype(compute_dtype),
    )
    memory_cap = max(1, int(workspace_bytes) // max(1, bytes_per_pair))
    return max(1, min(total, _PERIODIC_NEAR_PAIR_BATCH_CAP, memory_cap))


def _cupy_periodic_near_pair_batch_size(
    *,
    cupy: Any,
    total_pairs: int,
    lmax: int,
    compute_dtype: np.dtype,
) -> int:
    """Resolve an exact-near batch from current allocator headroom without mutation."""
    snapshot = cupy_allocator_snapshot(cupy, apply_pool_limit=False)
    reusable_or_fresh = int(snapshot.raw_free_bytes) + int(snapshot.pool_free_bytes)
    usable_headroom = max(
        1,
        min(int(snapshot.active_headroom_bytes), int(reusable_or_fresh)),
    )
    workspace_bytes = max(
        1,
        int(float(usable_headroom) * _PERIODIC_NEAR_WORKSPACE_HEADROOM_FRACTION),
    )
    return _periodic_near_pair_batch_size_for_workspace(
        total_pairs=int(total_pairs),
        lmax=int(lmax),
        compute_dtype=np.dtype(compute_dtype),
        workspace_bytes=int(workspace_bytes),
    )


def _resolve_eta(
    *,
    periodic: Any,
    k: float,
    k_parallel: np.ndarray,
    positions: np.ndarray,
    lmax: int,
    max_vertical_offset: float | None = None,
) -> float:
    return resolve_ewald_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        lmax=int(lmax),
        max_vertical_offset=max_vertical_offset,
    )


def periodic_local_regular_l1_coeffs_cupy(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    periodic: Any,
    k_parallel: np.ndarray,
    circumscribing_radii: np.ndarray | None = None,
    point_batch_size: int = 128,
    source_batch_size: int | None = None,
    show_progress: bool = False,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Return point-local regular ``l=1`` coefficients using the CuPy path.

    The function is intentionally specialized to periodic in-slab near fields.
    It uses explicit shell ranges when supplied, otherwise a centralized
    one-shot shell resolver chooses fixed ranges compatible with the NumPy
    adaptive options. With ``method='rayleigh'``, reciprocal scans evaluate
    vertically separated source-point pairs and only the exact near band enters
    the Ewald device loop.
    """
    cp, _ = import_cupy()
    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    compute_dtype_cp = cp.complex64 if compute_dtype == np.dtype(np.complex64) else cp.complex128
    accum_dtype_cp = cp.complex64 if accum_dtype == np.dtype(np.complex64) else cp.complex128
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    coeff_arr = np.asarray(coeffs, dtype=compute_dtype).reshape(pos.shape[0], n_modes(int(lmax)))
    out = np.zeros((pts.shape[0], 6), dtype=accum_dtype)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return out
    method = str(periodic.options.method)
    if method not in {"ewald", "rayleigh"}:
        raise NotImplementedError(
            "Periodic in-slab near-field evaluation requires "
            "`periodic.options.method` to be 'ewald' or 'rayleigh'."
        )

    point_batch_size = max(1, int(point_batch_size))
    if source_batch_size is None:
        # The dense Ewald-only path still uses a rectangular point/source
        # batch. The hybrid Rayleigh path resolves its sparse pair batch from
        # allocator headroom below.
        source_batch_size = max(1, min(pos.shape[0], 2048 // point_batch_size))
    source_batch_size = max(1, int(source_batch_size))

    (
        lmax_struct,
        structural_order,
        degree_indices_cp,
        order_indices_cp,
        source_kernel_cp,
    ) = _build_source_projection_kernels_cupy(
        lmax=int(lmax),
        coeffs=coeff_arr,
        compute_dtype=compute_dtype,
        cupy=cp,
    )

    rayleigh_z_cut: float | None = None
    if method == "rayleigh":
        plan = prepare_rayleigh_plan(
            lmax=int(lmax),
            k=float(k),
            positions=pos,
            circumscribing_radii=circumscribing_radii,
            periodic=periodic,
            k_parallel=k_parallel,
            dtype=compute_dtype,
        )
        local_cp = apply_rayleigh_far_to_points_cupy(
            plan,
            coeff_arr,
            pts,
            cupy=cp,
        ).astype(accum_dtype_cp, copy=False)
        _indptr, destination_indices, source_indices = near_point_source_csr(
            pts,
            pos,
            plan.z_cut,
        )
        rayleigh_z_cut = float(plan.z_cut)
        if destination_indices.size == 0:
            return np.asarray(cp.asnumpy(local_cp), dtype=accum_dtype)
    else:
        local_cp = None
        destination_indices = np.zeros((0,), dtype=np.int64)
        source_indices = np.zeros((0,), dtype=np.int64)

    eta = _resolve_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=pos,
        lmax=int(lmax),
        max_vertical_offset=rayleigh_z_cut,
    )

    shell_counts = resolve_ewald_shell_counts(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=pos,
        lmax=int(lmax),
        eta=float(eta),
        max_vertical_offset=rayleigh_z_cut,
    )
    real_count = int(shell_counts.real_shells)
    recip_count = int(shell_counts.reciprocal_shells)
    coordinate_scale = float(max(np.max(np.abs(pos)), np.max(np.abs(pts))))
    workspace = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=np.float64).reshape(2),
        eta=float(eta),
    )

    if method == "rayleigh":
        if local_cp is None:
            raise RuntimeError("Rayleigh near-field initialization failed.")
        pair_batch_size = _cupy_periodic_near_pair_batch_size(
            cupy=cp,
            total_pairs=int(destination_indices.size),
            lmax=int(lmax),
            compute_dtype=compute_dtype,
        )
        destinations_cp = cp.asarray(destination_indices, dtype=cp.int32)
        sources_cp = cp.asarray(source_indices, dtype=cp.int32)
        pts_cp = cp.asarray(pts, dtype=cp.float64)
        pos_cp = cp.asarray(pos, dtype=cp.float64)
        progress = None
        if show_progress:
            from tqdm.auto import tqdm

            progress = tqdm(
                total=(destination_indices.size + pair_batch_size - 1) // pair_batch_size,
                desc="Periodic slab field (CuPy)",
                unit="near-batch",
                leave=True,
            )
        try:
            for start in range(0, destination_indices.size, pair_batch_size):
                stop = min(destination_indices.size, start + pair_batch_size)
                dst = destinations_cp[start:stop]
                src = sources_cp[start:stop]
                rel_cp = pos_cp[src] - pts_cp[dst]
                sums = ewald_structural_sums_2d_fixed_cupy(
                    relative_source_minus_destination=rel_cp,
                    lmax_struct=int(lmax_struct),
                    structural_order=int(structural_order),
                    workspace=workspace,
                    real_shell_count=int(real_count),
                    reciprocal_shell_count=int(recip_count),
                    coordinate_scale=coordinate_scale,
                )
                compact = sums[:, degree_indices_cp, order_indices_cp].astype(
                    compute_dtype_cp, copy=False
                )
                pair_kernel = source_kernel_cp[src]
                pair_l1 = cp.einsum(
                    "nk,nrk->nr",
                    compact,
                    pair_kernel,
                    optimize=True,
                )
                scatter_add_complex(local_cp, dst, pair_l1, cupy=cp)
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
        return np.asarray(cp.asnumpy(local_cp), dtype=accum_dtype)

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            n_point_batches = (pts.shape[0] + point_batch_size - 1) // point_batch_size
            n_source_batches = (pos.shape[0] + source_batch_size - 1) // source_batch_size
            progress = tqdm(
                total=int(n_point_batches * n_source_batches),
                desc="Periodic slab field (CuPy)",
                unit="batch",
                leave=True,
            )
        except Exception:  # pragma: no cover - progress is diagnostic only
            progress = None

    try:
        for p0 in range(0, pts.shape[0], point_batch_size):
            p1 = min(pts.shape[0], p0 + point_batch_size)
            pts_batch = np.asarray(pts[p0:p1], dtype=np.float64)
            local_cp = cp.zeros((pts_batch.shape[0], 6), dtype=accum_dtype_cp)
            for s0 in range(0, pos.shape[0], source_batch_size):
                s1 = min(pos.shape[0], s0 + source_batch_size)
                src_batch = np.asarray(pos[s0:s1], dtype=np.float64)
                rel = src_batch[None, :, :] - pts_batch[:, None, :]
                n_points = int(pts_batch.shape[0])
                n_sources = int(src_batch.shape[0])
                rel_cp = cp.asarray(rel.reshape(-1, 3), dtype=cp.float64)
                sums = ewald_structural_sums_2d_fixed_cupy(
                    relative_source_minus_destination=rel_cp,
                    lmax_struct=int(lmax_struct),
                    structural_order=int(structural_order),
                    workspace=workspace,
                    real_shell_count=int(real_count),
                    reciprocal_shell_count=int(recip_count),
                    coordinate_scale=float(
                        max(np.max(np.abs(src_batch)), np.max(np.abs(pts_batch)))
                    ),
                )
                src_idx = cp.asarray(
                    np.tile(np.arange(n_sources, dtype=np.int64), n_points), dtype=cp.int64
                )
                compact = sums[:, degree_indices_cp, order_indices_cp].astype(
                    compute_dtype_cp, copy=False
                )
                pair_kernel = source_kernel_cp[s0:s1][src_idx]
                pair_l1 = cp.einsum(
                    "nk,nrk->nr",
                    compact,
                    pair_kernel,
                    optimize=True,
                )
                local_cp += cp.sum(
                    pair_l1.reshape(n_points, n_sources, 6),
                    axis=1,
                    dtype=accum_dtype_cp,
                )
                if progress is not None:
                    progress.update(1)
            out[p0:p1] = cp.asnumpy(local_cp)
    finally:
        if progress is not None:
            progress.close()
    return out


__all__ = ["periodic_local_regular_l1_coeffs_cupy"]
