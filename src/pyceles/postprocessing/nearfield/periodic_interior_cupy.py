"""CuPy owner for periodic in-slab near-field coefficients.

Exact Ewald batches evaluate all pairs for ``method="ewald"``.  The Rayleigh
path instead uses reciprocal scans for vertically separated sources and retains
exact Ewald only inside the configured z band.  Both paths contract directly
into the local regular ``l=1`` sector needed by field reconstruction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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

from .periodic_projection import l1_projection_data


def _build_source_projection_kernels_cupy(
    *,
    lmax: int,
    coeffs: np.ndarray,
    cupy: Any,
) -> tuple[int, int, Any]:
    """Return source-specific ``l=1`` projection kernels on the GPU."""
    lmax_struct, _m_offset, kernel_np, _row_idx = l1_projection_data(int(lmax))
    p_count = int(kernel_np.shape[3])
    # kernel: (row, coeff, m, p), coeffs: (source, coeff)
    source_kernel = np.einsum(
        "rcmp,jc->jrpm",
        np.asarray(kernel_np[:, :, :, :p_count], dtype=np.complex128),
        np.asarray(coeffs, dtype=np.complex128),
        optimize=True,
    )
    return int(lmax_struct), p_count, cupy.asarray(source_kernel, dtype=cupy.complex128)


def _resolve_eta(
    *,
    periodic: Any,
    k: float,
    k_parallel: np.ndarray,
    positions: np.ndarray,
    lmax: int,
) -> float:
    return resolve_ewald_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        lmax=int(lmax),
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
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    coeff_arr = np.asarray(coeffs, dtype=np.complex128).reshape(pos.shape[0], n_modes(int(lmax)))
    out = np.zeros((pts.shape[0], 6), dtype=np.complex128)
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
        # Keep the flattened exact-pair count modest. This is not a public
        # tuning parameter; it only bounds temporary GPU arrays.
        source_batch_size = max(1, min(pos.shape[0], 2048 // point_batch_size))
    source_batch_size = max(1, int(source_batch_size))

    lmax_struct, p_count, source_kernel_cp = _build_source_projection_kernels_cupy(
        lmax=int(lmax),
        coeffs=coeff_arr,
        cupy=cp,
    )

    if method == "rayleigh":
        plan = prepare_rayleigh_plan(
            lmax=int(lmax),
            k=float(k),
            positions=pos,
            circumscribing_radii=circumscribing_radii,
            periodic=periodic,
            k_parallel=k_parallel,
            dtype=np.complex128,
        )
        local_cp = apply_rayleigh_far_to_points_cupy(
            plan,
            coeff_arr,
            pts,
            cupy=cp,
        )
        _indptr, destination_indices, source_indices = near_point_source_csr(
            pts,
            pos,
            plan.z_cut,
        )
        if destination_indices.size == 0:
            return np.asarray(cp.asnumpy(local_cp), dtype=np.complex128)
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
    )

    shell_counts = resolve_ewald_shell_counts(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=pos,
        lmax=int(lmax),
        eta=float(eta),
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
        pair_batch_size = max(1, point_batch_size * source_batch_size)
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
                    workspace=workspace,
                    real_shell_count=int(real_count),
                    reciprocal_shell_count=int(recip_count),
                    coordinate_scale=coordinate_scale,
                )
                pair_kernel = source_kernel_cp[src]
                pair_l1 = cp.einsum(
                    "npm,nrpm->nr",
                    sums[:, :p_count, :],
                    pair_kernel,
                    optimize=True,
                )
                scatter_add_complex(local_cp, dst, pair_l1, cupy=cp)
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
        return np.asarray(cp.asnumpy(local_cp), dtype=np.complex128)

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
            local_cp = cp.zeros((pts_batch.shape[0], 6), dtype=cp.complex128)
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
                pair_kernel = source_kernel_cp[s0:s1][src_idx]
                pair_l1 = cp.einsum(
                    "npm,nrpm->nr",
                    sums[:, :p_count, :],
                    pair_kernel,
                    optimize=True,
                )
                local_cp += cp.sum(pair_l1.reshape(n_points, n_sources, 6), axis=1)
                if progress is not None:
                    progress.update(1)
            out[p0:p1] = cp.asnumpy(local_cp)
    finally:
        if progress is not None:
            progress.close()
    return out


__all__ = ["periodic_local_regular_l1_coeffs_cupy"]
