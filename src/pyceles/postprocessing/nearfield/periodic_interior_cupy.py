"""CuPy owner for periodic in-slab near-field coefficients.

The in-slab near-field path batches point/source pairs on the device and
contracts periodic Ewald scalar sums directly into the local regular ``l=1``
sector needed by field reconstruction.  Fixed shell ranges are resolved once at
call setup so the hot device loop avoids adaptive per-shell host synchronization.
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
    point_batch_size: int = 128,
    source_batch_size: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Return point-local regular ``l=1`` coefficients using the CuPy path.

    The function is intentionally specialized to periodic in-slab near fields.
    It uses explicit shell ranges when supplied, otherwise a centralized
    one-shot shell resolver chooses fixed ranges compatible with the NumPy
    adaptive options. This avoids adaptive host/device synchronizations in the
    hot CuPy loop.
    """
    cp, _ = import_cupy()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    coeff_arr = np.asarray(coeffs, dtype=np.complex128).reshape(pos.shape[0], n_modes(int(lmax)))
    out = np.zeros((pts.shape[0], 6), dtype=np.complex128)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return out
    eta = _resolve_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=pos,
        lmax=int(lmax),
    )

    lmax_struct, p_count, source_kernel_cp = _build_source_projection_kernels_cupy(
        lmax=int(lmax),
        coeffs=coeff_arr,
        cupy=cp,
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
    if source_batch_size is None:
        # Keep the flattened pair count modest.  This is not a public tuning
        # parameter; it only bounds temporary GPU arrays.
        source_batch_size = max(1, min(pos.shape[0], 2048 // max(1, int(point_batch_size))))
    source_batch_size = max(1, int(source_batch_size))
    point_batch_size = max(1, int(point_batch_size))

    workspace = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=np.float64).reshape(2),
        eta=float(eta),
    )

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
