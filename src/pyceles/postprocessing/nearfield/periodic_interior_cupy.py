"""CuPy owner for periodic in-slab near-field coefficients.

This module intentionally does **not** mirror the full NumPy Ewald structural-sum
implementation line-by-line.  The in-slab near-field path only needs local
regular ``l=1`` coefficients at field points, so this file batches point/source
pairs on the device and contracts the Ewald scalar sums directly into that
``l=1`` sector.

The implementation stages shell metadata once per call, uses fixed shell ranges
on device to avoid per-shell host synchronization, avoids scatter accumulation,
and transfers only the final local ``l=1`` coefficients back to NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyceles._optional import import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic.ewald import select_ewald_eta
from pyceles.core.periodic.scalar import (
    chebyshev_shell_indices,
    factorial_int,
    structural_sum_m_normalization,
)
from pyceles.core.periodic.special_cupy import (
    real_integral_sequence_cupy,
    shifted_delta_sequence_cupy_batched,
)
from pyceles.core.spherical import legendre_normalized_trigon

from .periodic_interior import _l1_projection_data, _reduce_structural_sums_to_l1


@dataclass
class _CupyReciprocalShell:
    kgt: Any
    rho: Any
    phi: Any
    gamma: Any


@dataclass
class _CupyRealShell:
    shifts: Any
    phase_xy: Any


@dataclass
class _CupyFixedShellWorkspace:
    """Device-side shell metadata cache for one periodic near-field call."""

    cupy: Any
    lattice: RectangularLattice2D
    k: float
    k_parallel: np.ndarray
    eta: float
    reciprocal_cache: dict[int, _CupyReciprocalShell] = field(default_factory=dict)
    real_cache: dict[int, _CupyRealShell] = field(default_factory=dict)

    def reciprocal_shell(self, shell: int) -> _CupyReciprocalShell:
        idx = int(shell)
        cached = self.reciprocal_cache.get(idx)
        if cached is not None:
            return cached
        cp = self.cupy
        reciprocal = np.asarray(
            [p * self.lattice.b1 + q * self.lattice.b2 for p, q in chebyshev_shell_indices(idx)],
            dtype=np.float64,
        )
        kgt_np = np.asarray(self.k_parallel, dtype=np.float64).reshape(2)[None, :] + reciprocal
        kgt = cp.asarray(kgt_np, dtype=cp.float64)
        rho = cp.linalg.norm(kgt, axis=1)
        phi = cp.arctan2(kgt[:, 1], kgt[:, 0])
        gamma = cp.sqrt((float(self.k) * float(self.k) - rho * rho) + 0.0j)
        gamma = cp.where(gamma == 0.0, gamma + 1.0e-10j, gamma)
        out = _CupyReciprocalShell(kgt=kgt, rho=rho, phi=phi, gamma=gamma.astype(cp.complex128))
        self.reciprocal_cache[idx] = out
        return out

    def real_shell(self, shell: int) -> _CupyRealShell:
        idx = int(shell)
        cached = self.real_cache.get(idx)
        if cached is not None:
            return cached
        cp = self.cupy
        shifts_np = np.asarray(
            [p * self.lattice.a1 + q * self.lattice.a2 for p, q in chebyshev_shell_indices(idx)],
            dtype=np.float64,
        )
        shifts = cp.asarray(shifts_np, dtype=cp.float64)
        kp = cp.asarray(np.asarray(self.k_parallel, dtype=np.float64).reshape(2), dtype=cp.float64)
        phase_xy = cp.exp(1j * (shifts[:, :2] @ kp))
        out = _CupyRealShell(shifts=shifts, phase_xy=phase_xy.astype(cp.complex128))
        self.real_cache[idx] = out
        return out


def _fixed_shell_count(value: int | None, *, max_shells: int) -> int:
    """Return an inclusive shell cap for the fixed-range GPU evaluator."""
    if value is not None:
        return max(0, int(value))
    # The GPU slab evaluator currently uses a bounded fixed range when the
    # operator itself is configured for adaptive shell accumulation.  This keeps
    # the field path device-resident and matches the cheap eta-preflight range;
    # callers that need a stricter field basis can set explicit shell counts.
    return min(12, max(1, int(max_shells)))


def _ewald_structural_sums_shifted_fixed_cupy(
    *,
    relative_source_minus_point: Any,
    lmax_struct: int,
    k: float,
    eta: float,
    workspace: _CupyFixedShellWorkspace,
    real_shell_count: int,
    reciprocal_shell_count: int,
) -> Any:
    """Evaluate shifted scalar Ewald tables for flattened source/point pairs.

    Parameters
    ----------
    relative_source_minus_point:
        CuPy array of shape ``(n_pairs, 3)`` equal to ``source - point``.

    Notes
    -----
    The routine assumes shifted (non same-plane) pairs.  Callers fall back to
    the NumPy reference for rare batches containing exact ``z_source == z_point``
    pairs, where the same-plane formulas must be used.
    """
    cp = workspace.cupy
    c = cp.asarray(relative_source_minus_point, dtype=cp.float64).reshape(-1, 3)
    n_pairs = int(c.shape[0])
    order = 2 * int(lmax_struct)
    offset = order
    sums = cp.zeros((n_pairs, order + 1, 2 * order + 1), dtype=cp.complex128)
    if n_pairs == 0:
        return sums

    cxy = c[:, :2]
    cz = c[:, 2]

    # Reciprocal contribution.  This uses fixed shell ranges and performs no
    # per-shell convergence synchronization on the host.
    for shell in range(int(reciprocal_shell_count) + 1):
        shell_data = workspace.reciprocal_shell(shell)
        kgt = shell_data.kgt
        rho = shell_data.rho
        phi = shell_data.phi
        gamma = shell_data.gamma
        phase = cp.exp(-1j * (cxy @ kgt.T))  # (n_pairs, n_recip)
        delta_full = shifted_delta_sequence_cupy_batched(order, gamma, cz, float(eta), cupy=cp)
        gamma_over_k = gamma / float(k)
        exp_m_phi = {m: cp.exp(1j * m * phi) for m in range(-order, order + 1)}

        for degree in range(order + 1):
            for m in range(-degree, degree + 1):
                root = (
                    math.sqrt(2 * degree + 1.0)
                    * math.sqrt(factorial_int(degree - m))
                    * math.sqrt(factorial_int(degree + m))
                )
                prefactor = (
                    (-1j) ** m
                    * root
                    / (((-2.0) ** degree) * workspace.lattice.area * float(k) * float(k))
                )
                n_vals = np.arange(0, degree - abs(m) + 1, dtype=np.int64)
                if n_vals.size == 0:
                    continue
                acc = cp.zeros((n_pairs, rho.size), dtype=cp.complex128)
                for n in n_vals:
                    s_vals = np.arange(int(n), min(degree - abs(m), 2 * int(n)) + 1, dtype=np.int64)
                    s_vals = (
                        s_vals[s_vals % 2 == 1]
                        if (degree - abs(m)) % 2
                        else s_vals[s_vals % 2 == 0]
                    )
                    if s_vals.size == 0:
                        continue
                    terms = cp.zeros((n_pairs, rho.size), dtype=cp.complex128)
                    for s_val in s_vals:
                        denom = (
                            factorial_int(2 * int(n) - int(s_val))
                            * factorial_int(int(s_val) - int(n))
                            * factorial_int((degree + abs(m) - int(s_val)) // 2)
                            * factorial_int((degree - abs(m) - int(s_val)) // 2)
                        )
                        terms += (
                            (-float(k) * cz[:, None]) ** (2 * int(n) - int(s_val))
                            * (rho[None, :] / float(k)) ** (degree - int(s_val))
                            / denom
                        )
                    acc += (
                        gamma_over_k[None, :] ** (2 * int(n) - 1) * delta_full[:, :, int(n)] * terms
                    )
                vec = exp_m_phi[m][None, :] * acc
                sums[:, degree, m + offset] += (
                    structural_sum_m_normalization(m) * prefactor * cp.sum(phase * vec, axis=1)
                )

    # Real-space contribution.  Each pair accumulates over shell shifts by a
    # dense reduction, avoiding scatter-add entirely.
    for shell in range(int(real_shell_count) + 1):
        real_shell_data = workspace.real_shell(shell)
        shifted = -(real_shell_data.shifts[None, :, :] + c[:, None, :])
        radii = cp.linalg.norm(shifted, axis=2)
        mask = radii > 0.0
        radii_safe = cp.where(mask, radii, 1.0)
        ct = shifted[:, :, 2] / radii_safe
        st = cp.sqrt(cp.maximum(0.0, 1.0 - ct * ct))
        phi = cp.arctan2(shifted[:, :, 1], shifted[:, :, 0])
        plm = legendre_normalized_trigon(ct, st, max(1, order), xp=cp)
        phase_shell = real_shell_data.phase_xy[None, :]
        kz_r = float(k) * radii_safe
        for degree in range(order + 1):
            integral = (0.5) ** (degree + 1.5) * real_integral_sequence_cupy(
                degree,
                float(eta),
                float(k),
                radii_safe,
                cupy=cp,
            )
            radial = phase_shell * kz_r**degree * integral
            radial = cp.where(mask, radial, 0.0 + 0.0j)
            for m in range(-degree, degree + 1):
                angular = (
                    plm[degree, abs(m), :, :]
                    * cp.exp(1j * m * phi)
                    / structural_sum_m_normalization(m)
                )
                contrib = -1j * math.sqrt(2.0 / math.pi) * radial * angular
                sums[:, degree, m + offset] += structural_sum_m_normalization(m) * cp.sum(
                    contrib,
                    axis=1,
                )
    return sums


def _build_source_projection_kernels_cupy(
    *,
    lmax: int,
    coeffs: np.ndarray,
    cupy: Any,
) -> tuple[int, int, Any]:
    """Return source-specific ``l=1`` projection kernels on the GPU."""
    lmax_struct, _m_offset, kernel_np, _row_idx = _l1_projection_data(int(lmax))
    p_count = int(kernel_np.shape[3])
    # kernel: (row, coeff, m, p), coeffs: (source, coeff)
    source_kernel = np.einsum(
        "rcmp,jc->jrpm",
        np.asarray(kernel_np[:, :, :, :p_count], dtype=np.complex128),
        np.asarray(coeffs, dtype=np.complex128),
        optimize=True,
    )
    return int(lmax_struct), p_count, cupy.asarray(source_kernel, dtype=cupy.complex128)


def _fallback_numpy_l1_for_batch(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    periodic: Any,
    k_parallel: np.ndarray,
    eta: float,
    kernel: np.ndarray,
    lmax_struct: int,
) -> np.ndarray:
    """Reference fallback for rare same-plane batches."""
    from pyceles.core.periodic.ewald import EwaldShellWorkspace, ewald_structural_sums_2d_batch

    workspace = EwaldShellWorkspace(
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        eta=float(eta),
    )
    out = np.zeros((points.shape[0], 6), dtype=np.complex128)
    for j in range(positions.shape[0]):
        sums = ewald_structural_sums_2d_batch(
            lmax_struct=int(lmax_struct),
            k=float(k),
            destinations=np.asarray(points, dtype=float),
            source=np.asarray(positions[j], dtype=float),
            lattice=periodic.lattice,
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            eta=float(eta),
            real_shells=periodic.options.real_shells,
            reciprocal_shells=periodic.options.reciprocal_shells,
            shell_tolerance=float(periodic.options.shell_tolerance),
            max_shells=int(periodic.options.max_shells),
            workspace=workspace,
        )
        out += _reduce_structural_sums_to_l1(sums, coeffs[j], kernel=kernel)
    return out


def _resolve_eta(
    *,
    periodic: Any,
    k: float,
    k_parallel: np.ndarray,
    positions: np.ndarray,
    lmax: int,
) -> float:
    if periodic.options.eta is not None:
        return float(periodic.options.eta)
    return float(
        select_ewald_eta(
            lattice=periodic.lattice,
            k=float(k),
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            positions=np.asarray(positions, dtype=float).reshape(-1, 3),
            lmax=int(lmax),
            shell_tolerance=float(periodic.options.shell_tolerance),
            max_shells=int(periodic.options.max_shells),
            real_shells=periodic.options.real_shells,
            reciprocal_shells=periodic.options.reciprocal_shells,
        )
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
    It uses fixed shell ranges (``real_shells``/``reciprocal_shells`` when set,
    otherwise a bounded eta-preflight range) to avoid adaptive host/device
    synchronizations.
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
    _lmax_struct_np, _m_offset, kernel_np, _row_idx = _l1_projection_data(int(lmax))
    real_count = _fixed_shell_count(
        periodic.options.real_shells, max_shells=int(periodic.options.max_shells)
    )
    recip_count = _fixed_shell_count(
        periodic.options.reciprocal_shells,
        max_shells=int(periodic.options.max_shells),
    )
    if source_batch_size is None:
        # Keep the flattened pair count modest.  This is not a public tuning
        # parameter; it only bounds temporary GPU arrays.
        source_batch_size = max(1, min(pos.shape[0], 2048 // max(1, int(point_batch_size))))
    source_batch_size = max(1, int(source_batch_size))
    point_batch_size = max(1, int(point_batch_size))

    workspace = _CupyFixedShellWorkspace(
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
                # Exact same-plane pairs need the CPU same-plane Ewald formula.
                # They are rare for regular diagnostic slices with random particle
                # heights, but the fallback keeps this path safe.
                cz_cpu = src_batch[None, :, 2] - pts_batch[:, None, 2]
                if np.any(np.isclose(cz_cpu, 0.0, atol=0.0, rtol=0.0)):
                    out[p0:p1] += _fallback_numpy_l1_for_batch(
                        points=pts_batch,
                        positions=src_batch,
                        coeffs=coeff_arr[s0:s1],
                        lmax=int(lmax),
                        k=float(k),
                        periodic=periodic,
                        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
                        eta=float(eta),
                        kernel=np.asarray(kernel_np, dtype=np.complex128),
                        lmax_struct=int(lmax_struct),
                    )
                    if progress is not None:
                        progress.update(1)
                    continue

                rel = src_batch[None, :, :] - pts_batch[:, None, :]
                n_points = int(pts_batch.shape[0])
                n_sources = int(src_batch.shape[0])
                rel_cp = cp.asarray(rel.reshape(-1, 3), dtype=cp.float64)
                sums = _ewald_structural_sums_shifted_fixed_cupy(
                    relative_source_minus_point=rel_cp,
                    lmax_struct=int(lmax_struct),
                    k=float(k),
                    eta=float(eta),
                    workspace=workspace,
                    real_shell_count=int(real_count),
                    reciprocal_shell_count=int(recip_count),
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
