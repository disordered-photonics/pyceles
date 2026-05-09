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
from pyceles.core.periodic.ewald import resolve_ewald_eta
from pyceles.core.periodic.scalar import (
    chebyshev_shell_indices,
    factorial_int,
    same_plane_z_tolerance,
    structural_sum_m_normalization,
    upper_gamma_sequence,
)
from pyceles.core.periodic.special_cupy import (
    real_integral_sequence_cupy,
    shifted_delta_sequence_cupy_batched,
)
from pyceles.core.spherical import legendre_normalized_trigon

from .periodic_interior import _l1_projection_data


@dataclass
class _CupyReciprocalShell:
    kgt: Any
    rho: Any
    phi: Any
    gamma: Any
    xarg: Any


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
    upper_gamma_cache: dict[tuple[int, int], Any] = field(default_factory=dict)

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
        gamma = gamma.astype(cp.complex128)
        xarg = -(gamma * gamma) / (4.0 * float(self.eta) * float(self.eta))
        out = _CupyReciprocalShell(
            kgt=kgt,
            rho=rho,
            phi=phi,
            gamma=gamma,
            xarg=xarg.astype(cp.complex128),
        )
        self.reciprocal_cache[idx] = out
        return out

    def upper_gamma(self, shell: int, max_index: int) -> Any:
        """Return same-plane upper-gamma sequence on the active device."""
        key = (int(shell), int(max_index))
        cached = self.upper_gamma_cache.get(key)
        if cached is not None:
            return cached
        shell_data = self.reciprocal_shell(int(shell))
        # There is no cheap CuPy equivalent for the half-integer/integer branch
        # helper used by the NumPy reference.  This quantity depends only on the
        # reciprocal shell and eta, not on field points or sources, so computing
        # it once on the host and staging it on the device avoids a per-pair CPU
        # fallback while keeping the CuPy same-plane path complete.
        out_np = upper_gamma_sequence(int(max_index), self.cupy.asnumpy(shell_data.xarg))
        out = self.cupy.asarray(out_np, dtype=self.cupy.complex128)
        self.upper_gamma_cache[key] = out
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
    coordinate_scale: float = 0.0,
) -> Any:
    """Evaluate scalar Ewald tables for flattened source/point pairs on CuPy.

    ``relative_source_minus_point`` has shape ``(n_pairs, 3)`` and equals
    ``source - point``.  The routine handles same-plane and roundoff-level
    same-plane pairs on the GPU rather than falling back to the NumPy reference.
    Same-plane reciprocal terms require an upper-gamma sequence; that sequence
    depends only on the reciprocal shell and eta, so it is computed once on the
    host and staged on the device by the workspace.
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
    same_plane_atol = same_plane_z_tolerance(float(k), coordinate_scale=float(coordinate_scale))
    cz_raw = c[:, 2]
    same_plane = cp.abs(cz_raw) <= float(same_plane_atol)
    cz = cp.where(same_plane, 0.0, cz_raw)
    has_same = bool(cp.any(same_plane).get())
    if has_same:
        c = c.copy()
        c[:, 2] = cz
    has_shifted = bool(cp.any(~same_plane).get())
    same_idx = cp.nonzero(same_plane)[0] if has_same else None
    shifted_idx = cp.nonzero(~same_plane)[0] if has_shifted else None

    # Reciprocal contribution.  This uses fixed shell ranges and performs no
    # adaptive per-shell convergence synchronization on the host.
    max_same_n = max(0, order // 2)
    for shell in range(int(reciprocal_shell_count) + 1):
        shell_data = workspace.reciprocal_shell(shell)
        kgt = shell_data.kgt
        rho = shell_data.rho
        phi = shell_data.phi
        gamma = shell_data.gamma
        phase_all = cp.exp(-1j * (cxy @ kgt.T))  # (n_pairs, n_recip)
        exp_m_phi = {m: cp.exp(1j * m * phi) for m in range(-order, order + 1)}

        if has_same:
            phase_same = phase_all[same_idx]
            gamma_fun = workspace.upper_gamma(shell, max_same_n)
            for degree in range(order + 1):
                for m in range(-degree, degree + 1):
                    if (degree - abs(m)) % 2:
                        continue
                    root = (
                        math.sqrt(2 * degree + 1.0)
                        * math.sqrt(factorial_int(degree - m))
                        * math.sqrt(factorial_int(degree + m))
                    )
                    prefactor = (
                        (1j) ** m
                        * root
                        / (workspace.lattice.area * float(k) * (2.0 * float(k)) ** degree)
                    )
                    n_vals = np.arange((degree - abs(m)) // 2 + 1, dtype=np.int64)
                    inner = cp.zeros_like(gamma, dtype=cp.complex128)
                    for n in n_vals:
                        denom = (
                            factorial_int(n)
                            * factorial_int((degree + m) // 2 - n)
                            * factorial_int((degree - m) // 2 - n)
                        )
                        inner += (
                            gamma_fun[:, int(n)]
                            * gamma ** (2 * int(n) - 1)
                            * rho ** (degree - 2 * int(n))
                            / denom
                        )
                    vec = exp_m_phi[m] * inner
                    vals = structural_sum_m_normalization(m) * prefactor * (phase_same @ vec)
                    sums[same_idx, degree, m + offset] = sums[same_idx, degree, m + offset] + vals

        if has_shifted:
            phase = phase_all[shifted_idx]
            cz_shifted = cz[shifted_idx]
            delta_full = shifted_delta_sequence_cupy_batched(
                order,
                gamma,
                cz_shifted,
                float(eta),
                cupy=cp,
            )
            gamma_over_k = gamma / float(k)
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
                    acc = cp.zeros((cz_shifted.size, rho.size), dtype=cp.complex128)
                    for n in n_vals:
                        s_vals = np.arange(
                            int(n), min(degree - abs(m), 2 * int(n)) + 1, dtype=np.int64
                        )
                        s_vals = (
                            s_vals[s_vals % 2 == 1]
                            if (degree - abs(m)) % 2
                            else s_vals[s_vals % 2 == 0]
                        )
                        if s_vals.size == 0:
                            continue
                        terms = cp.zeros((cz_shifted.size, rho.size), dtype=cp.complex128)
                        for s_val in s_vals:
                            denom = (
                                factorial_int(2 * int(n) - int(s_val))
                                * factorial_int(int(s_val) - int(n))
                                * factorial_int((degree + abs(m) - int(s_val)) // 2)
                                * factorial_int((degree - abs(m) - int(s_val)) // 2)
                            )
                            terms += (
                                (-float(k) * cz_shifted[:, None]) ** (2 * int(n) - int(s_val))
                                * (rho[None, :] / float(k)) ** (degree - int(s_val))
                                / denom
                            )
                        acc += (
                            gamma_over_k[None, :] ** (2 * int(n) - 1)
                            * delta_full[:, :, int(n)]
                            * terms
                        )
                    vec = exp_m_phi[m][None, :] * acc
                    vals = (
                        structural_sum_m_normalization(m) * prefactor * cp.sum(phase * vec, axis=1)
                    )
                    sums[shifted_idx, degree, m + offset] = (
                        sums[shifted_idx, degree, m + offset] + vals
                    )

    # Real-space contribution.  Each pair accumulates over shell shifts by a
    # dense reduction, avoiding scatter-add entirely.  Same-plane odd-parity
    # terms are zeroed per pair (not per batch), which matters for xz/yz slices
    # that pass through one particle height while also containing shifted rows.
    same_plane_pair = same_plane[:, None]
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
                if (degree - abs(m)) % 2:
                    contrib = cp.where(same_plane_pair, 0.0 + 0.0j, contrib)
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
