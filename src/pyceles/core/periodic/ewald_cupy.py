"""CuPy helpers for fixed-shell two-dimensional periodic Ewald sums.

The NumPy periodic operator can adapt shell-by-shell because all convergence
checks happen on the host.  GPU paths need fixed shell ranges to avoid a
host/device synchronization after every shell.  This module owns the shared
CuPy structural-sum evaluator used by periodic postprocessing and by the
periodic CuPy coupling operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyceles.core.lattice import RectangularLattice2D
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


@dataclass
class CupyReciprocalShell:
    """Device-side reciprocal-shell data for one Chebyshev-index shell."""

    kgt: Any
    rho: Any
    phi: Any
    gamma: Any
    xarg: Any


@dataclass
class CupyRealShell:
    """Device-side real-lattice shell data for one Chebyshev-index shell."""

    shifts: Any
    phase_xy: Any


@dataclass
class CupyEwaldShellWorkspace:
    """Device-side non-pair metadata cache for one periodic Ewald configuration."""

    cupy: Any
    lattice: RectangularLattice2D
    k: float
    k_parallel: np.ndarray
    eta: float
    reciprocal_cache: dict[int, CupyReciprocalShell] = field(default_factory=dict)
    real_cache: dict[int, CupyRealShell] = field(default_factory=dict)
    upper_gamma_cache: dict[tuple[int, int], Any] = field(default_factory=dict)

    def reciprocal_shell(self, shell: int) -> CupyReciprocalShell:
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
        gamma = cp.where(gamma == 0.0, gamma + 1.0e-10j, gamma).astype(cp.complex128)
        xarg = -(gamma * gamma) / (4.0 * float(self.eta) * float(self.eta))
        out = CupyReciprocalShell(
            kgt=kgt,
            rho=rho,
            phi=phi,
            gamma=gamma,
            xarg=xarg.astype(cp.complex128),
        )
        self.reciprocal_cache[idx] = out
        return out

    def upper_gamma(self, shell: int, max_index: int) -> Any:
        """Return the same-plane upper-gamma sequence on the active device."""
        key = (int(shell), int(max_index))
        cached = self.upper_gamma_cache.get(key)
        if cached is not None:
            return cached
        cp = self.cupy
        shell_data = self.reciprocal_shell(int(shell))
        # The half-integer/integer upper-gamma branch helper remains the NumPy
        # reference implementation.  It depends only on reciprocal shell and eta,
        # so stage it once instead of invoking the host in the hot pair loop.
        out_np = upper_gamma_sequence(int(max_index), cp.asnumpy(shell_data.xarg))
        out = cp.asarray(out_np, dtype=cp.complex128)
        self.upper_gamma_cache[key] = out
        return out

    def real_shell(self, shell: int) -> CupyRealShell:
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
        phase_xy = cp.exp(1j * (shifts[:, :2] @ kp)).astype(cp.complex128)
        out = CupyRealShell(shifts=shifts, phase_xy=phase_xy)
        self.real_cache[idx] = out
        return out


def ewald_structural_sums_2d_fixed_cupy(
    *,
    relative_source_minus_destination: Any,
    lmax_struct: int,
    k: float,
    eta: float,
    workspace: CupyEwaldShellWorkspace,
    real_shell_count: int,
    reciprocal_shell_count: int,
    coordinate_scale: float = 0.0,
) -> Any:
    """Evaluate scalar periodic Ewald tables for source/destination pairs.

    Parameters
    ----------
    relative_source_minus_destination:
        CuPy-compatible array with shape ``(n_pairs, 3)`` storing
        ``source - destination``.  This sign convention matches the NumPy
        structural-sum batch helper and the periodic near-field evaluator.
    lmax_struct:
        Structural multipole order.  Coupling blocks use the particle ``lmax``;
        local-field projection may pass a smaller order selected by the output
        projection kernel.

    Notes
    -----
    The returned table does not include the same-particle central-point Ewald
    correction.  Operator callers patch that correction only for true
    source==destination self blocks; point-field evaluators must not add it.
    """
    cp = workspace.cupy
    c = cp.asarray(relative_source_minus_destination, dtype=cp.float64).reshape(-1, 3)
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

    max_same_n = max(0, order // 2)
    for shell in range(int(reciprocal_shell_count) + 1):
        shell_data = workspace.reciprocal_shell(shell)
        kgt = shell_data.kgt
        rho = shell_data.rho
        phi = shell_data.phi
        gamma = shell_data.gamma
        phase_all = cp.exp(-1j * (cxy @ kgt.T))
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


__all__ = [
    "CupyEwaldShellWorkspace",
    "CupyRealShell",
    "CupyReciprocalShell",
    "ewald_structural_sums_2d_fixed_cupy",
]
