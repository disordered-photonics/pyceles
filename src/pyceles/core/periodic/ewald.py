"""Ewald structural constants for rectangular two-dimensional lattices."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from pyceles.core.indexing import n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.spherical import legendre_normalized_trigon_scalar

from .scalar import (
    factorial_int,
    real_integral_sequence,
    reciprocal_gamma,
    square_shell_indices,
    structural_sum_m_normalization,
    upper_gamma_sequence,
    validate_shell_count,
)
from .special import shifted_delta_sequence, upper_incomplete_gamma_int_or_halfint
from .structural import block_from_structural_sums

Array = np.ndarray


def default_ewald_eta(lattice: RectangularLattice2D) -> float:
    """Return the default real/reciprocal Ewald split for a 2D unit cell."""
    return float(np.sqrt(np.pi / lattice.area))


def _same_plane_reciprocal_sum(
    degree: int,
    order: int,
    *,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int,
    c_xy: Array,
) -> complex:
    l = int(degree)
    m = int(order)
    if (l - abs(m)) % 2:
        return 0.0 + 0.0j
    root = math.sqrt(2 * l + 1) * math.sqrt(factorial_int(l - m)) * math.sqrt(factorial_int(l + m))
    prefactor = (1j) ** m * root / (lattice.area * float(k) * (2.0 * float(k)) ** l)

    kp0 = np.asarray(k_parallel, dtype=float).reshape(2)
    cxy = np.asarray(c_xy, dtype=float).reshape(2)
    acc = 0.0 + 0.0j
    for shell in range(validate_shell_count(shells, name="reciprocal_shells") + 1):
        indices = square_shell_indices(shell)
        reciprocal = np.asarray(
            [p * lattice.b1 + q * lattice.b2 for p, q in indices],
            dtype=float,
        )
        kgt = kp0 + reciprocal
        rho = np.linalg.norm(kgt, axis=1)
        phi = np.arctan2(kgt[:, 1], kgt[:, 0])
        gamma = reciprocal_gamma(float(k), rho)
        gamma_arg = -(gamma * gamma) / (4.0 * float(eta) * float(eta))
        n_values = np.arange((l - abs(m)) // 2 + 1, dtype=np.int64)
        gamma_fun = upper_gamma_sequence(int(n_values[-1]), gamma_arg)
        inner = np.zeros_like(gamma, dtype=np.complex128)
        for n in n_values:
            denom = (
                factorial_int(n) * factorial_int((l + m) // 2 - n) * factorial_int((l - m) // 2 - n)
            )
            inner += (
                gamma_fun[:, int(n)] * gamma ** (2 * int(n) - 1) * rho ** (l - 2 * int(n)) / denom
            )
        acc += np.sum(np.exp(-1j * (kgt @ cxy)) * np.exp(1j * m * phi) * inner)
    return complex(prefactor * acc)


def _same_plane_real_sum(
    degree: int,
    order: int,
    *,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int,
) -> complex:
    l = int(degree)
    m = int(order)
    if (l - abs(m)) % 2:
        return 0.0 + 0.0j
    if shells <= 0:
        return 0.0 + 0.0j

    frac = (
        -1j
        * ((-1.0) ** ((l + m) // 2))
        / (2.0 ** (l + 1) * math.pi * factorial_int((l - m) // 2) * factorial_int((l + m) // 2))
    )
    root = math.sqrt(2 * l + 1) * math.sqrt(factorial_int(l - m)) * math.sqrt(factorial_int(l + m))
    prefactor = frac * root

    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    acc = 0.0 + 0.0j
    for shell in range(1, validate_shell_count(shells, name="real_shells") + 1):
        indices = square_shell_indices(shell)
        shifts_xy = np.asarray(
            [(p * lattice.a1 + q * lattice.a2)[:2] for p, q in indices],
            dtype=float,
        )
        radii = np.linalg.norm(shifts_xy, axis=1)
        phi = np.arctan2(shifts_xy[:, 1], shifts_xy[:, 0])
        integral = (float(k) * float(k) / 4.0) ** (l + 0.5) * real_integral_sequence(
            l,
            float(eta),
            float(k),
            radii,
        )
        acc += np.sum(
            np.exp(1j * (shifts_xy @ kp + m * (phi + math.pi)))
            / float(k)
            * (2.0 * radii / float(k)) ** l
            * integral
        )
    return complex(prefactor * acc)


def _self_correction(k: float, eta: float) -> complex:
    """Return the same-particle Ewald central-point correction."""
    eta_f = float(eta)
    x = -(float(k) * float(k)) / (4.0 * eta_f * eta_f)
    return complex(upper_incomplete_gamma_int_or_halfint(-0.5, x) / (4.0 * math.pi))


def _shifted_reciprocal_sum(
    degree: int,
    order: int,
    *,
    rvec: Array,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int,
) -> complex:
    l = int(degree)
    m = int(order)
    c = -np.asarray(rvec, dtype=float).reshape(3)
    if c[2] == 0.0:
        return _same_plane_reciprocal_sum(
            l,
            m,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=float(eta),
            shells=int(shells),
            c_xy=c[:2],
        )

    root = math.sqrt(2 * l + 1) * math.sqrt(factorial_int(l - m)) * math.sqrt(factorial_int(l + m))
    prefactor = (-1j) ** m * root / (((-2.0) ** l) * lattice.area * float(k) * float(k))

    kp0 = np.asarray(k_parallel, dtype=float).reshape(2)
    acc = 0.0 + 0.0j
    for shell in range(validate_shell_count(shells, name="reciprocal_shells") + 1):
        indices = square_shell_indices(shell)
        reciprocal = np.asarray(
            [p * lattice.b1 + q * lattice.b2 for p, q in indices],
            dtype=float,
        )
        kgt = kp0 + reciprocal
        rho = np.linalg.norm(kgt, axis=1)
        phi = np.arctan2(kgt[:, 1], kgt[:, 0])
        gamma = reciprocal_gamma(float(k), rho)
        n_values = np.arange(0, l - abs(m) + 1, dtype=np.int64)
        inner = np.zeros((rho.size, n_values.size), dtype=np.complex128)
        for n in n_values:
            s_values = np.arange(int(n), min(l - abs(m), 2 * int(n)) + 1, dtype=np.int64)
            if (l - abs(m)) % 2:
                s_values = s_values[s_values % 2 == 1]
            else:
                s_values = s_values[s_values % 2 == 0]
            if s_values.size == 0:
                continue
            terms = np.zeros_like(rho, dtype=np.complex128)
            for s in s_values:
                denom = (
                    factorial_int(2 * int(n) - int(s))
                    * factorial_int(int(s) - int(n))
                    * factorial_int((l + abs(m) - int(s)) // 2)
                    * factorial_int((l - abs(m) - int(s)) // 2)
                )
                terms += (
                    (-float(k) * c[2]) ** (2 * int(n) - int(s))
                    * (rho / float(k)) ** (l - int(s))
                    / denom
                )
            inner[:, int(n)] = terms
        delta = shifted_delta_sequence(int(n_values[-1]), gamma, float(c[2]), float(eta))
        acc += np.sum(
            np.exp(-1j * (kgt @ c[:2]))
            * np.exp(1j * m * phi)
            * np.sum((gamma / float(k))[:, None] ** (2 * n_values - 1) * delta * inner, axis=1)
        )
    return complex(prefactor * acc)


def _shifted_real_sum(
    degree: int,
    order: int,
    *,
    rvec: Array,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int,
) -> complex:
    l = int(degree)
    m = int(order)
    c = -np.asarray(rvec, dtype=float).reshape(3)
    if c[2] == 0.0 and (l - abs(m)) % 2:
        return 0.0 + 0.0j

    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    acc = 0.0 + 0.0j
    for shell in range(validate_shell_count(shells, name="real_shells") + 1):
        indices = square_shell_indices(shell)
        shifts = np.asarray([p * lattice.a1 + q * lattice.a2 for p, q in indices], dtype=float)
        shifted = -(shifts + c)
        radii = np.linalg.norm(shifted, axis=1)
        mask = radii > 0.0
        if not np.any(mask):
            continue
        shifts_xy = shifts[mask, :2]
        shifted = shifted[mask]
        radii = radii[mask]
        ct = shifted[:, 2] / radii
        st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
        phi = np.arctan2(shifted[:, 1], shifted[:, 0])

        angular = np.zeros_like(radii, dtype=np.complex128)
        for idx, (ct_i, st_i, phi_i) in enumerate(zip(ct, st, phi, strict=True)):
            plm = legendre_normalized_trigon_scalar(float(ct_i), float(st_i), max(1, l))
            angular[idx] = (
                plm[l, abs(m)] * np.exp(1j * m * float(phi_i)) / structural_sum_m_normalization(m)
            )

        integral = (0.5) ** (l + 1.5) * real_integral_sequence(
            l,
            float(eta),
            float(k),
            radii,
        )
        acc += np.sum(np.exp(1j * (shifts_xy @ kp)) * (float(k) * radii) ** l * angular * integral)
    return complex(-1j * math.sqrt(2.0 / math.pi) * acc)


def ewald_structural_constant_2d(
    degree: int,
    order: int,
    *,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int,
    reciprocal_shells: int,
    exclude_zero_shift: bool = False,
) -> complex:
    """Evaluate one pyceles-normalized scalar Ewald structural constant.

    The constant matches the scalar table consumed by
    `block_from_structural_sums`: degree `L`, order `M` stores the lattice sum
    of `h_L^(1)(k r) P_L^|M|(cos theta) exp(i M phi)` with the Bloch phase
    convention used by the periodic direct-sum oracle.
    """
    l = int(degree)
    m = int(order)
    if l < 0:
        raise ValueError(f"`degree` must be >= 0. Got {degree!r}.")
    if abs(m) > l:
        raise ValueError(f"`order` must satisfy |order| <= degree. Got {(degree, order)!r}.")
    eta_f = float(eta)
    if not np.isfinite(eta_f) or eta_f <= 0.0:
        raise ValueError(f"`eta` must be finite and positive. Got {eta!r}.")
    rvec = np.asarray(destination, dtype=float).reshape(3) - np.asarray(
        source, dtype=float
    ).reshape(3)
    is_self = bool(exclude_zero_shift) and float(np.linalg.norm(rvec)) == 0.0
    if is_self:
        value = _same_plane_reciprocal_sum(
            l,
            m,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=int(reciprocal_shells),
            c_xy=np.zeros(2, dtype=float),
        ) + _same_plane_real_sum(
            l,
            m,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=int(real_shells),
        )
        if l == 0:
            value += _self_correction(float(k), eta_f)
    else:
        value = _shifted_reciprocal_sum(
            l,
            m,
            rvec=rvec,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=int(reciprocal_shells),
        ) + _shifted_real_sum(
            l,
            m,
            rvec=rvec,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=int(real_shells),
        )
    return complex(structural_sum_m_normalization(m) * value)


def ewald_structural_sums_2d(
    *,
    lmax: int,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int,
    reciprocal_shells: int,
    exclude_zero_shift: bool = False,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Evaluate a scalar structural table using rectangular-lattice Ewald sums."""
    out_dtype = np.dtype(dtype)
    order = 2 * int(lmax)
    if order < 0:
        raise ValueError(f"`lmax` must be >= 0. Got {lmax!r}.")
    sums = np.zeros((order + 1, 2 * order + 1), dtype=np.complex128)
    offset = order
    for degree in range(order + 1):
        for m in range(-degree, degree + 1):
            sums[degree, m + offset] = ewald_structural_constant_2d(
                degree,
                m,
                k=float(k),
                destination=destination,
                source=source,
                lattice=lattice,
                k_parallel=k_parallel,
                eta=float(eta),
                real_shells=int(real_shells),
                reciprocal_shells=int(reciprocal_shells),
                exclude_zero_shift=bool(exclude_zero_shift),
            )
    return np.asarray(sums, dtype=out_dtype)


def periodic_ewald_block(
    *,
    lmax: int,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int,
    reciprocal_shells: int,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    exclude_zero_shift: bool = False,
) -> Array:
    """Assemble one periodic SVWF coupling block from Ewald structural constants."""
    sums = ewald_structural_sums_2d(
        lmax=int(lmax),
        k=float(k),
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=k_parallel,
        eta=float(eta),
        real_shells=int(real_shells),
        reciprocal_shells=int(reciprocal_shells),
        exclude_zero_shift=bool(exclude_zero_shift),
        dtype=np.complex128,
    )
    return block_from_structural_sums(
        lmax=int(lmax),
        structural_sums=sums,
        ab5=ab5,
        dtype=dtype,
    )


def apply_periodic_ewald_sum(
    *,
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int,
    reciprocal_shells: int,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    block_cache: dict[tuple[int, int], Array] | None = None,
) -> Array:
    """Apply the Ewald Bloch image sum to stacked SVWF coefficients."""
    out_dtype = np.dtype(dtype)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    ns = pos.shape[0]
    nm = n_modes(int(lmax))
    arr = np.asarray(x, dtype=out_dtype).reshape(ns, nm)
    y = np.zeros_like(arr, dtype=out_dtype)

    for i in range(ns):
        for j in range(ns):
            key = (i, j)
            wij = block_cache.get(key) if block_cache is not None else None
            if wij is None:
                wij = periodic_ewald_block(
                    lmax=int(lmax),
                    k=float(k),
                    destination=pos[i],
                    source=pos[j],
                    lattice=lattice,
                    k_parallel=k_parallel,
                    eta=float(eta),
                    real_shells=int(real_shells),
                    reciprocal_shells=int(reciprocal_shells),
                    ab5=ab5,
                    dtype=out_dtype,
                    exclude_zero_shift=(i == j),
                )
                if block_cache is not None:
                    block_cache[key] = wij
            y[i] += wij @ arr[j]
    return y.reshape(ns * nm)


__all__ = [
    "apply_periodic_ewald_sum",
    "default_ewald_eta",
    "ewald_structural_constant_2d",
    "ewald_structural_sums_2d",
    "periodic_ewald_block",
]
