"""Shared scalar helper kernels for periodic Ewald/direct structural sums."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
from scipy import special

Array = np.ndarray


def same_plane_z_tolerance(k: float, *, coordinate_scale: float = 0.0) -> float:
    """Return the roundoff-scale height tolerance for same-plane Ewald routing.

    The shifted reciprocal recurrence is singular in the same-plane limit.  Grid
    construction and particle coordinates can differ by a few ulps even when a
    diagnostic slice is intended to pass exactly through a particle height, so
    those roundoff-sized offsets should use the same-plane formula instead of the
    shifted recurrence.
    """
    k_abs = abs(float(k))
    wavelength_scale = 1.0 / k_abs if np.isfinite(k_abs) and k_abs > 0.0 else 1.0
    scale = max(1.0, wavelength_scale, abs(float(coordinate_scale)))
    return float(1024.0 * np.finfo(float).eps * scale)


def validate_shell_count(value: int, *, name: str) -> int:
    """Validate one nonnegative shell/window count."""
    shells = int(value)
    if shells < 0:
        raise ValueError(f"`{name}` must be >= 0. Got {value!r}.")
    return shells


def validate_optional_shell_count(value: int | None, *, name: str) -> int | None:
    """Validate one optional nonnegative shell count."""
    if value is None:
        return None
    return validate_shell_count(value, name=name)


def validate_shell_tolerance(value: float) -> float:
    """Validate the adaptive shell-accumulation tolerance."""
    tol = float(value)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"`shell_tolerance` must be finite and positive. Got {value!r}.")
    return tol


def chebyshev_shell_indices(shell: int) -> tuple[tuple[int, int], ...]:
    """Return index pairs on one rectangular-lattice Chebyshev shell."""
    s = int(shell)
    if s < 0:
        raise ValueError(f"`shell` must be >= 0. Got {shell!r}.")
    if s == 0:
        return ((0, 0),)
    out: list[tuple[int, int]] = []
    for p in (-s, s):
        for q in range(-s, s + 1):
            out.append((p, q))
    for p in range(-s + 1, s):
        for q in (-s, s):
            out.append((p, q))
    return tuple(out)


def factorial_int(value: int | np.integer) -> int:
    """Return factorial for one integer-like value."""
    return math.factorial(int(value))


@cache
def log_factorial(value: int | np.integer) -> float:
    """Return ``log(value!)`` without constructing the factorial integer."""
    return float(math.lgamma(float(int(value)) + 1.0))


@cache
def log_spherical_factorial_root(degree: int, order: int) -> float:
    """Return the log of the normalized spherical factorial root."""
    l = int(degree)
    m = int(order)
    if l < 0 or abs(m) > l:
        raise ValueError(f"Expected degree >= |order| >= 0, got {(degree, order)!r}.")
    return float(
        0.5 * math.log(2.0 * l + 1.0) + 0.5 * log_factorial(l - m) + 0.5 * log_factorial(l + m)
    )


def structural_sum_m_normalization(order: int) -> float:
    """Return pyceles's scalar structural-sum normalization for azimuthal order `M`."""
    m = int(order)
    if m >= 0:
        return math.sqrt(2.0 * math.pi) * ((-1.0) ** (-m))
    return math.sqrt(2.0 * math.pi)


def reciprocal_gamma_with_zero_mask(k: float, rho: Array) -> tuple[Array, Array]:
    """Return guarded reciprocal ``gamma`` and its exact Rayleigh-zero mask.

    The mask is captured before applying the established small imaginary guard.
    Near-plane stabilization must not mistake that guarded value for an ordinary
    nonzero reciprocal order and thereby hide the Rayleigh singular policy.
    """
    gamma = np.sqrt((float(k) * float(k) - np.asarray(rho, dtype=float) ** 2) + 0.0j)
    zero = np.asarray(gamma == 0.0, dtype=bool)
    gamma[zero] += 1.0e-10j
    return np.asarray(gamma, dtype=np.complex128), zero


def reciprocal_gamma(k: float, rho: Array) -> Array:
    """Return reciprocal ``gamma = sqrt(k^2-rho^2)`` with the Ewald zero guard."""
    return reciprocal_gamma_with_zero_mask(k, rho)[0]


def scaled_real_integral_sequence(degree: int, eta: float, k: float, radii: Array) -> Array:
    """Return the Ewald radial recurrence with its cancelling powers folded in.

    If ``I_j`` denotes the recurrence used by :func:`real_integral_sequence`,
    this helper returns ``(k*r/2) ** (degree + 1) * I_{degree + 1}`` directly.
    """
    l = int(degree)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if np.any(r <= 0.0):
        raise ValueError("Real-space Ewald radii must be positive.")

    k_f = float(k)
    eta_f = float(eta)
    x = k_f * r
    alpha = k_f * k_f / (4.0 * eta_f * eta_f)
    root_alpha = np.sqrt(alpha)
    wofz = special.wofz(root_alpha + 1j * x / (2.0 * root_alpha))
    exp_term = np.exp(alpha - x * x / (4.0 * alpha))

    # w_j = (x / 2)^j I_j. The first two values simplify exactly.
    w_prev2 = math.sqrt(math.pi) * exp_term * wofz.imag
    w_prev1 = math.sqrt(math.pi) * exp_term * wofz.real
    if l == 0:
        return np.asarray(w_prev1, dtype=np.float64)

    source = exp_term / root_alpha
    source_ratio = x / (2.0 * alpha)
    for idx in range(2, l + 2):
        current = ((2.0 * idx - 3.0) / x) * w_prev1 - w_prev2 + source
        w_prev2, w_prev1 = w_prev1, current
        source = source * source_ratio
    return np.asarray(w_prev1, dtype=np.float64)


def stable_free_space_minus_real_sequence(
    max_degree: int,
    eta: float,
    k: float,
    radii: Array,
    *,
    tail_steps: int = 24,
    direct_condition_limit: float = 1.0e6,
) -> Array:
    """Evaluate ``h_l^(1)(k*r) - R_l(r)`` without catastrophic cancellation.

    A direct ``h-R`` subtraction is most accurate while it is well conditioned.
    When the two terms become nearly equal, this routine switches degree by
    degree to a minimal-solution recurrence for the complement.  The stable
    recurrence is evolved as the complement-to-Hankel ratio; its roundoff
    homogeneous component appears as a high-order plateau and is removed before
    reconstructing the small complement.

    ``tail_steps`` controls the auxiliary high-order tail used to identify that
    plateau. ``direct_condition_limit`` is the largest estimated subtraction
    condition number for which the direct result is retained.  The helper is
    intended for Ewald preparation, not as a general spherical-Hankel evaluator.
    """
    degree_max = int(max_degree)
    if degree_max < 0:
        raise ValueError(f"`max_degree` must be >= 0. Got {max_degree!r}.")
    tail = int(tail_steps)
    if tail < 4:
        raise ValueError(f"`tail_steps` must be >= 4. Got {tail_steps!r}.")
    condition_limit = float(direct_condition_limit)
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError(
            "`direct_condition_limit` must be finite and greater than one. "
            f"Got {direct_condition_limit!r}."
        )
    r = np.asarray(radii, dtype=float).reshape(-1)
    if np.any(r <= 0.0):
        raise ValueError("Real-space Ewald radii must be positive.")
    k_f = float(k)
    eta_f = float(eta)
    if not np.isfinite(k_f) or k_f <= 0.0:
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    if not np.isfinite(eta_f) or eta_f <= 0.0:
        raise ValueError(f"`eta` must be finite and positive. Got {eta!r}.")

    x = k_f * r
    alpha = k_f * k_f / (4.0 * eta_f * eta_f)
    root_alpha = math.sqrt(alpha)
    wofz = special.wofz(root_alpha + 1j * x / (2.0 * root_alpha))
    exp_term = np.exp(alpha - x * x / (4.0 * alpha))
    source_base = exp_term / root_alpha
    source_ratio = x / (2.0 * alpha)

    w0 = math.sqrt(math.pi) * exp_term * wofz.imag
    w1 = math.sqrt(math.pi) * exp_term * wofz.real
    w2 = w1 / x - w0 + source_base

    stop = degree_max + tail
    hankel = np.empty((stop + 1, r.size), dtype=np.complex128)
    for degree in range(stop + 1):
        hankel[degree] = special.spherical_jn(degree, x) + 1j * special.spherical_yn(degree, x)
    if not np.all(np.isfinite(hankel)):
        raise FloatingPointError(
            "Stable same-plane Ewald complements require finite auxiliary "
            f"Hankel values through degree {stop}."
        )

    ratios = np.empty_like(hankel)
    ratios[0] = (hankel[0] + 1j * w1 / (math.sqrt(math.pi) * x)) / hankel[0]
    if stop == 0:
        return np.asarray(hankel[:1] * (ratios[:1] - ratios[0]), dtype=np.complex128)
    ratios[1] = (hankel[1] + 1j * w2 / (math.sqrt(math.pi) * x)) / hankel[1]
    for degree in range(2, stop + 1):
        hankel_ratio = hankel[degree - 2] / hankel[degree]
        forcing = (
            1j
            * source_base
            * source_ratio ** (degree - 1)
            / (math.sqrt(math.pi) * x * hankel[degree])
        )
        ratios[degree] = (
            ratios[degree - 1] + hankel_ratio * (ratios[degree - 1] - ratios[degree - 2]) + forcing
        )

    plateau = ratios[stop]
    stable = np.asarray(
        hankel[: degree_max + 1] * (ratios[: degree_max + 1] - plateau),
        dtype=np.complex128,
    )

    # Retain ordinary direct arithmetic wherever its own subtraction condition
    # is benign.  This preserves the validated low-order path and avoids asking
    # the minimal-solution recurrence to improve a result that was already safe.
    direct = np.empty_like(stable)
    real_terms = np.empty_like(stable)
    scaled_prev2 = w0
    scaled_prev1 = w1
    for degree in range(degree_max + 1):
        if degree == 0:
            scaled = scaled_prev1
        else:
            idx = degree + 1
            source = source_base * source_ratio ** (idx - 2)
            scaled = ((2.0 * idx - 3.0) / x) * scaled_prev1 - scaled_prev2 + source
            scaled_prev2, scaled_prev1 = scaled_prev1, scaled
        real_terms[degree] = -1j * scaled / (math.sqrt(math.pi) * x)
        direct[degree] = hankel[degree] - real_terms[degree]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        condition = (np.abs(hankel[: degree_max + 1]) + np.abs(real_terms)) / np.maximum(
            np.abs(direct), np.finfo(float).tiny
        )
    use_direct = np.isfinite(direct) & np.isfinite(condition) & (condition <= condition_limit)
    return np.asarray(np.where(use_direct, direct, stable), dtype=np.complex128)


def real_integral_sequence(degree: int, eta: float, k: float, radii: Array) -> Array:
    """Evaluate the real-space integral sequence used by Ewald summands."""
    l = int(degree)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if np.any(r <= 0.0):
        raise ValueError("Real-space Ewald radii must be positive.")
    alpha = float(k) * float(k) / (4.0 * float(eta) * float(eta))
    root_alpha = np.sqrt(alpha)
    w = special.wofz(root_alpha + 1j * float(k) * r / (2.0 * root_alpha))
    exp_term = np.exp(alpha - (float(k) * r) ** 2 / (4.0 * alpha))

    vals = np.zeros((r.size, l + 2), dtype=np.complex128)
    vals[:, 0] = math.sqrt(math.pi) * exp_term * w.imag
    vals[:, 1] = math.sqrt(math.pi) * 2.0 / (float(k) * r) * exp_term * w.real
    for idx in range(2, l + 2):
        vals[:, idx] = (2.0 / (float(k) * r)) ** 2 * (
            0.5 * float(2 * (idx - 2) + 1) * vals[:, idx - 1]
            - vals[:, idx - 2]
            + alpha ** (-(idx - 2) - 0.5) * exp_term
        )
    return vals[:, -1]
