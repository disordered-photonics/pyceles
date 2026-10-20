"""Shared scalar helper kernels for periodic Ewald/direct structural sums."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
from scipy import special

from .special import _scaled_erfcx_product

Array = np.ndarray


def same_plane_z_tolerance(k: complex, *, coordinate_scale: float = 0.0) -> float:
    """Return the roundoff-scale height tolerance for same-plane Ewald routing.

    The shifted reciprocal recurrence is singular in the same-plane limit.  Grid
    construction and particle coordinates can differ by a few ulps even when a
    diagnostic slice is intended to pass exactly through a particle height, so
    those roundoff-sized offsets should use the same-plane formula instead of the
    shifted recurrence.
    """
    k_abs = abs(complex(k))
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


def reciprocal_gamma_with_zero_mask(k: complex, rho: Array) -> tuple[Array, Array]:
    """Return guarded reciprocal ``gamma`` and its exact Rayleigh-zero mask.

    The mask is captured before applying the established small imaginary guard.
    Near-plane stabilization must not mistake that guarded value for an ordinary
    nonzero reciprocal order and thereby hide the Rayleigh singular policy.
    """
    wave_number = complex(k)
    gamma = np.sqrt(wave_number * wave_number - np.asarray(rho, dtype=float) ** 2 + 0.0j)
    zero = np.asarray(gamma == 0.0, dtype=bool)
    gamma[zero] += 1.0e-10j
    return np.asarray(gamma, dtype=np.complex128), zero


def reciprocal_gamma(k: complex, rho: Array) -> Array:
    """Return reciprocal ``gamma = sqrt(k^2-rho^2)`` with the Ewald zero guard."""
    return reciprocal_gamma_with_zero_mask(k, rho)[0]


def scaled_real_integral_sequence(degree: int, eta: float, k: complex, radii: Array) -> Array:
    """Return the Ewald radial recurrence with its cancelling powers folded in.

    If ``I_j`` denotes the recurrence used by :func:`real_integral_sequence`,
    this helper returns ``(k*r/2) ** (degree + 1) * I_{degree + 1}`` directly.
    """
    l = int(degree)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if np.any(r <= 0.0):
        raise ValueError("Real-space Ewald radii must be positive.")

    wave_number = complex(k)
    eta_f = float(eta)
    if wave_number.imag != 0.0:
        return scaled_real_integral_sequences(l, eta_f, wave_number, r)[:, l]

    k_f = wave_number.real
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


def scaled_real_integral_sequences(max_degree: int, eta: float, k: complex, radii: Array) -> Array:
    """Evaluate all scaled radial degrees with shared complex128 seeds.

    The complex-host continuation uses even-order Kambe integrals. Both seeds
    are scaled erfc products, followed by one recurrence over the degree;
    radii are independent vector entries. Reflection avoids forming an
    overflowing erfcx value before multiplying by its cancelling exponential.
    """
    degree_max = int(max_degree)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if degree_max < 0:
        raise ValueError("max_degree must be nonnegative")
    if np.any(r <= 0.0):
        raise ValueError("Real-space Ewald radii must be positive.")
    wave_number = complex(k)
    if wave_number.imag == 0.0:
        return np.stack(
            [scaled_real_integral_sequence(l, eta, wave_number, r) for l in range(degree_max + 1)],
            axis=-1,
        )

    eta_scaled = math.sqrt(2.0) * float(eta) / wave_number
    z = wave_number * r
    exponential = np.exp((wave_number / (2.0 * float(eta))) ** 2 - (float(eta) * r) ** 2)
    minus = _scaled_erfcx_product(
        float(eta) * r - 0.5j * wave_number / float(eta), exponential, np.exp(-1j * z)
    )
    plus = _scaled_erfcx_product(
        float(eta) * r + 0.5j * wave_number / float(eta), exponential, np.exp(1j * z)
    )
    previous = math.sqrt(math.pi / 8.0) * (minus + plus) / z
    seed_m2 = -0.5j * math.sqrt(math.pi / 2.0) * (minus - plus)
    values = np.empty((r.size, degree_max + 1), dtype=np.complex128)
    values[:, 0] = math.sqrt(2.0) * z * previous
    if degree_max:
        current = (previous - seed_m2 + eta_scaled * exponential) / z**2
        values[:, 1] = math.sqrt(2.0) * z**2 * current
        for degree in range(2, degree_max + 1):
            following = (
                (2 * degree - 1) * current - previous + eta_scaled ** (2 * degree - 1) * exponential
            ) / z**2
            values[:, degree] = math.sqrt(2.0) * z ** (degree + 1) * following
            previous, current = current, following
    return values


def real_integral_sequence(degree: int, eta: float, k: complex, radii: Array) -> Array:
    """Evaluate the real-space integral sequence used by Ewald summands."""
    l = int(degree)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if np.any(r <= 0.0):
        raise ValueError("Real-space Ewald radii must be positive.")
    wave_number = complex(k)
    if wave_number.imag != 0.0:
        scaled = scaled_real_integral_sequence(l, eta, wave_number, r)
        return scaled / (wave_number * r / 2.0) ** (l + 1)

    k_real = wave_number.real
    alpha = k_real * k_real / (4.0 * float(eta) * float(eta))
    root_alpha = np.sqrt(alpha)
    w = special.wofz(root_alpha + 1j * k_real * r / (2.0 * root_alpha))
    exp_term = np.exp(alpha - (k_real * r) ** 2 / (4.0 * alpha))

    vals = np.zeros((r.size, l + 2), dtype=np.complex128)
    vals[:, 0] = math.sqrt(math.pi) * exp_term * w.imag
    vals[:, 1] = math.sqrt(math.pi) * 2.0 / (k_real * r) * exp_term * w.real
    for idx in range(2, l + 2):
        vals[:, idx] = (2.0 / (k_real * r)) ** 2 * (
            0.5 * float(2 * (idx - 2) + 1) * vals[:, idx - 1]
            - vals[:, idx - 2]
            + alpha ** (-(idx - 2) - 0.5) * exp_term
        )
    return vals[:, -1]
