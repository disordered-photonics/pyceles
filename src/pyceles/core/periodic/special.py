"""Special-function helpers for two-dimensional periodic lattice sums."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import special

Array = np.ndarray

_KAMBE_SERIES_TERMS = 32
_SMALL_COMPLEX = 1e-14


def _integer_or_half_integer_twice(value: float) -> int:
    twice_float = 2.0 * float(value)
    twice = round(twice_float)
    if not np.isclose(twice_float, float(twice), rtol=0.0, atol=1e-12):
        raise ValueError(f"Order must be integer or half-integer. Got {value!r}.")
    return twice


def _branch_complex(z: complex) -> complex:
    zc = complex(z)
    if zc.imag == 0.0 and zc.real < 0.0:
        return complex(zc.real, -0.0)
    return zc


def upper_incomplete_gamma_int_or_halfint(order: float, z: complex) -> complex:
    """Return the upper incomplete gamma function on the Ewald branch.

    The implementation covers the integer and half-integer orders used by the
    periodic scalar sums and keeps the negative-real-axis branch explicit by
    evaluating exactly real negative inputs from below.
    """
    twice = _integer_or_half_integer_twice(float(order))
    zc = _branch_complex(z)
    a = 0.5 * float(twice)
    if zc == 0.0:
        if twice <= 0:
            return complex(np.inf)
        return complex(math.gamma(a))
    if twice == 0:
        return complex(special.exp1(zc))
    if twice == 2:
        return complex(np.exp(-zc))
    if twice == 1:
        return complex(math.sqrt(math.pi) * special.erfc(np.sqrt(zc)))
    if twice > 2:
        return complex(
            (a - 1.0) * upper_incomplete_gamma_int_or_halfint(a - 1.0, zc)
            + zc ** (a - 1.0) * np.exp(-zc)
        )
    return complex((upper_incomplete_gamma_int_or_halfint(a + 1.0, zc) - zc**a * np.exp(-zc)) / a)


def reduced_incomplete_gamma_int_or_halfint(order: float, z: complex) -> complex:
    """Return ``Gamma(order, z) / (-z)**order`` with the Ewald branch convention."""
    zc = _branch_complex(z)
    return upper_incomplete_gamma_int_or_halfint(order, zc) / ((-zc) ** float(order))


def _kambe_positive_z(z: complex) -> complex:
    zc = complex(z)
    return -zc if zc.real < 0.0 else zc


def _kambe_exp_factor(z: complex, eta: complex) -> complex:
    return complex(np.exp(0.5 * (-z * z * eta * eta + 1.0 / (eta * eta))))


def _kambe_m2(z: complex, eta: complex) -> complex:
    zc = _kambe_positive_z(z)
    root_half = 1.0 / math.sqrt(2.0)
    a_minus = (zc * eta - 1j / eta) * root_half
    a_plus = (zc * eta + 1j / eta) * root_half
    exp_factor = _kambe_exp_factor(zc, eta)
    return complex(
        -0.5j
        * math.sqrt(math.pi / 2.0)
        * exp_factor
        * (special.erfcx(a_minus) - special.erfcx(a_plus))
    )


def _kambe_0(z: complex, eta: complex) -> complex:
    zc = _kambe_positive_z(z)
    root_half = 1.0 / math.sqrt(2.0)
    a_minus = (zc * eta - 1j / eta) * root_half
    a_plus = (zc * eta + 1j / eta) * root_half
    exp_factor = _kambe_exp_factor(zc, eta)
    return complex(
        math.sqrt(math.pi / 8.0)
        * exp_factor
        * (special.erfcx(a_minus) + special.erfcx(a_plus))
        / zc
    )


def _kambe_m1(z: complex, eta: complex) -> complex:
    acc = 0.0 + 0.0j
    mult = z * z * 0.25
    macc = 0.5 + 0.0j
    arg = z * z * eta * eta * 0.5
    for mu in range(_KAMBE_SERIES_TERMS):
        acc += macc * upper_incomplete_gamma_int_or_halfint(float(-mu), arg)
        macc *= mult / float(mu + 1)
    return complex(acc)


def _kambe_m3(z: complex, eta: complex) -> complex:
    acc = 0.0 + 0.0j
    mult = z * z * 0.25
    macc = mult
    arg = z * z * eta * eta * 0.5
    for mu in range(_KAMBE_SERIES_TERMS):
        acc += macc * upper_incomplete_gamma_int_or_halfint(float(-1 - mu), arg)
        macc *= mult / float(mu + 1)
    return complex(acc)


def kambe_integral(order: int, z: complex, eta: complex) -> complex:
    """Evaluate the Kambe real-space Ewald integral by recurrence.

    This private helper evaluates
    ``integral_eta^inf t**order * exp(-z**2*t**2/2 + 1/(2*t**2)) dt``.
    It is used as the compact scalar-integral layer beneath periodic lattice
    sums; it is not part of the public pyceles API.
    """
    n = int(order)
    if n != order:
        raise ValueError(f"`order` must be an integer. Got {order!r}.")
    zc = complex(z)
    eta_c = complex(eta)
    if eta_c == 0.0:
        return complex(np.inf)
    cache: dict[int, complex] = {}

    def eval_order(idx: int) -> complex:
        cached = cache.get(idx)
        if cached is not None:
            return cached
        if zc == 0.0 and idx == -4:
            value = -eval_order(-2) + np.exp(0.5 / (eta_c * eta_c)) / eta_c
        elif zc == 0.0 and idx == -3:
            value = np.exp(0.5 / (eta_c * eta_c)) - 1.0
        elif zc == 0.0 and idx > -2:
            value = complex(np.inf)
        elif idx == -3:
            value = _kambe_m3(zc, eta_c)
        elif idx == -2:
            value = _kambe_m2(zc, eta_c)
        elif idx == -1:
            value = _kambe_m1(zc, eta_c)
        elif idx == 0:
            value = _kambe_0(zc, eta_c)
        elif idx == 1:
            value = (_kambe_exp_factor(zc, eta_c) - _kambe_m3(zc, eta_c)) / (zc * zc)
        elif idx == 2:
            value = (
                eval_order(0) - _kambe_m2(zc, eta_c) + eta_c * _kambe_exp_factor(zc, eta_c)
            ) / (zc * zc)
        elif idx == 3:
            value = (
                2.0 * eval_order(1)
                - _kambe_m1(zc, eta_c)
                + eta_c * eta_c * _kambe_exp_factor(zc, eta_c)
            ) / (zc * zc)
        elif idx < -3:
            value = (
                (idx + 3.0) * eval_order(idx + 2)
                - zc * zc * eval_order(idx + 4)
                + eta_c ** (idx + 3) * _kambe_exp_factor(zc, eta_c)
            )
        else:
            value = (
                (idx - 1.0) * eval_order(idx - 2)
                - eval_order(idx - 4)
                + eta_c ** (idx - 1) * _kambe_exp_factor(zc, eta_c)
            ) / (zc * zc)
        cache[idx] = complex(value)
        return cache[idx]

    return eval_order(n)


def shifted_delta_sequence(
    max_order: int,
    gamma: npt.ArrayLike,
    z_offset: float,
    eta: float,
    *,
    singular_atol: float = _SMALL_COMPLEX,
) -> Array:
    """Evaluate shifted reciprocal-space Ewald integrals for nonzero height offsets.

    This helper covers the off-plane particle-pair path. Same-plane or nearly
    same-plane cases should use the unshifted reciprocal formula instead of
    forcing the shifted recurrence into its unstable small-offset limit.
    """
    n_max = int(max_order)
    if n_max < 0:
        raise ValueError(f"`max_order` must be >= 0. Got {max_order!r}.")
    eta_f = float(eta)
    if not np.isfinite(eta_f) or eta_f <= 0.0:
        raise ValueError(f"`eta` must be finite and positive. Got {eta!r}.")
    cz = float(z_offset)
    gamma_arr = np.asarray(gamma, dtype=np.complex128)
    scaled = gamma_arr * cz
    if np.any(np.abs(scaled) <= float(singular_atol)):
        raise ValueError(
            "shifted reciprocal integrals require a nonzero scaled height offset; "
            "use the same-plane reciprocal formula for this pair."
        )
    x = -(gamma_arr * gamma_arr) / (4.0 * eta_f * eta_f)
    if np.any(np.abs(x) <= float(singular_atol)):
        raise ValueError("shifted reciprocal integrals are singular at gamma=0.")
    root_x = np.where(x.real < 0.0, -1j * np.sqrt(np.abs(x)), np.sqrt(x))
    z_arg = np.where(x.real < 0.0, scaled, 1j * np.abs(scaled))
    z_sq = scaled * scaled
    exp_term = np.exp(-x + z_sq / (4.0 * x))

    out = np.zeros((*gamma_arr.shape, n_max + 1), dtype=np.complex128)
    w_minus = special.wofz(-z_arg / (2.0 * root_x) + 1j * root_x)
    w_plus = special.wofz(z_arg / (2.0 * root_x) + 1j * root_x)
    out[..., 0] = 0.5 * math.sqrt(math.pi) * exp_term * (w_minus + w_plus)
    if n_max == 0:
        return out
    out[..., 1] = 1j * math.sqrt(math.pi) / z_arg * exp_term * (w_minus - w_plus)
    for idx in range(2, n_max + 1):
        out[..., idx] = (
            4.0
            / z_sq
            * (
                (1.5 - float(idx)) * out[..., idx - 1]
                - out[..., idx - 2]
                + root_x * x ** (1 - idx) * exp_term
            )
        )
    return out


__all__ = [
    "kambe_integral",
    "reduced_incomplete_gamma_int_or_halfint",
    "shifted_delta_sequence",
    "upper_incomplete_gamma_int_or_halfint",
]
