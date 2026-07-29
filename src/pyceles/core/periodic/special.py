"""Special-function helpers for two-dimensional periodic lattice sums."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import special

Array = np.ndarray

_KAMBE_SERIES_TERMS = 32
_SMALL_COMPLEX = 1e-14
_SHIFTED_DELTA_SERIES_TERMS = 16
_SHIFTED_DELTA_SERIES_SCALED_LIMIT = 8.0
_SHIFTED_DELTA_SERIES_ETA_Z_LIMIT = 0.5
_SHIFTED_DELTA_SERIES_X_MIN = _SMALL_COMPLEX
ShiftedReciprocalRegime = Literal["same_plane", "rayleigh_limit", "shifted"]


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


def upper_gamma_sequence(max_index: int, z: npt.ArrayLike) -> Array:
    """Return ``Gamma(1/2-n, z)`` for ``n=0..max_index``.

    The negative-real-axis branch is the same one used by the Ewald formulas.
    Building the sequence once by downward recurrence is substantially cheaper
    than recursively evaluating every order independently, and the resulting
    table is also the natural coefficient table for the near-plane shifted
    reciprocal expansion.
    """
    n_max = int(max_index)
    if n_max < 0:
        raise ValueError(f"`max_index` must be >= 0. Got {max_index!r}.")
    z_arr = np.asarray(z, dtype=np.complex128).reshape(-1)
    out = np.zeros((z_arr.size, n_max + 1), dtype=np.complex128)
    for row, value in enumerate(z_arr):
        zc = _branch_complex(complex(value))
        if zc == 0.0:
            raise ValueError("The shifted reciprocal upper-gamma sequence is singular at z=0.")
        out[row, 0] = upper_incomplete_gamma_int_or_halfint(0.5, zc)
        exp_term = complex(np.exp(-zc))
        power = zc**-0.5
        for index in range(n_max):
            order = -0.5 - float(index)
            out[row, index + 1] = (out[row, index] - power * exp_term) / order
            power /= zc
    return out


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


def shifted_reciprocal_regime(
    gamma: npt.ArrayLike,
    z_offset: float,
    *,
    same_plane_atol: float = 0.0,
    rayleigh_atol: float = _SMALL_COMPLEX,
) -> ShiftedReciprocalRegime:
    """Classify the reciprocal integral regime for one off-plane scalar sequence."""
    cz = float(z_offset)
    if abs(cz) <= float(same_plane_atol):
        return "same_plane"
    scaled = np.asarray(gamma, dtype=np.complex128) * cz
    if np.any(np.abs(scaled) <= float(rayleigh_atol)):
        return "rayleigh_limit"
    return "shifted"


def shifted_delta_series_max_index(max_order: int) -> int:
    """Return the upper-gamma order required by the stable shifted series."""
    order = int(max_order)
    if order < 0:
        raise ValueError(f"`max_order` must be >= 0. Got {max_order!r}.")
    return order + _SHIFTED_DELTA_SERIES_TERMS


def _shifted_delta_series_mask(
    gamma: Array,
    z_offset: Array,
    eta: float,
) -> Array:
    """Return entries where the generalized-gamma series is preferred.

    The forward recurrence loses roughly two powers of ``gamma*z`` at every
    order.  The exact power series converges in the complementary small-height
    region.  The two bounds keep both its scaled argument and its underlying
    ``eta*z`` expansion comfortably inside the machine-precision regime.
    """
    scaled = z_offset[:, None] * gamma[None, :]
    x = -(gamma * gamma) / (4.0 * float(eta) * float(eta))
    return np.asarray(
        (np.abs(scaled) <= _SHIFTED_DELTA_SERIES_SCALED_LIMIT)
        & (np.abs(float(eta) * z_offset[:, None]) <= _SHIFTED_DELTA_SERIES_ETA_Z_LIMIT)
        & (np.abs(x)[None, :] > _SHIFTED_DELTA_SERIES_X_MIN),
        dtype=bool,
    )


def shifted_delta_series_required(
    gamma: npt.ArrayLike,
    z_offset: npt.ArrayLike,
    eta: float,
) -> bool:
    """Return whether any shifted entry needs the stable near-plane series."""
    gamma_arr = np.asarray(gamma, dtype=np.complex128).reshape(-1)
    z_arr = np.asarray(z_offset, dtype=float).reshape(-1)
    if gamma_arr.size == 0 or z_arr.size == 0:
        return False
    return bool(np.any(_shifted_delta_series_mask(gamma_arr, z_arr, float(eta))))


def _shifted_delta_sequence_series(
    max_order: int,
    gamma: Array,
    z_offset: Array,
    eta: float,
    *,
    upper_gamma: Array | None = None,
) -> Array:
    r"""Evaluate the exact near-plane generalized-incomplete-gamma series.

    For ``x=-gamma**2/(4*eta**2)`` and ``q=(gamma*z)**2/4``,

    ``Delta_n = sum_j q**j/j! * Gamma(1/2-n-j, x)``.

    This is obtained by expanding the generalized incomplete-gamma integral
    before applying the ill-conditioned order recurrence.  Horner evaluation
    needs only one upper-gamma table and remains valid for arbitrary requested
    order; there is no low-``lmax`` special case.
    """
    n_max = int(max_order)
    gamma_arr = np.asarray(gamma, dtype=np.complex128).reshape(-1)
    z_arr = np.asarray(z_offset, dtype=float).reshape(-1)
    required = shifted_delta_series_max_index(n_max)
    if upper_gamma is None:
        x = -(gamma_arr * gamma_arr) / (4.0 * float(eta) * float(eta))
        gamma_table = upper_gamma_sequence(required, x)
    else:
        gamma_table = np.asarray(upper_gamma, dtype=np.complex128)
        expected_rows = int(gamma_arr.size)
        if gamma_table.ndim != 2 or gamma_table.shape[0] != expected_rows:
            raise ValueError(
                f"`upper_gamma` must have shape (n_gamma, n_orders); got {gamma_table.shape!r}."
            )
        if gamma_table.shape[1] <= required:
            raise ValueError(
                "`upper_gamma` does not contain enough orders for the shifted series: "
                f"need at least {required + 1}, got {gamma_table.shape[1]}."
            )

    q = 0.25 * (z_arr[:, None] * gamma_arr[None, :]) ** 2
    out = np.broadcast_to(
        gamma_table[None, :, _SHIFTED_DELTA_SERIES_TERMS : required + 1],
        (z_arr.size, gamma_arr.size, n_max + 1),
    ).copy()
    for term in range(_SHIFTED_DELTA_SERIES_TERMS - 1, -1, -1):
        out *= q[:, :, None] / float(term + 1)
        out += gamma_table[None, :, term : term + n_max + 1]
    return out


def _shifted_delta_sequence_batched_raw(
    max_order: int,
    gamma: npt.ArrayLike,
    z_offset: npt.ArrayLike,
    eta: float,
    *,
    singular_atol: float = _SMALL_COMPLEX,
) -> Array:
    """Evaluate the established shifted recurrence away from its removable limit."""
    n_max = int(max_order)
    gamma_arr = np.asarray(gamma, dtype=np.complex128).reshape(-1)
    z_arr = np.asarray(z_offset, dtype=float).reshape(-1)
    out = np.zeros((z_arr.size, gamma_arr.size, n_max + 1), dtype=np.complex128)
    if z_arr.size == 0 or gamma_arr.size == 0:
        return out
    scaled = z_arr[:, None] * gamma_arr[None, :]
    if np.any(np.abs(scaled) <= float(singular_atol)):
        raise ValueError(
            "shifted reciprocal integrals are singular in the same-plane or "
            "Rayleigh-threshold limit; route those points to the same-plane "
            "formula or avoid gamma*z_offset == 0."
        )
    x = -(gamma_arr * gamma_arr) / (4.0 * float(eta) * float(eta))
    if np.any(np.abs(x) <= float(singular_atol)):
        raise ValueError("shifted reciprocal integrals are singular at gamma=0.")
    root_x = np.where(x.real < 0.0, -1j * np.sqrt(np.abs(x)), np.sqrt(x))
    z_arg = np.where(x.real[None, :] < 0.0, scaled, 1j * np.abs(scaled))
    z_sq = scaled * scaled
    x_b = x[None, :]
    root_b = root_x[None, :]
    exp_term = np.exp(-x_b + z_sq / (4.0 * x_b))

    w_minus = special.wofz(-z_arg / (2.0 * root_b) + 1j * root_b)
    w_plus = special.wofz(z_arg / (2.0 * root_b) + 1j * root_b)
    out[:, :, 0] = 0.5 * math.sqrt(math.pi) * exp_term * (w_minus + w_plus)
    if n_max == 0:
        return out
    out[:, :, 1] = 1j * math.sqrt(math.pi) / z_arg * exp_term * (w_minus - w_plus)
    x_power = 1.0 / x_b
    for index in range(2, n_max + 1):
        out[:, :, index] = (
            4.0
            / z_sq
            * (
                (1.5 - float(index)) * out[:, :, index - 1]
                - out[:, :, index - 2]
                + root_b * x_power * exp_term
            )
        )
        x_power /= x_b
    return out


def shifted_delta_sequence(
    max_order: int,
    gamma: npt.ArrayLike,
    z_offset: float,
    eta: float,
    *,
    singular_atol: float = _SMALL_COMPLEX,
    upper_gamma: npt.ArrayLike | None = None,
) -> Array:
    """Evaluate shifted reciprocal-space Ewald integrals for one height."""
    if float(z_offset) == 0.0:
        raise ValueError(
            "shifted reciprocal integrals require a nonzero height offset; "
            "use the same-plane reciprocal formula for this pair."
        )
    result = shifted_delta_sequence_batched(
        int(max_order),
        gamma,
        np.asarray([float(z_offset)]),
        float(eta),
        singular_atol=singular_atol,
        upper_gamma=upper_gamma,
    )
    return np.asarray(result[0], dtype=np.complex128)


def shifted_delta_sequence_batched(
    max_order: int,
    gamma: npt.ArrayLike,
    z_offset: npt.ArrayLike,
    eta: float,
    *,
    singular_atol: float = _SMALL_COMPLEX,
    upper_gamma: npt.ArrayLike | None = None,
) -> Array:
    """Evaluate shifted reciprocal Ewald integrals for many height offsets.

    The ordinary Faddeeva recurrence is retained where it is well conditioned.
    Near a plane, the exact generalized-incomplete-gamma power series removes
    the repeated ``(gamma*z)**-2`` cancellation without extra Ewald evaluations.
    """
    n_max = int(max_order)
    if n_max < 0:
        raise ValueError(f"`max_order` must be >= 0. Got {max_order!r}.")
    eta_f = float(eta)
    if not np.isfinite(eta_f) or eta_f <= 0.0:
        raise ValueError(f"`eta` must be finite and positive. Got {eta!r}.")
    gamma_arr = np.asarray(gamma, dtype=np.complex128).reshape(-1)
    z_arr = np.asarray(z_offset, dtype=float).reshape(-1)
    out = np.zeros((z_arr.size, gamma_arr.size, n_max + 1), dtype=np.complex128)
    if z_arr.size == 0 or gamma_arr.size == 0:
        return out
    scaled = z_arr[:, None] * gamma_arr[None, :]
    if np.any(np.abs(scaled) <= float(singular_atol)):
        raise ValueError(
            "shifted reciprocal integrals are singular in the same-plane or "
            "Rayleigh-threshold limit; route those points to the same-plane "
            "formula or avoid gamma*z_offset == 0."
        )
    x = -(gamma_arr * gamma_arr) / (4.0 * eta_f * eta_f)
    if np.any(np.abs(x) <= float(singular_atol)):
        raise ValueError("shifted reciprocal integrals are singular at gamma=0.")

    series_mask = _shifted_delta_series_mask(gamma_arr, z_arr, eta_f)
    series_rows = np.flatnonzero(np.any(series_mask, axis=1))
    if series_rows.size:
        series = _shifted_delta_sequence_series(
            n_max,
            gamma_arr,
            z_arr[series_rows],
            eta_f,
            upper_gamma=None if upper_gamma is None else np.asarray(upper_gamma),
        )
        row_values = out[series_rows]
        row_mask = series_mask[series_rows]
        row_values[row_mask] = series[row_mask]
        out[series_rows] = row_values
    if np.all(series_mask):
        return out

    # A near-coplanar row usually needs the series for every reciprocal term.
    # Restrict each vectorized evaluator to rows that use it so one exceptional
    # pair does not double the work for an otherwise ordinary particle batch.
    raw_rows = np.flatnonzero(np.any(~series_mask, axis=1))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = _shifted_delta_sequence_batched_raw(
            n_max,
            gamma_arr,
            z_arr[raw_rows],
            eta_f,
            singular_atol=singular_atol,
        )
    row_values = out[raw_rows]
    row_mask = ~series_mask[raw_rows]
    row_values[row_mask] = raw[row_mask]
    out[raw_rows] = row_values
    return out


__all__ = [
    "kambe_integral",
    "reduced_incomplete_gamma_int_or_halfint",
    "shifted_delta_sequence",
    "shifted_delta_sequence_batched",
    "shifted_delta_series_max_index",
    "shifted_delta_series_required",
    "shifted_reciprocal_regime",
    "upper_gamma_sequence",
    "upper_incomplete_gamma_int_or_halfint",
]
