"""CuPy special-function helpers for two-dimensional periodic Ewald sums.

This module mirrors the small subset of :mod:`pyceles.core.periodic.special`
needed by accelerated periodic-Ewald paths. CuPy/cupyx currently does not expose
``scipy.special.wofz`` (the Faddeeva function), so we provide a self-contained
GPU implementation based on the modified trapezoidal rules from M. Al Azah and
S. N. Chandler-Wilde, SIAM J. Numer. Anal. 59, 2346-2367 (2021).
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import import_cupy
from pyceles.core.periodic.special import (
    _SHIFTED_DELTA_SERIES_ETA_Z_LIMIT,
    _SHIFTED_DELTA_SERIES_SCALED_LIMIT,
    _SHIFTED_DELTA_SERIES_TERMS,
    _SHIFTED_DELTA_SERIES_X_MIN,
    upper_gamma_sequence,
)

_SMALL_COMPLEX = 1.0e-14
_DEFAULT_WOFZ_TERMS = 11


@cache
def _wtrap_quadrature_table_host(terms: int) -> np.ndarray:
    """Return integer- and midpoint-rule ``(t**2, exp(-t**2))`` pairs."""
    n_terms = int(terms)
    h = math.sqrt(math.pi / float(n_terms + 1))
    indices = np.arange(n_terms + 1, dtype=np.float64)
    table = np.empty((2, n_terms + 1, 2), dtype=np.float64)
    for branch, nodes in enumerate((h * indices, h * (indices + 0.5))):
        squared = nodes * nodes
        table[branch, :, 0] = squared
        table[branch, :, 1] = np.exp(-squared)
    return table


@cache
def _wtrap_quadrature_table_cupy(device_id: int, terms: int) -> Any:
    """Cache the small W-trapezoidal quadrature table on one CUDA device."""
    cp, _ = import_cupy()
    with cp.cuda.Device(int(device_id)):
        return cp.asarray(_wtrap_quadrature_table_host(int(terms)))


# Source fragment reused by periodic Ewald RawKernels.  CuPy exposes complex
# arithmetic through ``complex.cuh`` but not a Faddeeva implementation.
_WTRAP_DEVICE_CUDA_SOURCE = r"""
#include <cupy/complex.cuh>

__device__ complex<double> _wtrap_upper_one(
    const complex<double> z,
    const double* quadrature,
    const int terms,
    const double h,
    const double H
) {
    const double rz = z.real();
    const double iz = z.imag();
    const double rzh = rz / h;
    const double buff = fabs(rzh - floor(rzh) - 0.5);
    const bool use_midpoint_only = iz >= fmax(rz, H);
    const bool use_modified_trapezium = (iz < rz) && (buff <= 0.25);

    const complex<double> z2 = z * z;
    const complex<double> az = complex<double>(0.0, 2.0 / H) * z;

    if (use_modified_trapezium) {
        complex<double> sum = complex<double>(0.0, 0.0);
        for (int j = 1; j <= terms; ++j) {
            const double t2 = quadrature[2 * j];
            const double et2 = quadrature[2 * j + 1];
            sum += complex<double>(et2, 0.0) / (z2 - complex<double>(t2, 0.0));
        }
        const complex<double> exp_z2 = exp(z2);
        const complex<double> exp_shift = exp(complex<double>(0.0, -2.0 * H) * z);
        const complex<double> correction = complex<double>(2.0, 0.0)
            / (exp_z2 * (complex<double>(1.0, 0.0) - exp_shift));
        return complex<double>(0.0, 1.0 / H) / z + az * sum + correction;
    }

    const int midpoint_offset = 2 * (terms + 1);
    const double h0_squared = quadrature[midpoint_offset];
    complex<double> sum = complex<double>(quadrature[midpoint_offset + 1], 0.0)
        / (z2 - complex<double>(h0_squared, 0.0));
    for (int j = 1; j <= terms; ++j) {
        const double t2 = quadrature[midpoint_offset + 2 * j];
        const double et2 = quadrature[midpoint_offset + 2 * j + 1];
        sum += complex<double>(et2, 0.0) / (z2 - complex<double>(t2, 0.0));
    }
    const complex<double> midpoint = az * sum;
    if (use_midpoint_only) {
        return midpoint;
    }

    const complex<double> exp_z2 = exp(z2);
    const complex<double> exp_shift = exp(complex<double>(0.0, -2.0 * H) * z);
    const complex<double> correction = complex<double>(2.0, 0.0)
        / (exp_z2 * (complex<double>(1.0, 0.0) + exp_shift));
    return midpoint + correction;
}

__device__ complex<double> _wtrap_wofz_one(
    const complex<double> z0,
    const double* quadrature,
    const int terms,
    const double h,
    const double H
) {
    const bool xneg = z0.real() < 0.0;
    const bool yneg = z0.imag() < 0.0;
    const bool not_both = xneg != yneg;

    complex<double> z = z0;
    if (xneg) {
        z = -z;
    }
    if (not_both) {
        z = conj(z);
    }

    complex<double> w = _wtrap_upper_one(z, quadrature, terms, h, H);
    if (not_both) {
        w = conj(w);
    }
    if (yneg) {
        w = complex<double>(2.0, 0.0) * exp(-(z0 * z0)) - w;
    }
    return w;
}
"""


_WTRAP_CUDA_SOURCE = (
    _WTRAP_DEVICE_CUDA_SOURCE
    + r"""
extern "C" __global__ void pyceles_wofz_wtrap_c128(
    const long long n,
    const complex<double>* z_in,
    complex<double>* out,
    const double* quadrature,
    const int terms,
    const double h,
    const double H
) {
    const long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (idx >= n) {
        return;
    }

    out[idx] = _wtrap_wofz_one(z_in[idx], quadrature, terms, h, H);
}
"""
)


@cache
def _wtrap_raw_kernel() -> Any:
    cp, _ = import_cupy()
    return cp.RawKernel(_WTRAP_CUDA_SOURCE, "pyceles_wofz_wtrap_c128")


def wofz_cupy(z: Any, *, terms: int = _DEFAULT_WOFZ_TERMS, cupy: Any | None = None) -> Any:
    """Evaluate the Faddeeva function ``w(z)`` on CuPy arrays.

    Parameters
    ----------
    z:
        Scalar or array-like complex argument. The result is returned as a CuPy
        ``complex128`` array with the same shape.
    terms:
        Quadrature order ``N`` for the modified trapezoidal rules. The reference
        MATLAB implementation recommends ``N=11`` for near-double-precision
        accuracy; this is the default used here.
    cupy:
        Optional CuPy module handle, useful for callers that already imported it.
    """
    cp = cupy
    if cp is None:
        cp, _ = import_cupy()
    n_terms = int(terms)
    if n_terms < 1:
        raise ValueError(f"`terms` must be >= 1. Got {terms!r}.")

    z_arr = cp.asarray(z, dtype=cp.complex128)
    if int(z_arr.size) == 0:
        return cp.empty_like(z_arr, dtype=cp.complex128)

    z_flat = cp.ascontiguousarray(z_arr.reshape(-1))
    out_flat = cp.empty_like(z_flat, dtype=cp.complex128)
    h = math.sqrt(math.pi / float(n_terms + 1))
    quadrature = _wtrap_quadrature_table_cupy(int(cp.cuda.runtime.getDevice()), n_terms)
    H = math.pi / h
    threads = 256
    blocks = (int(z_flat.size) + threads - 1) // threads
    _wtrap_raw_kernel()(
        (blocks,),
        (threads,),
        (
            np.int64(int(z_flat.size)),
            z_flat,
            out_flat,
            quadrature,
            np.int32(n_terms),
            np.float64(h),
            np.float64(H),
        ),
    )
    return out_flat.reshape(z_arr.shape)


def _shifted_delta_sequence_series_cupy(
    max_order: int,
    gamma: Any,
    z_offset: Any,
    eta: float,
    *,
    upper_gamma: Any | None = None,
    cupy: Any,
) -> Any:
    """Evaluate the exact near-plane generalized-gamma power series on device."""
    cp = cupy
    n_max = int(max_order)
    gamma_arr = cp.asarray(gamma, dtype=cp.complex128).reshape(-1)
    z_arr = cp.asarray(z_offset, dtype=cp.float64).reshape(-1)
    x = -(gamma_arr * gamma_arr) / (4.0 * float(eta) * float(eta))
    if upper_gamma is None:
        # This standalone helper is a parity/reference API. The production
        # Ewald kernel receives its exact host-built table through the cached
        # reciprocal workspace and never takes this host route per matvec.
        gamma_table = cp.asarray(
            upper_gamma_sequence(
                n_max + _SHIFTED_DELTA_SERIES_TERMS,
                cp.asnumpy(x),
            ),
            dtype=cp.complex128,
        )
    else:
        gamma_table = cp.asarray(upper_gamma, dtype=cp.complex128)
    q = 0.25 * (z_arr[:, None] * gamma_arr[None, :]) ** 2
    out = cp.broadcast_to(
        gamma_table[None, :, _SHIFTED_DELTA_SERIES_TERMS:],
        (int(z_arr.size), int(gamma_arr.size), n_max + 1),
    ).copy()
    for term in range(_SHIFTED_DELTA_SERIES_TERMS - 1, -1, -1):
        out *= q[:, :, None] / float(term + 1)
        out += gamma_table[None, :, term : term + n_max + 1]
    return out


def _shifted_delta_sequence_cupy_raw(
    max_order: int,
    gamma: Any,
    z_offset: Any,
    eta: float,
    *,
    terms: int,
    cupy: Any,
) -> Any:
    """Evaluate the established Faddeeva recurrence on device."""
    cp = cupy
    n_max = int(max_order)
    gamma_arr = cp.asarray(gamma, dtype=cp.complex128).reshape(-1)
    z_arr = cp.asarray(z_offset, dtype=cp.float64).reshape(-1)
    x = -(gamma_arr * gamma_arr) / (4.0 * float(eta) * float(eta))
    root_x = cp.where(x.real < 0.0, -1j * cp.sqrt(cp.abs(x)), cp.sqrt(x))
    scaled = z_arr[:, None] * gamma_arr[None, :]
    z_arg = cp.where(x.real[None, :] < 0.0, scaled, 1j * cp.abs(scaled))
    z_sq = scaled * scaled
    x_b = x[None, :]
    root_b = root_x[None, :]
    exp_term = cp.exp(-x_b + z_sq / (4.0 * x_b))
    out = cp.zeros((int(z_arr.size), int(gamma_arr.size), n_max + 1), dtype=cp.complex128)
    w_minus = wofz_cupy(-z_arg / (2.0 * root_b) + 1j * root_b, terms=int(terms), cupy=cp)
    w_plus = wofz_cupy(z_arg / (2.0 * root_b) + 1j * root_b, terms=int(terms), cupy=cp)
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


def shifted_delta_sequence_cupy(
    max_order: int,
    gamma: Any,
    z_offset: float,
    eta: float,
    *,
    singular_atol: float = _SMALL_COMPLEX,
    terms: int = _DEFAULT_WOFZ_TERMS,
    cupy: Any | None = None,
) -> Any:
    """CuPy version of the stable shifted reciprocal sequence."""
    cp = cupy
    if cp is None:
        cp, _ = import_cupy()
    cz = float(z_offset)
    if cz == 0.0:
        raise ValueError(
            "shifted reciprocal integrals require a nonzero scaled height offset; "
            "use the same-plane reciprocal formula for this pair."
        )
    gamma_arr = cp.asarray(gamma, dtype=cp.complex128).reshape(-1)
    scaled = gamma_arr * cz
    if bool(cp.any(cp.abs(scaled) <= float(singular_atol)).get()):
        raise ValueError(
            "shifted reciprocal integrals are singular in the Rayleigh-threshold limit; "
            "use the exact same-plane formula when applicable."
        )
    result = shifted_delta_sequence_cupy_batched(
        int(max_order),
        gamma_arr,
        cp.asarray([cz], dtype=cp.float64),
        float(eta),
        terms=int(terms),
        cupy=cp,
    )
    return result[0]


def real_integral_sequence_cupy(
    degree: int,
    eta: float,
    k: float,
    radii: Any,
    *,
    terms: int = _DEFAULT_WOFZ_TERMS,
    cupy: Any | None = None,
) -> Any:
    """CuPy version of the real-space Ewald radial integral sequence.

    This mirrors the NumPy helper used by periodic Ewald evaluators and is kept
    here because it depends on the same Faddeeva primitive.
    """
    cp = cupy
    if cp is None:
        cp, _ = import_cupy()
    l = int(degree)
    radii_arr = cp.asarray(radii, dtype=cp.float64)
    radii_shape = tuple(radii_arr.shape)
    r = radii_arr.reshape(-1)
    alpha = float(k) * float(k) / (4.0 * float(eta) * float(eta))
    root_alpha = math.sqrt(alpha)
    # Avoid exact r=0 divisions in the recurrence; callers normally exclude the
    # zero self term, but this keeps the device path finite under roundoff.
    r_safe = cp.where(cp.abs(r) < 1.0e-30, 1.0e-30, r)
    w = wofz_cupy(
        root_alpha + 1j * float(k) * r_safe / (2.0 * root_alpha), terms=int(terms), cupy=cp
    )
    exp_term = cp.exp(alpha - (float(k) * r_safe) ** 2 / (4.0 * alpha))
    vals = cp.zeros((r_safe.size, l + 2), dtype=cp.complex128)
    vals[:, 0] = math.sqrt(math.pi) * exp_term * w.imag
    vals[:, 1] = math.sqrt(math.pi) * 2.0 / (float(k) * r_safe) * exp_term * w.real
    for idx in range(2, l + 2):
        vals[:, idx] = (2.0 / (float(k) * r_safe)) ** 2 * (
            0.5 * float(2 * (idx - 2) + 1) * vals[:, idx - 1]
            - vals[:, idx - 2]
            + alpha ** (-(idx - 2) - 0.5) * exp_term
        )
    return vals[:, -1].reshape(radii_shape)


def shifted_delta_sequence_cupy_batched(
    max_order: int,
    gamma: Any,
    z_offset: Any,
    eta: float,
    *,
    singular_atol: float = _SMALL_COMPLEX,
    terms: int = _DEFAULT_WOFZ_TERMS,
    cupy: Any | None = None,
) -> Any:
    """Evaluate stable shifted reciprocal sequences for many heights on device."""
    cp = cupy
    if cp is None:
        cp, _ = import_cupy()
    n_max = int(max_order)
    if n_max < 0:
        raise ValueError(f"`max_order` must be >= 0. Got {max_order!r}.")
    eta_f = float(eta)
    if not np.isfinite(eta_f) or eta_f <= 0.0:
        raise ValueError(f"`eta` must be finite and positive. Got {eta!r}.")
    gamma_arr = cp.asarray(gamma, dtype=cp.complex128).reshape(-1)
    z_arr = cp.asarray(z_offset, dtype=cp.float64).reshape(-1)
    if int(z_arr.size) == 0 or int(gamma_arr.size) == 0:
        return cp.zeros((int(z_arr.size), int(gamma_arr.size), n_max + 1), dtype=cp.complex128)

    scaled = z_arr[:, None] * gamma_arr[None, :]
    if bool(cp.any(cp.abs(scaled) <= float(singular_atol)).get()):
        raise ValueError(
            "shifted reciprocal integrals are singular in the same-plane or "
            "Rayleigh-threshold limit; route those points to the same-plane "
            "formula or avoid gamma*z_offset == 0."
        )
    x = -(gamma_arr * gamma_arr) / (4.0 * eta_f * eta_f)
    if bool(cp.any(cp.abs(x) <= float(singular_atol)).get()):
        raise ValueError("shifted reciprocal integrals are singular at gamma=0.")
    series_mask = (
        (cp.abs(scaled) <= _SHIFTED_DELTA_SERIES_SCALED_LIMIT)
        & (cp.abs(eta_f * z_arr[:, None]) <= _SHIFTED_DELTA_SERIES_ETA_Z_LIMIT)
        & (cp.abs(x)[None, :] > _SHIFTED_DELTA_SERIES_X_MIN)
    )
    all_series = bool(cp.all(series_mask).get())
    if all_series:
        return _shifted_delta_sequence_series_cupy(n_max, gamma_arr, z_arr, eta_f, cupy=cp)

    series_rows = cp.nonzero(cp.any(series_mask, axis=1))[0]
    raw_rows = cp.nonzero(cp.any(~series_mask, axis=1))[0]
    out = cp.zeros((int(z_arr.size), int(gamma_arr.size), n_max + 1), dtype=cp.complex128)
    if int(series_rows.size):
        series = _shifted_delta_sequence_series_cupy(
            n_max,
            gamma_arr,
            z_arr[series_rows],
            eta_f,
            cupy=cp,
        )
        row_mask = series_mask[series_rows]
        row_values = out[series_rows]
        row_values[row_mask] = series[row_mask]
        out[series_rows] = row_values
    if int(raw_rows.size):
        raw = _shifted_delta_sequence_cupy_raw(
            n_max,
            gamma_arr,
            z_arr[raw_rows],
            eta_f,
            terms=int(terms),
            cupy=cp,
        )
        row_mask = ~series_mask[raw_rows]
        row_values = out[raw_rows]
        row_values[row_mask] = raw[row_mask]
        out[raw_rows] = row_values
    return out


__all__ = [
    "real_integral_sequence_cupy",
    "shifted_delta_sequence_cupy",
    "shifted_delta_sequence_cupy_batched",
    "wofz_cupy",
]
