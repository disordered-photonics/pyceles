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

_SMALL_COMPLEX = 1.0e-14
_DEFAULT_WOFZ_TERMS = 11


# Source fragment reused by periodic Ewald RawKernels.  CuPy exposes complex
# arithmetic through ``complex.cuh`` but not a Faddeeva implementation.
_WTRAP_DEVICE_CUDA_SOURCE = r"""
#include <cupy/complex.cuh>

extern "C" __device__ complex<double> _wtrap_upper_one(
    const complex<double> z,
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
            const double t = h * (double)j;
            const double t2 = t * t;
            const double et2 = exp(-t2);
            sum += complex<double>(et2, 0.0) / (z2 - complex<double>(t2, 0.0));
        }
        const complex<double> exp_z2 = exp(z2);
        const complex<double> exp_shift = exp(complex<double>(0.0, -2.0 * H) * z);
        const complex<double> correction = complex<double>(2.0, 0.0)
            / (exp_z2 * (complex<double>(1.0, 0.0) - exp_shift));
        return complex<double>(0.0, 1.0 / H) / z + az * sum + correction;
    }

    const double h0 = 0.5 * h;
    complex<double> sum = complex<double>(exp(-(h0 * h0)), 0.0)
        / (z2 - complex<double>(h0 * h0, 0.0));
    for (int j = 1; j <= terms; ++j) {
        const double t = h * ((double)j + 0.5);
        const double t2 = t * t;
        const double et2 = exp(-t2);
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

extern "C" __device__ complex<double> _wtrap_wofz_one(
    const complex<double> z0,
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

    complex<double> w = _wtrap_upper_one(z, terms, h, H);
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
    const int terms,
    const double h,
    const double H
) {
    const long long idx = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (idx >= n) {
        return;
    }

    out[idx] = _wtrap_wofz_one(z_in[idx], terms, h, H);
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
            np.int32(n_terms),
            np.float64(h),
            np.float64(H),
        ),
    )
    return out_flat.reshape(z_arr.shape)


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
    """CuPy version of ``periodic.special.shifted_delta_sequence``.

    The formula and recurrence mirror the NumPy/SciPy reference implementation;
    only the Faddeeva evaluation is supplied by :func:`wofz_cupy`.
    """
    cp = cupy
    if cp is None:
        cp, _ = import_cupy()
    n_max = int(max_order)
    if n_max < 0:
        raise ValueError(f"`max_order` must be >= 0. Got {max_order!r}.")
    eta_f = float(eta)
    if not np.isfinite(eta_f) or eta_f <= 0.0:
        raise ValueError(f"`eta` must be finite and positive. Got {eta!r}.")
    cz = float(z_offset)
    if abs(cz) <= 0.0:
        raise ValueError(
            "shifted reciprocal integrals require a nonzero scaled height offset; "
            "use the same-plane reciprocal formula for this pair."
        )
    gamma_arr = cp.asarray(gamma, dtype=cp.complex128)
    scaled = gamma_arr * cz
    if bool(cp.any(cp.abs(scaled) <= float(singular_atol)).get()):
        raise ValueError(
            "shifted reciprocal integrals are singular in the Rayleigh-threshold limit; "
            "use the exact same-plane formula when applicable, otherwise avoid evaluating "
            "exactly at gamma*z_offset == 0 until a limiting formula is implemented."
        )
    x = -(gamma_arr * gamma_arr) / (4.0 * eta_f * eta_f)
    if bool(cp.any(cp.abs(x) <= float(singular_atol)).get()):
        raise ValueError("shifted reciprocal integrals are singular at gamma=0.")
    root_x = cp.where(x.real < 0.0, -1j * cp.sqrt(cp.abs(x)), cp.sqrt(x))
    z_arg = cp.where(x.real < 0.0, scaled, 1j * cp.abs(scaled))
    z_sq = scaled * scaled
    exp_term = cp.exp(-x + z_sq / (4.0 * x))

    out = cp.zeros((*gamma_arr.shape, n_max + 1), dtype=cp.complex128)
    w_minus = wofz_cupy(-z_arg / (2.0 * root_x) + 1j * root_x, terms=int(terms), cupy=cp)
    w_plus = wofz_cupy(z_arg / (2.0 * root_x) + 1j * root_x, terms=int(terms), cupy=cp)
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
    terms: int = _DEFAULT_WOFZ_TERMS,
    cupy: Any | None = None,
) -> Any:
    """Evaluate shifted reciprocal delta sequences for many heights at once.

    ``gamma`` is flattened to ``(n_gamma,)`` and ``z_offset`` to
    ``(n_offsets,)``. The result has shape
    ``(n_offsets, n_gamma, max_order + 1)``. This batched helper performs no
    device-to-host singularity checks; callers must route exact same-plane or
    Rayleigh-threshold cases to the same-plane formulas before calling it.
    """
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

    x = -(gamma_arr * gamma_arr) / (4.0 * eta_f * eta_f)
    root_x = cp.where(x.real < 0.0, -1j * cp.sqrt(cp.abs(x)), cp.sqrt(x))
    scaled = z_arr[:, None] * gamma_arr[None, :]
    z_arg = cp.where(x.real[None, :] < 0.0, scaled, 1j * cp.abs(scaled))
    z_sq = scaled * scaled
    x_b = x[None, :]
    root_b = root_x[None, :]
    exp_term = cp.exp(-x_b + z_sq / (4.0 * x_b))

    out = cp.zeros((z_arr.size, gamma_arr.size, n_max + 1), dtype=cp.complex128)
    w_minus = wofz_cupy(-z_arg / (2.0 * root_b) + 1j * root_b, terms=int(terms), cupy=cp)
    w_plus = wofz_cupy(z_arg / (2.0 * root_b) + 1j * root_b, terms=int(terms), cupy=cp)
    out[:, :, 0] = 0.5 * math.sqrt(math.pi) * exp_term * (w_minus + w_plus)
    if n_max == 0:
        return out

    z_safe = cp.where(cp.abs(z_arg) < _SMALL_COMPLEX, _SMALL_COMPLEX + 0.0j, z_arg)
    out[:, :, 1] = 1j * math.sqrt(math.pi) / z_safe * exp_term * (w_minus - w_plus)
    z_sq_safe = cp.where(cp.abs(z_sq) < _SMALL_COMPLEX, _SMALL_COMPLEX + 0.0j, z_sq)
    for idx in range(2, n_max + 1):
        out[:, :, idx] = (
            4.0
            / z_sq_safe
            * (
                (1.5 - float(idx)) * out[:, :, idx - 1]
                - out[:, :, idx - 2]
                + root_b * x_b ** (1 - idx) * exp_term
            )
        )
    return out


__all__ = [
    "real_integral_sequence_cupy",
    "shifted_delta_sequence_cupy",
    "shifted_delta_sequence_cupy_batched",
    "wofz_cupy",
]
