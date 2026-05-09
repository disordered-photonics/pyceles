"""CuPy special-function helpers for two-dimensional periodic Ewald sums.

This module mirrors the small subset of :mod:`pyceles.core.periodic.special`
needed by the experimental CuPy periodic-Ewald path.  CuPy/cupyx currently does
not expose ``scipy.special.wofz`` (the Faddeeva function), so we provide a
self-contained GPU implementation based on Weideman's rational approximation.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import import_cupy

_SMALL_COMPLEX = 1.0e-14
_DEFAULT_WOFZ_TERMS = 64


@cache
def _weideman_coefficients_host(terms: int) -> tuple[float, np.ndarray]:
    """Return Weideman coefficients for the Faddeeva approximation.

    This is the compact rational approximation described by J. A. C. Weideman,
    SIAM J. Numer. Anal. 31, 1497--1518 (1994).  The coefficients are generated
    once on the host using NumPy FFTs and then transferred to the active CUDA
    device on demand.
    """
    n = int(terms)
    if n < 8:
        raise ValueError(f"`terms` must be >= 8. Got {terms!r}.")
    m = 2 * n
    m2 = 2 * m
    k = np.arange(-m + 1, m, dtype=np.float64)
    length = math.sqrt(n / math.sqrt(2.0))
    theta = k * math.pi / float(m)
    t = length * np.tan(theta / 2.0)
    f = np.exp(-(t * t)) * (length * length + t * t)
    f = np.concatenate(([0.0], f))
    coeff = np.real(np.fft.fft(np.fft.fftshift(f))) / float(m2)
    coeff = coeff[1 : n + 1][::-1].copy().astype(np.float64)
    return float(length), coeff


_DEVICE_COEFF_CACHE: dict[tuple[int, int], tuple[float, Any]] = {}


def _weideman_coefficients_device(terms: int, cupy: Any) -> tuple[float, Any]:
    """Return ``(L, coeffs_on_current_device)`` for the active CUDA device."""
    device_id = int(cupy.cuda.Device().id)
    key = (int(terms), device_id)
    cached = _DEVICE_COEFF_CACHE.get(key)
    if cached is not None:
        return cached
    length, coeff_host = _weideman_coefficients_host(int(terms))
    coeff_dev = cupy.asarray(coeff_host, dtype=cupy.float64)
    _DEVICE_COEFF_CACHE[key] = (length, coeff_dev)
    return length, coeff_dev


_WOFZ_KERNEL = r"""
extern "C" __global__ void pyceles_wofz_weideman64(
    const double2* __restrict__ z,
    const double* __restrict__ coeff,
    const long n,
    const int terms,
    const double L,
    double2* __restrict__ out
) {
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    const double inv_sqrt_pi = 0.564189583547756286948079451560772585844050629329;

    double x = z[i].x;
    double y = z[i].y;
    const bool lower = (y < 0.0);

    // Evaluate in the upper half-plane, then use w(z)=2*exp(-z*z)-w(-z)
    // for lower-half-plane arguments.  This avoids the pole-like quadrature
    // difficulty near the real axis and follows the standard Faddeeva symmetry.
    double xu = lower ? -x : x;
    double yu = lower ? -y : y;

    // Z = (L + i*z) / (L - i*z)
    // For z=x+iy: L+i*z = (L-y) + i*x, L-i*z = (L+y) - i*x.
    const double ar = L - yu;
    const double ai = xu;
    const double br = L + yu;
    const double bi = -xu;
    const double bden = br * br + bi * bi;
    const double Zr = (ar * br + ai * bi) / bden;
    const double Zi = (ai * br - ar * bi) / bden;

    // Horner evaluation of the real-coefficient polynomial.
    double pr = 0.0;
    double pi = 0.0;
    for (int j = 0; j < terms; ++j) {
        const double nr = pr * Zr - pi * Zi + coeff[j];
        const double ni = pr * Zi + pi * Zr;
        pr = nr;
        pi = ni;
    }

    // den = L - i*z = (L+y) - i*x; den2 = den*den.
    const double dr = br;
    const double di = bi;
    const double d2r = dr * dr - di * di;
    const double d2i = 2.0 * dr * di;
    const double d2den = d2r * d2r + d2i * d2i;

    // term1 = 2*p/den^2
    const double t1r = 2.0 * (pr * d2r + pi * d2i) / d2den;
    const double t1i = 2.0 * (pi * d2r - pr * d2i) / d2den;

    // term2 = invsqrtpi / den
    const double dden = dr * dr + di * di;
    const double t2r = inv_sqrt_pi * dr / dden;
    const double t2i = -inv_sqrt_pi * di / dden;

    double wr = t1r + t2r;
    double wi = t1i + t2i;

    if (lower) {
        // 2*exp(-z*z) - w(-z)
        const double er = exp(-x * x + y * y);
        const double phase = -2.0 * x * y;
        const double exr = er * cos(phase);
        const double exi = er * sin(phase);
        wr = 2.0 * exr - wr;
        wi = 2.0 * exi - wi;
    }

    out[i].x = wr;
    out[i].y = wi;
}
"""

_WOFZ_RAW_KERNEL: Any | None = None


def _wofz_raw_kernel(cupy: Any) -> Any:
    global _WOFZ_RAW_KERNEL
    if _WOFZ_RAW_KERNEL is None:
        _WOFZ_RAW_KERNEL = cupy.RawKernel(_WOFZ_KERNEL, "pyceles_wofz_weideman64")
    return _WOFZ_RAW_KERNEL


def wofz_cupy(z: Any, *, terms: int = _DEFAULT_WOFZ_TERMS, cupy: Any | None = None) -> Any:
    """Evaluate the Faddeeva function ``w(z)`` on CuPy arrays.

    Parameters
    ----------
    z:
        Scalar or array-like complex argument.  The result is returned as a
        CuPy ``complex128`` array.
    terms:
        Number of Weideman rational-approximation terms.  ``64`` is the default
        and is typically close to double precision over the Ewald argument range.
    cupy:
        Optional CuPy module handle, useful for callers that already imported it.

    Notes
    -----
    The Faddeeva function is ``w(z) = exp(-z**2) * erfc(-1j*z)``.
    """
    cp = cupy
    if cp is None:
        cp, _ = import_cupy()
    z_arr = cp.asarray(z, dtype=cp.complex128)
    out = cp.empty_like(z_arr, dtype=cp.complex128)
    if z_arr.size == 0:
        return out
    length, coeff = _weideman_coefficients_device(int(terms), cp)
    flat_z = cp.ascontiguousarray(z_arr.reshape(-1))
    flat_out = cp.empty_like(flat_z, dtype=cp.complex128)
    block = 256
    grid = (int((flat_z.size + block - 1) // block),)
    _wofz_raw_kernel(cp)(
        grid,
        (block,),
        (flat_z, coeff, np.int64(flat_z.size), np.int32(int(terms)), np.float64(length), flat_out),
    )
    return flat_out.reshape(z_arr.shape)


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

    This mirrors the helper used by the experimental CuPy Ewald evaluator.  It is
    intentionally kept here because it depends on the same Faddeeva primitive.
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
