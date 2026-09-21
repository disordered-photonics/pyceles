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
from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import import_cupy
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic.scalar import (
    chebyshev_shell_indices,
    factorial_int,
    reciprocal_gamma_with_zero_mask,
    same_plane_z_tolerance,
    structural_sum_m_normalization,
)
from pyceles.core.periodic.special import (
    _SHIFTED_DELTA_SERIES_ETA_Z_LIMIT,
    _SHIFTED_DELTA_SERIES_SCALED_LIMIT,
    _SHIFTED_DELTA_SERIES_TERMS,
    upper_gamma_sequence,
)
from pyceles.core.periodic.special_cupy import (
    _DEFAULT_WOFZ_TERMS,
    _WTRAP_DEVICE_CUDA_SOURCE,
    _wtrap_quadrature_table_cupy,
)


@dataclass
class CupyReciprocalTerms:
    """Device-side reciprocal data for all shells in one fixed range."""

    kgt: Any
    rho: Any
    phi: Any
    gamma: Any
    xarg: Any
    xarg_host: np.ndarray
    rayleigh_zero: Any
    has_rayleigh_zero: bool
    shell_offsets: tuple[int, ...]


@dataclass
class CupyShiftedReciprocalTables:
    """Pair-invariant device tables reused by the shifted reciprocal kernel.

    Term tables use shape ``(n_terms, order + 1)`` except ``azimuth``, whose
    second axis spans ``m=-order..order``.  ``prefactor`` is indexed by the
    flattened structural entry, while ``inverse_denominator`` appends the
    recurrence ``(n, s)`` axes. None of these arrays scale with particle count.
    """

    azimuth: Any
    rho_powers: Any
    gamma_powers: Any
    root_x: Any
    prefactor: Any
    inverse_denominator: Any


@dataclass
class CupyRealTerms:
    """Device-side real-lattice data for all shells in one fixed range."""

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
    reciprocal_terms_cache: dict[int, CupyReciprocalTerms] = field(default_factory=dict)
    shifted_reciprocal_cache: dict[tuple[int, int], CupyShiftedReciprocalTables] = field(
        default_factory=dict
    )
    real_terms_cache: dict[int, CupyRealTerms] = field(default_factory=dict)
    upper_gamma_terms_cache: dict[int, tuple[int, Any]] = field(default_factory=dict)

    def reciprocal_terms(self, shell_count: int) -> CupyReciprocalTerms:
        """Return contiguous reciprocal vectors through one fixed shell count."""
        count = int(shell_count)
        cached = self.reciprocal_terms_cache.get(count)
        if cached is not None:
            return cached
        cp = self.cupy
        reciprocal_indices: list[tuple[int, int]] = []
        shell_offsets = [0]
        for shell in range(count + 1):
            reciprocal_indices.extend(chebyshev_shell_indices(shell))
            shell_offsets.append(len(reciprocal_indices))
        reciprocal = np.asarray(
            [p * self.lattice.b1 + q * self.lattice.b2 for p, q in reciprocal_indices],
            dtype=np.float64,
        )
        kgt_np = np.asarray(self.k_parallel, dtype=np.float64).reshape(2)[None, :] + reciprocal
        rho_np = np.linalg.norm(kgt_np, axis=1)
        phi_np = np.arctan2(kgt_np[:, 1], kgt_np[:, 0])
        gamma_np, rayleigh_zero_np = reciprocal_gamma_with_zero_mask(float(self.k), rho_np)
        xarg_np = -(gamma_np * gamma_np) / (4.0 * float(self.eta) * float(self.eta))
        out = CupyReciprocalTerms(
            kgt=cp.asarray(kgt_np, dtype=cp.float64),
            rho=cp.asarray(rho_np, dtype=cp.float64),
            phi=cp.asarray(phi_np, dtype=cp.float64),
            gamma=cp.asarray(gamma_np, dtype=cp.complex128),
            xarg=cp.asarray(xarg_np, dtype=cp.complex128),
            xarg_host=np.asarray(xarg_np, dtype=np.complex128),
            rayleigh_zero=cp.asarray(rayleigh_zero_np, dtype=cp.uint8),
            has_rayleigh_zero=bool(np.any(rayleigh_zero_np)),
            shell_offsets=tuple(int(value) for value in shell_offsets),
        )
        self.reciprocal_terms_cache[count] = out
        return out

    def upper_gamma_terms(self, shell_count: int, max_index: int) -> Any:
        """Return one growable upper-gamma table aligned with fixed reciprocal terms."""
        count = int(shell_count)
        max_i = int(max_index)
        cached = self.upper_gamma_terms_cache.get(count)
        if cached is not None and cached[0] >= max_i:
            return cached[1][:, : max_i + 1]
        cp = self.cupy
        terms = self.reciprocal_terms(count)
        out = cp.asarray(
            upper_gamma_sequence(max_i, terms.xarg_host),
            dtype=cp.complex128,
        )
        self.upper_gamma_terms_cache[count] = (max_i, out)
        return out

    def shifted_reciprocal_tables(
        self, shell_count: int, order: int
    ) -> CupyShiftedReciprocalTables:
        """Return pair-invariant tables for the shifted reciprocal kernel."""
        key = (int(shell_count), int(order))
        cached = self.shifted_reciprocal_cache.get(key)
        if cached is not None:
            return cached
        cp = self.cupy
        terms = self.reciprocal_terms(int(shell_count))
        order_i = int(order)
        width = 2 * order_i + 1
        n_terms = int(terms.rho.shape[0])
        m_values = cp.arange(-order_i, order_i + 1, dtype=cp.float64)
        azimuth = cp.exp(1j * terms.phi[:, None] * m_values[None, :]).astype(cp.complex128)

        rho_base = terms.rho / float(self.k)
        gamma_base = terms.gamma / float(self.k)
        rho_powers = cp.empty((n_terms, order_i + 1), dtype=cp.float64)
        gamma_powers = cp.empty((n_terms, order_i + 1), dtype=cp.complex128)
        rho_powers[:, 0] = 1.0
        gamma_powers[:, 0] = 1.0 / gamma_base
        gamma_step = gamma_base * gamma_base
        for exponent in range(1, order_i + 1):
            rho_powers[:, exponent] = rho_powers[:, exponent - 1] * rho_base
            gamma_powers[:, exponent] = gamma_powers[:, exponent - 1] * gamma_step
        root_x = cp.where(
            terms.xarg.real < 0.0,
            -1j * cp.sqrt(cp.abs(terms.xarg)),
            cp.sqrt(terms.xarg),
        ).astype(cp.complex128)

        n_entries = (order_i + 1) * width
        prefactor = np.zeros(n_entries, dtype=np.complex128)
        inverse_denominator = np.zeros((n_entries, order_i + 1, order_i + 1), dtype=np.float64)
        for degree in range(order_i + 1):
            for m in range(-degree, degree + 1):
                entry = degree * width + m + order_i
                m_norm = math.sqrt(2.0 * math.pi) * (-1.0 if m >= 0 and (m & 1) else 1.0)
                root = (
                    math.sqrt(2.0 * degree + 1.0)
                    * math.sqrt(factorial_int(degree - m))
                    * math.sqrt(factorial_int(degree + m))
                )
                prefactor[entry] = (
                    m_norm
                    * (-1j) ** m
                    * root
                    / ((-2.0) ** degree * self.lattice.area * float(self.k) ** 2)
                )
                n_max = degree - abs(m)
                for n in range(n_max + 1):
                    for s_val in range(n, min(n_max, 2 * n) + 1):
                        if (s_val - n_max) & 1:
                            continue
                        denominator = (
                            factorial_int(2 * n - s_val)
                            * factorial_int(s_val - n)
                            * factorial_int((degree + abs(m) - s_val) // 2)
                            * factorial_int((degree - abs(m) - s_val) // 2)
                        )
                        inverse_denominator[entry, n, s_val] = 1.0 / float(denominator)

        out = CupyShiftedReciprocalTables(
            azimuth=azimuth,
            rho_powers=cp.ascontiguousarray(rho_powers),
            gamma_powers=cp.ascontiguousarray(gamma_powers),
            root_x=cp.ascontiguousarray(root_x),
            prefactor=cp.asarray(prefactor, dtype=cp.complex128),
            inverse_denominator=cp.asarray(inverse_denominator, dtype=cp.float64),
        )
        self.shifted_reciprocal_cache[key] = out
        return out

    def real_terms(self, shell_count: int) -> CupyRealTerms:
        """Return contiguous real-lattice shifts through one fixed shell count."""
        count = int(shell_count)
        cached = self.real_terms_cache.get(count)
        if cached is not None:
            return cached
        cp = self.cupy
        indices: list[tuple[int, int]] = []
        for shell in range(count + 1):
            indices.extend(chebyshev_shell_indices(shell))
        shifts_np = np.asarray(
            [p * self.lattice.a1 + q * self.lattice.a2 for p, q in indices],
            dtype=np.float64,
        )
        k_parallel = np.asarray(self.k_parallel, dtype=np.float64).reshape(2)
        phase_xy_np = np.exp(1j * (shifts_np[:, :2] @ k_parallel))
        out = CupyRealTerms(
            shifts=cp.asarray(shifts_np, dtype=cp.float64),
            phase_xy=cp.asarray(phase_xy_np, dtype=cp.complex128),
        )
        self.real_terms_cache[count] = out
        return out


# ``complex.cuh`` supplies thrust complex math but no pi constants.  Keep
# pyceles-scoped, guarded constants local instead of relying on unrelated
# headers or generic ``M_PI`` / ``NPY_*`` macro names.
_EWALD_DEVICE_CONSTANTS = r"""
#ifndef PYCELES_PI
#define PYCELES_PI 3.141592653589793238462643383279502884
#endif
#ifndef PYCELES_SQRT1_2
#define PYCELES_SQRT1_2 0.707106781186547524400844362104849039
#endif
"""


_REAL_SPACE_STRUCTURAL_CUDA_SOURCE = r"""

__device__ void _pyceles_legendre_table(
    const double ct,
    const double st,
    const int order,
    double* plm
) {
    const int width = PYCELES_REAL_MAX_ORDER + 1;
    for (int idx = 0; idx < width * width; ++idx) {
        plm[idx] = 0.0;
    }
    plm[0] = PYCELES_SQRT1_2;
    if (order == 0) {
        return;
    }
    plm[1 * width + 0] = sqrt(1.5) * ct;

    for (int l = 1; l < order; ++l) {
        const double lp1 = (double)(l + 1);
        const double a0 = sqrt(((2.0 * l + 1.0) * (2.0 * l + 3.0))) / lp1;
        const double b0 = ((double)l / lp1) * sqrt((2.0 * l + 3.0) / (2.0 * l - 1.0));
        plm[(l + 1) * width + 0] = a0 * ct * plm[l * width + 0] - b0 * plm[(l - 1) * width + 0];
    }

    double cmm = PYCELES_SQRT1_2;
    double st_pow = st;
    for (int m = 1; m <= order; ++m) {
        cmm *= sqrt((2.0 * m + 1.0) / (2.0 * m));
        plm[m * width + m] = cmm * st_pow;
        for (int l = m; l < order; ++l) {
            const double lp1 = (double)(l + 1);
            const double den = (lp1 - (double)m) * (lp1 + (double)m);
            const double alm = sqrt((2.0 * l + 1.0) * (2.0 * l + 3.0) / den);
            const double blm = sqrt(
                (2.0 * l + 3.0)
                * ((double)l - (double)m)
                * ((double)l + (double)m)
                / ((2.0 * l - 1.0) * den)
            );
            plm[(l + 1) * width + m] =
                alm * ct * plm[l * width + m] - blm * plm[(l - 1) * width + m];
        }
        st_pow *= st;
    }
}

__device__ void _pyceles_real_integrals(
    const double r,
    const int order,
    const double eta,
    const double k,
    const double* quadrature,
    const int terms,
    const double h,
    const double H,
    double* integral
) {
    const double alpha = k * k / (4.0 * eta * eta);
    const double root_alpha = sqrt(alpha);
    const double kr = k * r;
    const complex<double> z(root_alpha, kr / (2.0 * root_alpha));
    const complex<double> w = _wtrap_wofz_one(z, quadrature, terms, h, H);
    const double exp_term = exp(alpha - (kr * kr) / (4.0 * alpha));
    double vals[PYCELES_REAL_MAX_ORDER + 2];
    for (int idx = 0; idx < PYCELES_REAL_MAX_ORDER + 2; ++idx) {
        vals[idx] = 0.0;
    }
    vals[0] = sqrt(PYCELES_PI) * exp_term * w.imag();
    vals[1] = sqrt(PYCELES_PI) * 2.0 / kr * exp_term * w.real();
    const double inv = 2.0 / kr;
    const double inv2 = inv * inv;
    double alpha_power = 1.0 / root_alpha;
    for (int idx = 2; idx < order + 2; ++idx) {
        vals[idx] = inv2 * (
            0.5 * (double)(2 * (idx - 2) + 1) * vals[idx - 1]
            - vals[idx - 2]
            + alpha_power * exp_term
        );
        alpha_power /= alpha;
    }
    double half_power = 0.5 * PYCELES_SQRT1_2;
    for (int l = 0; l <= order; ++l) {
        integral[l] = half_power * vals[l + 1];
        half_power *= 0.5;
    }
}

extern "C" __global__ void pyceles_ewald_real_space_structural_c128(
    const long long n_pairs,
    const int order,
    const long long n_shifts,
    const double* c,
    const unsigned char* same_plane,
    const double* shifts,
    const complex<double>* phase_xy,
    complex<double>* sums,
    const double eta,
    const double k,
    const double* quadrature,
    const int terms,
    const double h,
    const double H
) {
    const long long pair = (long long)blockIdx.x;
    if (pair >= n_pairs || order < 0 || order > PYCELES_REAL_MAX_ORDER) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int width = 2 * order + 1;
    const int n_entries = (order + 1) * width;
    extern __shared__ double shared[];

    for (int idx = tid; idx < 2 * n_entries; idx += (int)blockDim.x) {
        shared[idx] = 0.0;
    }
    __syncthreads();

    const double cx = c[3 * pair + 0];
    const double cy = c[3 * pair + 1];
    const double cz = c[3 * pair + 2];
    const bool is_same_plane = same_plane[pair] != 0;

    for (long long shift_idx = tid; shift_idx < n_shifts; shift_idx += (long long)blockDim.x) {
        const double x = -(shifts[3 * shift_idx + 0] + cx);
        const double y = -(shifts[3 * shift_idx + 1] + cy);
        const double z = -(shifts[3 * shift_idx + 2] + cz);
        const double r2 = x * x + y * y + z * z;
        if (r2 <= 0.0) {
            continue;
        }
        const double r = sqrt(r2);
        const double ct = z / r;
        const double st = sqrt(fmax(0.0, 1.0 - ct * ct));
        const double phi = atan2(y, x);
        const double kr = k * r;

        double plm[(PYCELES_REAL_MAX_ORDER + 1) * (PYCELES_REAL_MAX_ORDER + 1)];
        double integrals[PYCELES_REAL_MAX_ORDER + 1];
        double kr_pow[PYCELES_REAL_MAX_ORDER + 1];
        _pyceles_legendre_table(ct, st, order, plm);
        _pyceles_real_integrals(r, order, eta, k, quadrature, terms, h, H, integrals);
        kr_pow[0] = 1.0;
        for (int l = 1; l <= order; ++l) {
            kr_pow[l] = kr_pow[l - 1] * kr;
        }

        const complex<double> phase = phase_xy[shift_idx];
        const int table_width = PYCELES_REAL_MAX_ORDER + 1;
        for (int l = 0; l <= order; ++l) {
            const complex<double> radial = phase * (kr_pow[l] * integrals[l]);
            const complex<double> base = complex<double>(0.0, -sqrt(2.0 / PYCELES_PI)) * radial;
            for (int m = -l; m <= l; ++m) {
                const int abs_m = m < 0 ? -m : m;
                if (is_same_plane && (((l - abs_m) & 1) != 0)) {
                    continue;
                }
                const double angle = (double)m * phi;
                double sin_angle;
                double cos_angle;
                sincos(angle, &sin_angle, &cos_angle);
                const complex<double> azimuth(cos_angle, sin_angle);
                const complex<double> contrib =
                    base * (plm[l * table_width + abs_m]) * azimuth;
                const int entry = l * width + (m + order);
                atomicAdd(&shared[2 * entry], contrib.real());
                atomicAdd(&shared[2 * entry + 1], contrib.imag());
            }
        }
    }
    __syncthreads();

    for (int entry = tid; entry < n_entries; entry += (int)blockDim.x) {
        const double re = shared[2 * entry];
        const double im = shared[2 * entry + 1];
        if (re != 0.0 || im != 0.0) {
            sums[pair * n_entries + entry] += complex<double>(re, im);
        }
    }
}
"""


@cache
def _real_space_structural_raw_kernel(max_order: int) -> Any:
    cp, _ = import_cupy()
    order = int(max_order)
    source = (
        _WTRAP_DEVICE_CUDA_SOURCE
        + _EWALD_DEVICE_CONSTANTS
        + f"\n#define PYCELES_REAL_MAX_ORDER {order}\n"
        + _REAL_SPACE_STRUCTURAL_CUDA_SOURCE
    )
    return cp.RawKernel(source, "pyceles_ewald_real_space_structural_c128")


def _add_real_space_structural_sums_cupy(
    *,
    c: Any,
    same_plane: Any,
    sums: Any,
    workspace: CupyEwaldShellWorkspace,
    real_shell_count: int,
    order: int,
    eta: float,
    k: float,
) -> None:
    """Add the real-space Ewald contribution with one fused device kernel."""
    cp = workspace.cupy
    n_pairs = int(c.shape[0])
    if n_pairs == 0:
        return
    real_terms = workspace.real_terms(int(real_shell_count))
    n_shifts = int(real_terms.shifts.shape[0])
    if n_shifts == 0:
        return
    threads = 128
    n_entries = (int(order) + 1) * (2 * int(order) + 1)
    shared_bytes = 2 * n_entries * np.dtype(np.float64).itemsize
    terms = int(_DEFAULT_WOFZ_TERMS)
    h = math.sqrt(math.pi / float(terms + 1))
    quadrature = _wtrap_quadrature_table_cupy(int(cp.cuda.runtime.getDevice()), terms)
    _real_space_structural_raw_kernel(int(order))(
        (n_pairs,),
        (threads,),
        (
            np.int64(n_pairs),
            np.int32(int(order)),
            np.int64(n_shifts),
            cp.ascontiguousarray(c, dtype=cp.float64),
            cp.ascontiguousarray(same_plane.astype(cp.uint8, copy=False)),
            real_terms.shifts,
            real_terms.phase_xy,
            sums,
            np.float64(float(eta)),
            np.float64(float(k)),
            quadrature,
            np.int32(terms),
            np.float64(h),
            np.float64(math.pi / h),
        ),
        shared_mem=shared_bytes,
    )


_SHIFTED_RECIPROCAL_STRUCTURAL_CUDA_SOURCE = r"""

__device__ double _pyceles_warp_sum(const double value) {
    double out = value;
    for (int offset = 16; offset > 0; offset >>= 1) {
        out += __shfl_down_sync(0xffffffffu, out, offset);
    }
    return out;
}

__device__ void _pyceles_shifted_delta_sequence(
    const int order,
    const complex<double> gamma,
    const complex<double> x,
    const complex<double> root_x,
    const double z_offset,
    const bool series_height_eligible,
    const bool rayleigh_zero,
    const complex<double>* upper_gamma,
    const double* quadrature,
    const int terms,
    const double h,
    const double H,
    complex<double>* delta
) {
    const complex<double> scaled = gamma * z_offset;
    const complex<double> z_sq = scaled * scaled;
    const double scaled_abs_sq =
        scaled.real() * scaled.real() + scaled.imag() * scaled.imag();
    const bool use_series =
        series_height_eligible
        && !rayleigh_zero
        && scaled_abs_sq <= (PYCELES_SHIFTED_SERIES_SCALED_LIMIT
            * PYCELES_SHIFTED_SERIES_SCALED_LIMIT);

    if (use_series) {
        // Exact generalized-incomplete-gamma expansion:
        // Delta_n = sum_j ((gamma*z)^2/4)^j / j! * Gamma(1/2-n-j, x).
        // Horner evaluation avoids the removable (gamma*z)^-2 recurrence and
        // uses only pair-invariant coefficients staged during preparation.
        const complex<double> q = 0.25 * z_sq;
        for (int n = 0; n <= order; ++n) {
            complex<double> value = upper_gamma[PYCELES_SHIFTED_SERIES_TERMS + n];
            for (int j = PYCELES_SHIFTED_SERIES_TERMS - 1; j >= 0; --j) {
                value = upper_gamma[j + n] + q * value / (double)(j + 1);
            }
            delta[n] = value;
        }
        return;
    }

    const complex<double> z_arg =
        x.real() < 0.0 ? scaled : complex<double>(0.0, abs(scaled));
    const complex<double> exp_term = exp(-x + z_sq / (4.0 * x));
    if (x.real() < 0.0) {
        // The two Faddeeva arguments are conjugates.  Reflect the lower
        // half-plane value analytically before multiplying by exp_term; the
        // direct product otherwise becomes 0*inf for tall cells.
        const double a = fabs(root_x.imag());
        const double scaled_abs = fabs(scaled.real());
        const double b = scaled_abs / (2.0 * a);
        const complex<double> upper_w = _wtrap_wofz_one(
            complex<double>(a, b), quadrature, terms, h, H
        );
        const double prop_exp = exp(a * a - b * b);
        const complex<double> phase(cos(scaled_abs), sin(scaled_abs));
        delta[0] = sqrt(PYCELES_PI) * (
            phase + complex<double>(0.0, prop_exp * upper_w.imag())
        );
        if (order >= 1) {
            delta[1] = complex<double>(0.0, 2.0 * sqrt(PYCELES_PI) / scaled_abs) * (
                phase - complex<double>(prop_exp * upper_w.real(), 0.0)
            );
        }
    } else if (x.real() > 0.0) {
        // For evanescent orders z_arg is imaginary.  Once its lower argument
        // crosses the real axis, use the same Faddeeva reflection identity;
        // the physical Ewald product is exponentially decaying.
        const double a = sqrt(x.real());
        const double scaled_abs = fabs(z_arg.imag());
        const double c = scaled_abs / (2.0 * a);
        const double ev_exp = exp(-a * a - c * c);
        const complex<double> w_plus = _wtrap_wofz_one(
            complex<double>(0.0, a + c), quadrature, terms, h, H
        );
        complex<double> product_sum;
        complex<double> product_diff;
        if (c > a) {
            const complex<double> w_lower_reflected = _wtrap_wofz_one(
                complex<double>(0.0, c - a), quadrature, terms, h, H
            );
            const double lower_term = 2.0 * exp(-scaled_abs);
            product_sum = complex<double>(lower_term, 0.0)
                - ev_exp * w_lower_reflected + ev_exp * w_plus;
            product_diff = complex<double>(lower_term, 0.0)
                - ev_exp * w_lower_reflected - ev_exp * w_plus;
        } else {
            const complex<double> w_minus = _wtrap_wofz_one(
                complex<double>(0.0, a - c), quadrature, terms, h, H
            );
            product_sum = ev_exp * (w_minus + w_plus);
            product_diff = ev_exp * (w_minus - w_plus);
        }
        delta[0] = 0.5 * sqrt(PYCELES_PI) * product_sum;
        if (order >= 1) {
            delta[1] = sqrt(PYCELES_PI) / scaled_abs * product_diff;
        }
    } else {
        const complex<double> w_minus = _wtrap_wofz_one(
            -z_arg / (2.0 * root_x) + complex<double>(0.0, 1.0) * root_x,
            quadrature,
            terms,
            h,
            H
        );
        const complex<double> w_plus = _wtrap_wofz_one(
            z_arg / (2.0 * root_x) + complex<double>(0.0, 1.0) * root_x,
            quadrature,
            terms,
            h,
            H
        );
        delta[0] = 0.5 * sqrt(PYCELES_PI) * exp_term * (w_minus + w_plus);
        if (order >= 1) {
            delta[1] = complex<double>(0.0, sqrt(PYCELES_PI)) / z_arg * exp_term * (w_minus - w_plus);
        }
    }
    if (order == 0) {
        return;
    }
    complex<double> x_power = complex<double>(1.0, 0.0) / x;
    for (int idx = 2; idx <= order; ++idx) {
        delta[idx] = 4.0 / z_sq * (
            (1.5 - (double)idx) * delta[idx - 1]
            - delta[idx - 2]
            + root_x * x_power * exp_term
        );
        x_power /= x;
    }
}

extern "C" __global__ void pyceles_ewald_shifted_reciprocal_structural_c128(
    const long long n_pairs,
    const int order,
    const long long n_terms,
    const double* c,
    const unsigned char* same_plane,
    const double* kgt,
    const complex<double>* azimuth,
    const complex<double>* gamma,
    const complex<double>* xarg,
    const unsigned char* rayleigh_zero,
    const complex<double>* root_x,
    const double* rho_powers,
    const complex<double>* gamma_powers,
    const complex<double>* prefactor,
    const double* inverse_denominator,
    const complex<double>* upper_gamma,
    const int upper_gamma_stride,
    complex<double>* sums,
    const double k,
    const double eta,
    const double* quadrature,
    const int terms,
    const double h,
    const double H
) {
    const long long pair = (long long)blockIdx.x;
    if (pair >= n_pairs || same_plane[pair] != 0
        || order < 0 || order > PYCELES_SHIFTED_MAX_ORDER) {
        return;
    }
    const int tid = (int)threadIdx.x;
    const int width = 2 * order + 1;
    const int n_entries = (order + 1) * width;
    extern __shared__ double shared[];

    for (int idx = tid; idx < 2 * n_entries; idx += (int)blockDim.x) {
        shared[idx] = 0.0;
    }
    double* cz_powers = shared + 2 * n_entries;
    if (tid == 0) {
        cz_powers[0] = 1.0;
        const double cz_base = -k * c[3 * pair + 2];
        for (int exponent = 1; exponent <= order; ++exponent) {
            cz_powers[exponent] = cz_powers[exponent - 1] * cz_base;
        }
    }
    __syncthreads();

    const double cx = c[3 * pair + 0];
    const double cy = c[3 * pair + 1];
    const double cz = c[3 * pair + 2];
    const bool series_height_eligible =
        fabs(eta * cz) <= PYCELES_SHIFTED_SERIES_ETA_Z_LIMIT;
    const int lane = tid & 31;

    // The 128-thread launch consists of complete warps.  Each lane retains one
    // reciprocal term's delta sequence, then the warp combines contributions
    // before touching shared memory.  This preserves one sequence evaluation
    // per term while avoiding a contended double atomic from every lane.
    for (long long term_base = 0; term_base < n_terms; term_base += (long long)blockDim.x) {
        const long long term_idx = term_base + tid;
        const bool has_term = term_idx < n_terms;
        const unsigned int active_lanes = __ballot_sync(0xffffffffu, has_term);
        complex<double> phase(0.0, 0.0);
        complex<double> delta[PYCELES_SHIFTED_MAX_ORDER + 1];
        if (has_term) {
            const double kx = kgt[2 * term_idx + 0];
            const double ky = kgt[2 * term_idx + 1];
            const double phase_angle = -(cx * kx + cy * ky);
            double sin_phase;
            double cos_phase;
            sincos(phase_angle, &sin_phase, &cos_phase);
            phase = complex<double>(cos_phase, sin_phase);
            _pyceles_shifted_delta_sequence(
                order,
                gamma[term_idx],
                xarg[term_idx],
                root_x[term_idx],
                cz,
                series_height_eligible,
                rayleigh_zero[term_idx] != 0,
                upper_gamma + term_idx * (long long)upper_gamma_stride,
                quadrature,
                terms,
                h,
                H,
                delta
            );
        }

        for (int l = 0; l <= order; ++l) {
            for (int m = -l; m <= l; ++m) {
                const int entry = l * width + (m + order);
                complex<double> contrib(0.0, 0.0);
                if (has_term) {
                    const int abs_m = m < 0 ? -m : m;
                    complex<double> acc(0.0, 0.0);
                    const int n_max = l - abs_m;
                    for (int n = 0; n <= n_max; ++n) {
                        double terms_acc = 0.0;
                        const int s_stop = min(n_max, 2 * n);
                        const int s_start = ((n - n_max) & 1) == 0 ? n : n + 1;
                        for (int s_val = s_start; s_val <= s_stop; s_val += 2) {
                            const long long denominator_idx =
                                ((long long)entry * (order + 1) + n) * (order + 1) + s_val;
                            terms_acc += cz_powers[2 * n - s_val]
                                * rho_powers[term_idx * (order + 1) + (l - s_val)]
                                * inverse_denominator[denominator_idx];
                        }
                        acc += gamma_powers[term_idx * (order + 1) + n]
                            * delta[n]
                            * terms_acc;
                    }
                    contrib = prefactor[entry]
                        * phase
                        * azimuth[term_idx * width + (m + order)]
                        * acc;
                }
                const double re = _pyceles_warp_sum(contrib.real());
                const double im = _pyceles_warp_sum(contrib.imag());
                if (lane == 0 && active_lanes != 0) {
                    atomicAdd(&shared[2 * entry], re);
                    atomicAdd(&shared[2 * entry + 1], im);
                }
            }
        }
    }
    __syncthreads();

    for (int entry = tid; entry < n_entries; entry += (int)blockDim.x) {
        const double re = shared[2 * entry];
        const double im = shared[2 * entry + 1];
        if (re != 0.0 || im != 0.0) {
            sums[pair * n_entries + entry] += complex<double>(re, im);
        }
    }
}
"""

_SHIFTED_RECIPROCAL_COMPACT_TERMS_CUDA_SOURCE = r"""

extern "C" __global__ void pyceles_ewald_shifted_reciprocal_compact_terms_c128(
    const long long n_sources,
    const int order,
    const long long n_terms,
    const double* source_positions,
    const double plane_z,
    const double* kgt,
    const complex<double>* azimuth,
    const complex<double>* gamma,
    const complex<double>* xarg,
    const unsigned char* rayleigh_zero,
    const complex<double>* root_x,
    const double* rho_powers,
    const complex<double>* gamma_powers,
    const complex<double>* prefactor,
    const double* inverse_denominator,
    const complex<double>* upper_gamma,
    const int upper_gamma_stride,
    complex<double>* compact_terms,
    const double k,
    const double eta,
    const double* quadrature,
    const int terms,
    const double h,
    const double H
) {
    const long long source = (long long)blockIdx.x;
    const int tid = (int)threadIdx.x;
    if (source >= n_sources || order < 0 || order > PYCELES_SHIFTED_MAX_ORDER) {
        return;
    }
    const bool has_term = tid < n_terms;

    extern __shared__ double cz_powers[];
    const double cz = source_positions[3 * source + 2] - plane_z;
    if (tid == 0) {
        cz_powers[0] = 1.0;
        const double cz_base = -k * cz;
        for (int exponent = 1; exponent <= order; ++exponent) {
            cz_powers[exponent] = cz_powers[exponent - 1] * cz_base;
        }
    }
    __syncthreads();
    if (!has_term) {
        return;
    }

    const double cx = source_positions[3 * source + 0];
    const double cy = source_positions[3 * source + 1];
    const double kx = kgt[2 * tid + 0];
    const double ky = kgt[2 * tid + 1];
    const double phase_angle = -(cx * kx + cy * ky);
    double sin_phase;
    double cos_phase;
    sincos(phase_angle, &sin_phase, &cos_phase);
    const complex<double> source_phase(cos_phase, sin_phase);

    const bool series_height_eligible =
        fabs(eta * cz) <= PYCELES_SHIFTED_SERIES_ETA_Z_LIMIT;
    complex<double> delta[PYCELES_SHIFTED_MAX_ORDER + 1];
    _pyceles_shifted_delta_sequence(
        order,
        gamma[tid],
        xarg[tid],
        root_x[tid],
        cz,
        series_height_eligible,
        rayleigh_zero[tid] != 0,
        upper_gamma + (long long)tid * upper_gamma_stride,
        quadrature,
        terms,
        h,
        H,
        delta
    );

    const int width = 2 * order + 1;
    const int compact_width = (order + 1) * (order + 1);
    const long long output_base =
        ((source * n_terms + (long long)tid) * (long long)compact_width);

    for (int l = 0; l <= order; ++l) {
        for (int m = -l; m <= l; ++m) {
            const int entry = l * width + (m + order);
            const int compact_entry = l * l + (m + l);
            const int abs_m = m < 0 ? -m : m;
            const int n_max = l - abs_m;
            complex<double> acc(0.0, 0.0);
            for (int n = 0; n <= n_max; ++n) {
                double terms_acc = 0.0;
                const int s_stop = min(n_max, 2 * n);
                const int s_start = ((n - n_max) & 1) == 0 ? n : n + 1;
                for (int s_val = s_start; s_val <= s_stop; s_val += 2) {
                    const long long denominator_idx =
                        ((long long)entry * (order + 1) + n) * (order + 1) + s_val;
                    terms_acc += cz_powers[2 * n - s_val]
                        * rho_powers[(long long)tid * (order + 1) + (l - s_val)]
                        * inverse_denominator[denominator_idx];
                }
                acc += gamma_powers[(long long)tid * (order + 1) + n]
                    * delta[n]
                    * terms_acc;
            }
            compact_terms[output_base + compact_entry] = prefactor[entry]
                * source_phase
                * azimuth[(long long)tid * width + (m + order)]
                * acc;
        }
    }
}
"""


@cache
def _shifted_reciprocal_compact_terms_raw_kernel(max_order: int) -> Any:
    cp, _ = import_cupy()
    order = int(max_order)
    source = (
        _WTRAP_DEVICE_CUDA_SOURCE
        + _EWALD_DEVICE_CONSTANTS
        + f"\n#define PYCELES_SHIFTED_MAX_ORDER {order}\n"
        + f"#define PYCELES_SHIFTED_SERIES_TERMS {_SHIFTED_DELTA_SERIES_TERMS}\n"
        + (
            "#define PYCELES_SHIFTED_SERIES_SCALED_LIMIT "
            f"{_SHIFTED_DELTA_SERIES_SCALED_LIMIT:.17g}\n"
        )
        + (f"#define PYCELES_SHIFTED_SERIES_ETA_Z_LIMIT {_SHIFTED_DELTA_SERIES_ETA_Z_LIMIT:.17g}\n")
        + _SHIFTED_RECIPROCAL_STRUCTURAL_CUDA_SOURCE
        + _SHIFTED_RECIPROCAL_COMPACT_TERMS_CUDA_SOURCE
    )
    return cp.RawKernel(source, "pyceles_ewald_shifted_reciprocal_compact_terms_c128")


@cache
def _shifted_reciprocal_structural_raw_kernel(max_order: int) -> Any:
    cp, _ = import_cupy()
    order = int(max_order)
    source = (
        _WTRAP_DEVICE_CUDA_SOURCE
        + _EWALD_DEVICE_CONSTANTS
        + f"\n#define PYCELES_SHIFTED_MAX_ORDER {order}\n"
        + f"#define PYCELES_SHIFTED_SERIES_TERMS {_SHIFTED_DELTA_SERIES_TERMS}\n"
        + (
            "#define PYCELES_SHIFTED_SERIES_SCALED_LIMIT "
            f"{_SHIFTED_DELTA_SERIES_SCALED_LIMIT:.17g}\n"
        )
        + (f"#define PYCELES_SHIFTED_SERIES_ETA_Z_LIMIT {_SHIFTED_DELTA_SERIES_ETA_Z_LIMIT:.17g}\n")
        + _SHIFTED_RECIPROCAL_STRUCTURAL_CUDA_SOURCE
    )
    return cp.RawKernel(source, "pyceles_ewald_shifted_reciprocal_structural_c128")


def _shifted_reciprocal_compact_terms_cupy(
    *,
    source_positions: Any,
    plane_z: float,
    workspace: CupyEwaldShellWorkspace,
    reciprocal_shell_count: int,
    order: int,
    term_start: int = 0,
    term_stop: int | None = None,
) -> Any:
    """Return per-source/per-reciprocal-term compact shifted Ewald channels.

    Unlike :func:`ewald_structural_sums_2d_fixed_cupy`, this helper deliberately
    does *not* sum reciprocal orders or apply a destination lateral phase.  For
    destinations on one horizontal plane those operations are separable, so a
    near-field caller can evaluate the expensive shifted multipole recurrence
    once per source/order and synthesize arbitrarily many lateral points with a
    matrix product.  Only valid ``(l,m)`` channels are materialized, ordered as
    ``l*l + (m+l)``.

    The sources must have nonzero vertical offset from ``plane_z``; same-plane
    reciprocal terms use a different analytic formula and remain on the generic
    structural path.
    """
    cp = workspace.cupy
    order_i = int(order)
    if order_i < 0:
        raise ValueError(f"`order` must be >= 0. Got {order!r}.")
    sources = cp.ascontiguousarray(cp.asarray(source_positions, dtype=cp.float64).reshape(-1, 3))
    n_sources = int(sources.shape[0])
    reciprocal_terms = workspace.reciprocal_terms(int(reciprocal_shell_count))
    if reciprocal_terms.has_rayleigh_zero:
        raise ValueError(
            "shifted reciprocal integrals are singular at a Rayleigh/Wood anomaly; "
            "move away from the anomaly or use the same-plane formula."
        )
    total_terms = int(reciprocal_terms.rho.shape[0])
    start = max(0, int(term_start))
    stop = total_terms if term_stop is None else min(total_terms, int(term_stop))
    if stop < start:
        raise ValueError(f"Expected term_stop >= term_start; got {start} and {stop}.")
    n_terms = stop - start
    compact_width = (order_i + 1) ** 2
    out = cp.empty((n_sources, n_terms, compact_width), dtype=cp.complex128)
    if n_sources == 0 or n_terms == 0:
        return out
    if n_terms > 128:
        raise ValueError(
            "The compact shifted reciprocal kernel accepts at most 128 terms per launch; "
            f"got {n_terms}. Chunk the reciprocal range before calling it."
        )

    shifted_tables = workspace.shifted_reciprocal_tables(
        int(reciprocal_shell_count),
        order_i,
    )
    upper_stride = order_i + _SHIFTED_DELTA_SERIES_TERMS + 1
    upper_gamma = workspace.upper_gamma_terms(
        int(reciprocal_shell_count),
        upper_stride - 1,
    )
    terms = int(_DEFAULT_WOFZ_TERMS)
    h = math.sqrt(math.pi / float(terms + 1))
    quadrature = _wtrap_quadrature_table_cupy(int(cp.cuda.runtime.getDevice()), terms)
    # RawKernel arguments are pointers, not strided array views.  Materialize
    # contiguous term slices so chunks after the first retain the same term
    # indexing as the full reciprocal tables.
    kgt = cp.ascontiguousarray(reciprocal_terms.kgt[start:stop])
    azimuth = cp.ascontiguousarray(shifted_tables.azimuth[start:stop])
    gamma = cp.ascontiguousarray(reciprocal_terms.gamma[start:stop])
    xarg = cp.ascontiguousarray(reciprocal_terms.xarg[start:stop])
    rayleigh_zero = cp.ascontiguousarray(reciprocal_terms.rayleigh_zero[start:stop])
    root_x = cp.ascontiguousarray(shifted_tables.root_x[start:stop])
    rho_powers = cp.ascontiguousarray(shifted_tables.rho_powers[start:stop])
    gamma_powers = cp.ascontiguousarray(shifted_tables.gamma_powers[start:stop])
    upper_gamma_chunk = cp.ascontiguousarray(upper_gamma[start:stop])
    _shifted_reciprocal_compact_terms_raw_kernel(order_i)(
        (n_sources,),
        (128,),
        (
            np.int64(n_sources),
            np.int32(order_i),
            np.int64(n_terms),
            sources,
            np.float64(float(plane_z)),
            kgt,
            azimuth,
            gamma,
            xarg,
            rayleigh_zero,
            root_x,
            rho_powers,
            gamma_powers,
            shifted_tables.prefactor,
            shifted_tables.inverse_denominator,
            upper_gamma_chunk,
            np.int32(upper_stride),
            out,
            np.float64(float(workspace.k)),
            np.float64(float(workspace.eta)),
            quadrature,
            np.int32(terms),
            np.float64(h),
            np.float64(math.pi / h),
        ),
        shared_mem=(order_i + 1) * np.dtype(np.float64).itemsize,
    )
    return out


def _add_shifted_reciprocal_structural_sums_cupy(
    *,
    c: Any,
    same_plane: Any,
    sums: Any,
    workspace: CupyEwaldShellWorkspace,
    reciprocal_shell_count: int,
    order: int,
    k: float,
) -> None:
    """Add shifted-pair reciprocal Ewald terms with one fused device kernel."""
    cp = workspace.cupy
    n_pairs = int(c.shape[0])
    if n_pairs == 0:
        return
    reciprocal_terms = workspace.reciprocal_terms(int(reciprocal_shell_count))
    if reciprocal_terms.has_rayleigh_zero:
        raise ValueError(
            "shifted reciprocal integrals are singular at a Rayleigh/Wood anomaly; "
            "move away from the anomaly or use the same-plane formula."
        )
    shifted_tables = workspace.shifted_reciprocal_tables(int(reciprocal_shell_count), int(order))
    upper_gamma = workspace.upper_gamma_terms(
        int(reciprocal_shell_count),
        int(order) + _SHIFTED_DELTA_SERIES_TERMS,
    )
    n_terms = int(reciprocal_terms.rho.shape[0])
    if n_terms == 0:
        return
    threads = 128
    n_entries = (int(order) + 1) * (2 * int(order) + 1)
    shared_bytes = (2 * n_entries + int(order) + 1) * np.dtype(np.float64).itemsize
    terms = int(_DEFAULT_WOFZ_TERMS)
    h = math.sqrt(math.pi / float(terms + 1))
    quadrature = _wtrap_quadrature_table_cupy(int(cp.cuda.runtime.getDevice()), terms)
    _shifted_reciprocal_structural_raw_kernel(int(order))(
        (n_pairs,),
        (threads,),
        (
            np.int64(n_pairs),
            np.int32(int(order)),
            np.int64(n_terms),
            cp.ascontiguousarray(c, dtype=cp.float64),
            cp.ascontiguousarray(same_plane.astype(cp.uint8, copy=False)),
            reciprocal_terms.kgt,
            shifted_tables.azimuth,
            reciprocal_terms.gamma,
            reciprocal_terms.xarg,
            reciprocal_terms.rayleigh_zero,
            shifted_tables.root_x,
            shifted_tables.rho_powers,
            shifted_tables.gamma_powers,
            shifted_tables.prefactor,
            shifted_tables.inverse_denominator,
            upper_gamma,
            np.int32(int(order) + _SHIFTED_DELTA_SERIES_TERMS + 1),
            sums,
            np.float64(float(k)),
            np.float64(float(workspace.eta)),
            quadrature,
            np.int32(terms),
            np.float64(h),
            np.float64(math.pi / h),
        ),
        shared_mem=shared_bytes,
    )


def ewald_structural_sums_2d_fixed_cupy(
    *,
    relative_source_minus_destination: Any,
    lmax_struct: int,
    structural_order: int | None = None,
    workspace: CupyEwaldShellWorkspace,
    real_shell_count: int,
    reciprocal_shell_count: int,
    coordinate_scale: float = 0.0,
    include_shifted_reciprocal: bool = True,
) -> Any:
    """Evaluate scalar periodic Ewald tables for source/destination pairs.

    Parameters
    ----------
    relative_source_minus_destination:
        CuPy-compatible array with shape ``(n_pairs, 3)`` storing
        ``source - destination``.  This sign convention matches the NumPy
        structural-sum batch helper and the periodic near-field evaluator.
    lmax_struct:
        Historical structural-table capacity.  Without ``structural_order``
        the evaluator computes scalar degrees through ``2*lmax_struct``.
    structural_order:
        Optional exact maximum scalar degree, bounded by the capacity above.
        This lets point-local ``l=1`` projection stop at
        ``p = lmax_source + 1`` even when that degree is odd; no physical
        coefficient is rounded or truncated by the capacity convention.
    workspace:
        Device workspace owning the lattice, wavenumber, Ewald splitting
        parameter, and all tables derived from them.

    Notes
    -----
    The returned table does not include the same-particle central-point Ewald
    correction.  Operator callers patch that correction only for true
    source==destination self blocks; point-field evaluators must not add it.
    """
    cp = workspace.cupy
    k = float(workspace.k)
    eta = float(workspace.eta)
    c = cp.asarray(relative_source_minus_destination, dtype=cp.float64).reshape(-1, 3)
    n_pairs = int(c.shape[0])
    maximum_order = 2 * int(lmax_struct)
    if maximum_order < 0:
        raise ValueError(f"`lmax_struct` must be >= 0. Got {lmax_struct!r}.")
    order = maximum_order if structural_order is None else int(structural_order)
    if order < 0 or order > maximum_order:
        raise ValueError(
            "`structural_order` must lie between 0 and 2*lmax_struct. "
            f"Got structural_order={structural_order!r}, lmax_struct={lmax_struct!r}."
        )
    offset = order
    sums = cp.zeros((n_pairs, order + 1, 2 * order + 1), dtype=cp.complex128)
    if n_pairs == 0:
        return sums

    cxy = c[:, :2]
    same_plane_atol = same_plane_z_tolerance(float(k), coordinate_scale=float(coordinate_scale))
    cz_raw = c[:, 2]
    same_plane = cp.abs(cz_raw) <= float(same_plane_atol)
    # Keep same-plane classification local to the structural evaluator.  The
    # mask is needed below regardless, and retaining caller-side pair-index
    # caches can otherwise turn matrix-free source batching into O(N^2) state.
    same_idx = cp.nonzero(same_plane)[0].astype(cp.int32, copy=False)
    c = c.copy()
    c[:, 2] = cp.where(same_plane, 0.0, cz_raw)

    n_same = int(same_idx.size)
    if n_same:
        max_same_n = max(0, order // 2)
        max_gamma_index = max_same_n
        if n_same < n_pairs and include_shifted_reciprocal:
            max_gamma_index = max(max_gamma_index, int(order) + _SHIFTED_DELTA_SERIES_TERMS)
        reciprocal_terms = workspace.reciprocal_terms(int(reciprocal_shell_count))
        gamma_fun_all = workspace.upper_gamma_terms(
            int(reciprocal_shell_count),
            max_gamma_index,
        )
        for shell in range(int(reciprocal_shell_count) + 1):
            start = reciprocal_terms.shell_offsets[shell]
            stop = reciprocal_terms.shell_offsets[shell + 1]
            kgt = reciprocal_terms.kgt[start:stop]
            rho = reciprocal_terms.rho[start:stop]
            phi = reciprocal_terms.phi[start:stop]
            gamma = reciprocal_terms.gamma[start:stop]

            phase_same = cp.exp(-1j * (cxy[same_idx] @ kgt.T))
            exp_m_phi = {m: cp.exp(1j * m * phi) for m in range(-order, order + 1)}
            gamma_fun = gamma_fun_all[start:stop, : max_same_n + 1]
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

    if n_same < n_pairs and include_shifted_reciprocal:
        _add_shifted_reciprocal_structural_sums_cupy(
            c=c,
            same_plane=same_plane,
            sums=sums,
            workspace=workspace,
            reciprocal_shell_count=int(reciprocal_shell_count),
            order=int(order),
            k=float(k),
        )

    _add_real_space_structural_sums_cupy(
        c=c,
        same_plane=same_plane,
        sums=sums,
        workspace=workspace,
        real_shell_count=int(real_shell_count),
        order=int(order),
        eta=float(eta),
        k=float(k),
    )
    return sums


__all__ = [
    "CupyEwaldShellWorkspace",
    "ewald_structural_sums_2d_fixed_cupy",
]
