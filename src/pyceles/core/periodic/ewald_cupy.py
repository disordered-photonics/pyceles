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
    same_plane_z_tolerance,
    structural_sum_m_normalization,
    upper_gamma_sequence,
)
from pyceles.core.periodic.special_cupy import (
    _DEFAULT_WOFZ_TERMS,
    _WTRAP_DEVICE_CUDA_SOURCE,
)


@dataclass
class CupyReciprocalShell:
    """Device-side reciprocal-shell data for one Chebyshev-index shell."""

    kgt: Any
    rho: Any
    phi: Any
    gamma: Any
    xarg: Any


@dataclass
class CupyReciprocalTerms:
    """Device-side reciprocal data for all shells in one fixed range."""

    kgt: Any
    rho: Any
    phi: Any
    gamma: Any


@dataclass
class CupyRealShell:
    """Device-side real-lattice shell data for one Chebyshev-index shell."""

    shifts: Any
    phase_xy: Any


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
    reciprocal_cache: dict[int, CupyReciprocalShell] = field(default_factory=dict)
    reciprocal_terms_cache: dict[int, CupyReciprocalTerms] = field(default_factory=dict)
    real_cache: dict[int, CupyRealShell] = field(default_factory=dict)
    real_terms_cache: dict[int, CupyRealTerms] = field(default_factory=dict)
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

    def reciprocal_terms(self, shell_count: int) -> CupyReciprocalTerms:
        """Return contiguous reciprocal vectors through one fixed shell count."""
        count = int(shell_count)
        cached = self.reciprocal_terms_cache.get(count)
        if cached is not None:
            return cached
        cp = self.cupy
        reciprocal_indices: list[tuple[int, int]] = []
        for shell in range(count + 1):
            reciprocal_indices.extend(chebyshev_shell_indices(shell))
        reciprocal = np.asarray(
            [p * self.lattice.b1 + q * self.lattice.b2 for p, q in reciprocal_indices],
            dtype=np.float64,
        )
        kgt_np = np.asarray(self.k_parallel, dtype=np.float64).reshape(2)[None, :] + reciprocal
        kgt = cp.asarray(kgt_np, dtype=cp.float64)
        rho = cp.linalg.norm(kgt, axis=1)
        phi = cp.arctan2(kgt[:, 1], kgt[:, 0])
        gamma = cp.sqrt((float(self.k) * float(self.k) - rho * rho) + 0.0j)
        gamma = cp.where(gamma == 0.0, gamma + 1.0e-10j, gamma).astype(cp.complex128)
        out = CupyReciprocalTerms(kgt=kgt, rho=rho, phi=phi, gamma=gamma)
        self.reciprocal_terms_cache[count] = out
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
        shifts = cp.asarray(shifts_np, dtype=cp.float64)
        kp = cp.asarray(np.asarray(self.k_parallel, dtype=np.float64).reshape(2), dtype=cp.float64)
        phase_xy = cp.exp(1j * (shifts[:, :2] @ kp)).astype(cp.complex128)
        out = CupyRealTerms(shifts=shifts, phase_xy=phase_xy)
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

extern "C" __device__ void _pyceles_legendre_table(
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

extern "C" __device__ void _pyceles_real_integrals(
    const double r,
    const int order,
    const double eta,
    const double k,
    const int terms,
    const double h,
    const double H,
    double* integral
) {
    const double alpha = k * k / (4.0 * eta * eta);
    const double root_alpha = sqrt(alpha);
    const double kr = k * r;
    const complex<double> z(root_alpha, kr / (2.0 * root_alpha));
    const complex<double> w = _wtrap_wofz_one(z, terms, h, H);
    const double exp_term = exp(alpha - (kr * kr) / (4.0 * alpha));
    double vals[PYCELES_REAL_MAX_ORDER + 2];
    vals[0] = sqrt(PYCELES_PI) * exp_term * w.imag();
    vals[1] = sqrt(PYCELES_PI) * 2.0 / kr * exp_term * w.real();
    const double inv = 2.0 / kr;
    const double inv2 = inv * inv;
    for (int idx = 2; idx < order + 2; ++idx) {
        vals[idx] = inv2 * (
            0.5 * (double)(2 * (idx - 2) + 1) * vals[idx - 1]
            - vals[idx - 2]
            + pow(alpha, -((double)(idx - 2)) - 0.5) * exp_term
        );
    }
    double half_power = pow(0.5, 1.5);
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
    const int terms,
    const double h,
    const double H
) {
    const long long pair = (long long)blockIdx.x;
    if (pair >= n_pairs) {
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
        _pyceles_real_integrals(r, order, eta, k, terms, h, H, integrals);
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
                const complex<double> azimuth(cos(angle), sin(angle));
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
            np.int32(terms),
            np.float64(h),
            np.float64(math.pi / h),
        ),
        shared_mem=shared_bytes,
    )


_SHIFTED_RECIPROCAL_STRUCTURAL_CUDA_SOURCE = r"""

extern "C" __device__ double _pyceles_factorial_double(const int n) {
    double out = 1.0;
    for (int idx = 2; idx <= n; ++idx) {
        out *= (double)idx;
    }
    return out;
}

extern "C" __device__ double _pyceles_int_power_real(const double base, const int exponent) {
    if (exponent == 0) {
        return 1.0;
    }
    double out = 1.0;
    const int n = exponent < 0 ? -exponent : exponent;
    for (int idx = 0; idx < n; ++idx) {
        out *= base;
    }
    return exponent < 0 ? 1.0 / out : out;
}

extern "C" __device__ complex<double> _pyceles_int_power_complex(
    const complex<double> base,
    const int exponent
) {
    if (exponent == 0) {
        return complex<double>(1.0, 0.0);
    }
    complex<double> out(1.0, 0.0);
    const int n = exponent < 0 ? -exponent : exponent;
    for (int idx = 0; idx < n; ++idx) {
        out *= base;
    }
    return exponent < 0 ? complex<double>(1.0, 0.0) / out : out;
}

extern "C" __device__ complex<double> _pyceles_minus_i_power(const int exponent) {
    int mod = exponent % 4;
    if (mod < 0) {
        mod += 4;
    }
    if (mod == 0) {
        return complex<double>(1.0, 0.0);
    }
    if (mod == 1) {
        return complex<double>(0.0, -1.0);
    }
    if (mod == 2) {
        return complex<double>(-1.0, 0.0);
    }
    return complex<double>(0.0, 1.0);
}

extern "C" __device__ double _pyceles_structural_m_norm(const int m) {
    const double sign = (m >= 0 && ((m & 1) != 0)) ? -1.0 : 1.0;
    return sqrt(2.0 * PYCELES_PI) * sign;
}

extern "C" __device__ void _pyceles_shifted_delta_sequence(
    const int order,
    const complex<double> gamma,
    const double z_offset,
    const double eta,
    const int terms,
    const double h,
    const double H,
    complex<double>* delta
) {
    const complex<double> x = -(gamma * gamma) / (4.0 * eta * eta);
    const complex<double> root_x =
        x.real() < 0.0 ? complex<double>(0.0, -sqrt(abs(x))) : sqrt(x);
    const complex<double> scaled = gamma * z_offset;
    const complex<double> z_arg =
        x.real() < 0.0 ? scaled : complex<double>(0.0, abs(scaled));
    const complex<double> z_sq = scaled * scaled;
    const complex<double> exp_term = exp(-x + z_sq / (4.0 * x));
    const complex<double> w_minus = _wtrap_wofz_one(
        -z_arg / (2.0 * root_x) + complex<double>(0.0, 1.0) * root_x,
        terms,
        h,
        H
    );
    const complex<double> w_plus = _wtrap_wofz_one(
        z_arg / (2.0 * root_x) + complex<double>(0.0, 1.0) * root_x,
        terms,
        h,
        H
    );
    delta[0] = 0.5 * sqrt(PYCELES_PI) * exp_term * (w_minus + w_plus);
    if (order == 0) {
        return;
    }
    delta[1] = complex<double>(0.0, sqrt(PYCELES_PI)) / z_arg * exp_term * (w_minus - w_plus);
    for (int idx = 2; idx <= order; ++idx) {
        delta[idx] = 4.0 / z_sq * (
            (1.5 - (double)idx) * delta[idx - 1]
            - delta[idx - 2]
            + root_x * pow(x, 1 - idx) * exp_term
        );
    }
}

extern "C" __global__ void pyceles_ewald_shifted_reciprocal_structural_c128(
    const long long n_pairs,
    const int order,
    const long long n_terms,
    const double* c,
    const unsigned char* same_plane,
    const double* kgt,
    const double* rho,
    const double* phi,
    const complex<double>* gamma,
    complex<double>* sums,
    const double area,
    const double eta,
    const double k,
    const int terms,
    const double h,
    const double H
) {
    const long long pair = (long long)blockIdx.x;
    if (pair >= n_pairs || same_plane[pair] != 0) {
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

    for (long long term_idx = tid; term_idx < n_terms; term_idx += (long long)blockDim.x) {
        const double kx = kgt[2 * term_idx + 0];
        const double ky = kgt[2 * term_idx + 1];
        const double phase_angle = -(cx * kx + cy * ky);
        const complex<double> phase(cos(phase_angle), sin(phase_angle));
        const double rho_g = rho[term_idx];
        const double phi_g = phi[term_idx];
        const complex<double> gamma_g = gamma[term_idx];
        const complex<double> gamma_over_k = gamma_g / k;

        complex<double> delta[PYCELES_SHIFTED_MAX_ORDER + 1];
        _pyceles_shifted_delta_sequence(order, gamma_g, cz, eta, terms, h, H, delta);

        for (int l = 0; l <= order; ++l) {
            const double root_l = sqrt(2.0 * l + 1.0);
            const double pow_minus_two = _pyceles_int_power_real(-2.0, l);
            for (int m = -l; m <= l; ++m) {
                const int abs_m = m < 0 ? -m : m;
                const double root = root_l
                    * sqrt(_pyceles_factorial_double(l - m))
                    * sqrt(_pyceles_factorial_double(l + m));
                const complex<double> prefactor =
                    _pyceles_minus_i_power(m) * root / (pow_minus_two * area * k * k);
                complex<double> acc(0.0, 0.0);
                const int n_max = l - abs_m;
                for (int n = 0; n <= n_max; ++n) {
                    complex<double> terms_acc(0.0, 0.0);
                    const int s_stop = min(n_max, 2 * n);
                    for (int s_val = n; s_val <= s_stop; ++s_val) {
                        if (((s_val - n_max) & 1) != 0) {
                            continue;
                        }
                        const int a = 2 * n - s_val;
                        const int b = s_val - n;
                        const int c1 = (l + abs_m - s_val) / 2;
                        const int c2 = (l - abs_m - s_val) / 2;
                        const double denom =
                            _pyceles_factorial_double(a)
                            * _pyceles_factorial_double(b)
                            * _pyceles_factorial_double(c1)
                            * _pyceles_factorial_double(c2);
                        const double term =
                            _pyceles_int_power_real(-k * cz, 2 * n - s_val)
                            * _pyceles_int_power_real(rho_g / k, l - s_val)
                            / denom;
                        terms_acc += complex<double>(term, 0.0);
                    }
                    acc += _pyceles_int_power_complex(gamma_over_k, 2 * n - 1)
                        * delta[n]
                        * terms_acc;
                }
                const double angle = (double)m * phi_g;
                const complex<double> azimuth(cos(angle), sin(angle));
                const complex<double> contrib =
                    _pyceles_structural_m_norm(m) * prefactor * phase * azimuth * acc;
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
def _shifted_reciprocal_structural_raw_kernel(max_order: int) -> Any:
    cp, _ = import_cupy()
    order = int(max_order)
    source = (
        _WTRAP_DEVICE_CUDA_SOURCE
        + _EWALD_DEVICE_CONSTANTS
        + f"\n#define PYCELES_SHIFTED_MAX_ORDER {order}\n"
        + _SHIFTED_RECIPROCAL_STRUCTURAL_CUDA_SOURCE
    )
    return cp.RawKernel(source, "pyceles_ewald_shifted_reciprocal_structural_c128")


def _add_shifted_reciprocal_structural_sums_cupy(
    *,
    c: Any,
    same_plane: Any,
    sums: Any,
    workspace: CupyEwaldShellWorkspace,
    reciprocal_shell_count: int,
    order: int,
    eta: float,
    k: float,
) -> None:
    """Add shifted-pair reciprocal Ewald terms with one fused device kernel."""
    cp = workspace.cupy
    n_pairs = int(c.shape[0])
    if n_pairs == 0:
        return
    reciprocal_terms = workspace.reciprocal_terms(int(reciprocal_shell_count))
    n_terms = int(reciprocal_terms.rho.shape[0])
    if n_terms == 0:
        return
    threads = 128
    n_entries = (int(order) + 1) * (2 * int(order) + 1)
    shared_bytes = 2 * n_entries * np.dtype(np.float64).itemsize
    terms = int(_DEFAULT_WOFZ_TERMS)
    h = math.sqrt(math.pi / float(terms + 1))
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
            reciprocal_terms.rho,
            reciprocal_terms.phi,
            reciprocal_terms.gamma,
            sums,
            np.float64(float(workspace.lattice.area)),
            np.float64(float(eta)),
            np.float64(float(k)),
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
    k: float,
    eta: float,
    workspace: CupyEwaldShellWorkspace,
    real_shell_count: int,
    reciprocal_shell_count: int,
    coordinate_scale: float = 0.0,
    same_plane_pair_indices: Any | None = None,
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
    # Avoid a host-side `any(...).get()` in the hot structural-sum path.  Coupling
    # callers can also provide the compact same-plane pair index list from their
    # host-side particle metadata, avoiding a device-side nonzero scan.
    if same_plane_pair_indices is None:
        same_idx = cp.nonzero(same_plane)[0]
    else:
        same_idx = cp.asarray(same_plane_pair_indices, dtype=cp.int64).reshape(-1)
    c = c.copy()
    c[:, 2] = cp.where(same_plane, 0.0, cz_raw)

    max_same_n = max(0, order // 2)
    for shell in range(int(reciprocal_shell_count) + 1):
        shell_data = workspace.reciprocal_shell(shell)
        kgt = shell_data.kgt
        rho = shell_data.rho
        phi = shell_data.phi
        gamma = shell_data.gamma

        phase_same = cp.exp(-1j * (cxy[same_idx] @ kgt.T))
        exp_m_phi = {m: cp.exp(1j * m * phi) for m in range(-order, order + 1)}
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

    _add_shifted_reciprocal_structural_sums_cupy(
        c=c,
        same_plane=same_plane,
        sums=sums,
        workspace=workspace,
        reciprocal_shell_count=int(reciprocal_shell_count),
        order=int(order),
        eta=float(eta),
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
    "CupyRealShell",
    "CupyReciprocalShell",
    "ewald_structural_sums_2d_fixed_cupy",
]
