"""Fused CuPy pairwise coupling for direct GPU matvecs.

The direct pairwise path assigns one CUDA lane to one destination particle and
one block to one scalar output mode. Sources are visited in the same launch,
so each output has a unique owner and no atomics are required. A small
compile-time RHS tile reuses translation work for block right-hand sides.

The scalar-mode decomposition is deliberate. A general output-mode tile was
useful while exploring the kernel, with workload-dependent tradeoffs. One
scalar mode provides a deliberately simple schedule for the supported
workloads; it is not a claim that larger tiles never win. RHS tiling reuses
the same translation rather than partitioning output modes. Neither choice
restricts the particle T matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from operator import index
from typing import Any

import numpy as np

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
)

from .mode_metadata import mode_metadata_tables, mode_pair_p_range_tables

COMPLEX128_DTYPE = np.dtype(np.complex128)


@cache
def _source_parallel_kernel(
    lmax: int, dtype_str: str, adjoint: bool = False, rhs_tile_size: int = 1
):
    """Compile one fixed scalar-mode kernel for ``W`` or ``W^H``."""

    cupy, _ = import_cupy()
    dtype = np.dtype(dtype_str)
    if dtype == np.dtype(np.complex64):
        real_type = "float"
        complex_type = "complex<float>"
        math = {
            "atan2": "atan2f",
            "floor": "floorf",
            "max": "fmaxf",
            "sincos": "sincosf",
            "sqrt": "sqrtf",
        }
    elif dtype == np.dtype(np.complex128):
        real_type = "double"
        complex_type = "complex<double>"
        math = {
            "atan2": "atan2",
            "floor": "floor",
            "max": "fmax",
            "sincos": "sincos",
            "sqrt": "sqrt",
        }
    else:
        raise TypeError(f"Unsupported CuPy raw-kernel dtype {dtype!r}.")

    lmax_i = int(lmax)
    rhs_tile = index(rhs_tile_size)
    if rhs_tile < 1:
        raise ValueError("rhs_tile_size must be positive.")
    nmodes = n_modes(lmax_i)
    scalar_modes = nmodes // 2
    n_orders = 2 * lmax_i + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    action = "adjoint" if adjoint else "forward"
    name = f"pairwise_source_parallel_{action}_{real_type}_l{lmax_i}_r{rhs_tile}"

    # W^H reverses the displacement, transposes the mode pair, and conjugates
    # the scalar translation coefficient.
    delta = "m1 - mode_m[n2]" if adjoint else "mode_m[n2] - m1"
    pair = f"n2 * {nmodes} + n1" if adjoint else f"n1 * {nmodes} + n2"
    geom_dst, geom_src = ("source_idx", "s1") if adjoint else ("s1", "source_idx")
    imag_sign = "-" if adjoint else ""

    source = f"""
#include <cupy/complex.cuh>

__device__ {real_type} assoc_legendre_function(
    const int l, const int m, const {real_type}* ct_powers,
    const {real_type}* st_powers, const {real_type}* plm_coeffs
) {{
    {real_type} plm = 0;
    const {real_type} st_pow = st_powers[m];
    int jj = 0;
    for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
        const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
        plm += st_pow * ct_powers[lambda] * plm_coeffs[idx];
        ++jj;
    }}
    return plm;
}}

extern "C" __global__ void {name}(
    const int ns, const int nrhs,
    const double* __restrict__ positions,
    const {real_type}* __restrict__ re_h,
    const {real_type}* __restrict__ im_h,
    const {real_type} inv_dr, const int last_index,
    const {real_type}* __restrict__ plm_coeffs,
    const {real_type}* __restrict__ re_ab,
    const {real_type}* __restrict__ im_ab,
    const int* __restrict__ mode_m,
    const int* __restrict__ pair_offset,
    const int* __restrict__ pair_pmin,
    const int* __restrict__ pair_pcount,
    const {complex_type}* __restrict__ x,
    {complex_type}* __restrict__ wx
) {{
    const int n1 = blockIdx.y;
    if (n1 >= {scalar_modes}) return;

    __shared__ {real_type} x_re_shared[{nmodes * rhs_tile}];
    __shared__ {real_type} x_im_shared[{nmodes * rhs_tile}];
    const int m1 = mode_m[n1];

    for (int rhs_base = blockIdx.z * {rhs_tile}; rhs_base < nrhs;
         rhs_base += gridDim.z * {rhs_tile}) {{
        // The destination tile is block-uniform: padding lanes still stage
        // every source and participate in both barriers. Keep one owned
        // output per lane across the complete ascending source traversal.
        for (long long destination_base = (long long)blockIdx.x * blockDim.x;
             destination_base < ns;
             destination_base += (long long)blockDim.x * gridDim.x) {{
            const long long s1 = destination_base + threadIdx.x;
            {real_type} total_m_re[{rhs_tile}] = {{}};
            {real_type} total_m_im[{rhs_tile}] = {{}};
            {real_type} total_n_re[{rhs_tile}] = {{}};
            {real_type} total_n_im[{rhs_tile}] = {{}};
            for (int source_idx = 0; source_idx < ns; ++source_idx) {{
                for (int entry = threadIdx.x; entry < {nmodes * rhs_tile};
                     entry += blockDim.x) {{
                    const int n2 = entry / {rhs_tile};
                    const int rhs = rhs_base + entry % {rhs_tile};
                    const long long x_idx =
                        (((long long)source_idx * {nmodes} + n2) * nrhs) + rhs;
                    const {complex_type} value =
                        rhs < nrhs ? x[x_idx] : {complex_type}(0, 0);
                    x_re_shared[entry] = value.real();
                    x_im_shared[entry] = value.imag();
                }}
                __syncthreads();

                if (s1 < ns && s1 != source_idx) {{
                    const {real_type} x21 = ({real_type})
                        (positions[3 * {geom_dst}] - positions[3 * {geom_src}]);
                    const {real_type} y21 = ({real_type})
                        (positions[3 * {geom_dst} + 1] - positions[3 * {geom_src} + 1]);
                    const {real_type} z21 = ({real_type})
                        (positions[3 * {geom_dst} + 2] - positions[3 * {geom_src} + 2]);
                    const {real_type} r = {math["sqrt"]}(
                        x21 * x21 + y21 * y21 + z21 * z21);
                    const {real_type} ct = z21 / r;
                    const {real_type} st = {math["sqrt"]}({math["max"]}(
                        ({real_type})0, ({real_type})1 - ct * ct));
                    const {real_type} phi = {math["atan2"]}(y21, x21);

                    {real_type} ct_powers[{n_orders}];
                    {real_type} st_powers[{n_orders}];
                    {real_type} re_h_local[{n_orders}];
                    {real_type} im_h_local[{n_orders}];
                    {real_type} p_pdm[{n_p_pdm}];
                    {real_type} cos_mphi[{n_phase}];
                    {real_type} sin_mphi[{n_phase}];
                    ct_powers[0] = ({real_type})1;
                    st_powers[0] = ({real_type})1;
                    for (int p = 1; p < {n_orders}; ++p) {{
                        ct_powers[p] = ct_powers[p - 1] * ct;
                        st_powers[p] = st_powers[p - 1] * st;
                    }}
                    int i0 = 0;
                    {real_type} frac = ({real_type})0;
                    if (r > ({real_type})0) {{
                        const {real_type} t = r * inv_dr;
                        i0 = (int){math["floor"]}(t);
                        frac = t - ({real_type})i0;
                        if (i0 < 0) {{ i0 = 0; frac = ({real_type})0; }}
                        if (i0 >= last_index) {{
                            i0 = last_index - 1; frac = ({real_type})1;
                        }}
                    }}
                    const {real_type} one_minus_frac = ({real_type})1 - frac;
                    for (int p = 0; p < {n_orders}; ++p) {{
                        const long long lo = (long long)i0 * {n_orders} + p;
                        const long long hi = lo + {n_orders};
                        re_h_local[p] = r <= ({real_type})0 ? re_h[p] :
                            one_minus_frac * re_h[lo] + frac * re_h[hi];
                        im_h_local[p] = r <= ({real_type})0 ? im_h[p] :
                            one_minus_frac * im_h[lo] + frac * im_h[hi];
                        for (int abs_m = 0; abs_m <= p; ++abs_m) {{
                            p_pdm[p * (p + 1) / 2 + abs_m] =
                                assoc_legendre_function(
                                    p, abs_m, ct_powers, st_powers, plm_coeffs);
                        }}
                    }}
                    for (int idx = 0; idx < {n_phase}; ++idx) {{
                        const int dm = idx - 2 * {lmax_i};
                        {math["sincos"]}(({real_type})dm * phi,
                            &sin_mphi[idx], &cos_mphi[idx]);
                    }}

                    {real_type} re_m[{rhs_tile}] = {{}};
                    {real_type} im_m[{rhs_tile}] = {{}};
                    {real_type} re_n[{rhs_tile}] = {{}};
                    {real_type} im_n[{rhs_tile}] = {{}};
                    for (int n2 = 0; n2 < {scalar_modes}; ++n2) {{
                        const int delta_m = {delta};
                        const int abs_dm = delta_m < 0 ? -delta_m : delta_m;
                        const int phase_idx = delta_m + 2 * {lmax_i};
                        const int pair_a = {pair};
                        const int pair_b = pair_a + {scalar_modes};
                        {real_type} a_re = 0, a_im = 0;
                        {real_type} b_re = 0, b_im = 0;
                        const int base_a = pair_offset[pair_a];
                        const int start_a = pair_pmin[pair_a];
                        for (int ip = 0; ip < pair_pcount[pair_a]; ++ip) {{
                            const int p = start_a + 2 * ip;
                            const {real_type} v = re_ab[base_a + ip] *
                                p_pdm[p * (p + 1) / 2 + abs_dm];
                            a_re += v * re_h_local[p];
                            a_im += v * im_h_local[p];
                        }}
                        const int base_b = pair_offset[pair_b];
                        const int start_b = pair_pmin[pair_b];
                        for (int ip = 0; ip < pair_pcount[pair_b]; ++ip) {{
                            const int p = start_b + 2 * ip;
                            const {real_type} v = im_ab[base_b + ip] *
                                p_pdm[p * (p + 1) / 2 + abs_dm];
                            b_re += v * re_h_local[p];
                            b_im += v * im_h_local[p];
                        }}
                        const {real_type} c = cos_mphi[phase_idx];
                        const {real_type} s = sin_mphi[phase_idx];
                        const {real_type} ar = a_re * c - a_im * s;
                        const {real_type} ai = {imag_sign}(a_re * s + a_im * c);
                        const {real_type} br = -b_im * c - b_re * s;
                        const {real_type} bi = {imag_sign}(-b_im * s + b_re * c);
                        #pragma unroll
                        for (int rr = 0; rr < {rhs_tile}; ++rr) {{
                            const int xm = n2 * {rhs_tile} + rr;
                            const int xn = (n2 + {scalar_modes}) * {rhs_tile} + rr;
                            const {real_type} xm_re = x_re_shared[xm];
                            const {real_type} xm_im = x_im_shared[xm];
                            const {real_type} xn_re = x_re_shared[xn];
                            const {real_type} xn_im = x_im_shared[xn];
                            re_m[rr] += ar * xm_re - ai * xm_im;
                            im_m[rr] += ar * xm_im + ai * xm_re;
                            re_m[rr] += br * xn_re - bi * xn_im;
                            im_m[rr] += br * xn_im + bi * xn_re;
                            re_n[rr] += br * xm_re - bi * xm_im;
                            im_n[rr] += br * xm_im + bi * xm_re;
                            re_n[rr] += ar * xn_re - ai * xn_im;
                            im_n[rr] += ar * xn_im + ai * xn_re;
                        }}
                    }}
                    #pragma unroll
                    for (int rr = 0; rr < {rhs_tile}; ++rr) {{
                        // Preserve the per-pair subtotal and source order.
                        // Do not fold each mode/p term into the source total.
                        total_m_re[rr] += re_m[rr];
                        total_m_im[rr] += im_m[rr];
                        total_n_re[rr] += re_n[rr];
                        total_n_im[rr] += im_n[rr];
                    }}
                }}
                __syncthreads();
            }}
            if (s1 < ns) {{
                #pragma unroll
                for (int rr = 0; rr < {rhs_tile}; ++rr) {{
                    const int rhs = rhs_base + rr;
                    if (rhs < nrhs) {{
                        const long long ym =
                            (s1 * {nmodes} + n1) * nrhs + rhs;
                        const long long yn = ym + (long long){scalar_modes} * nrhs;
                        wx[ym] = {complex_type}(total_m_re[rr], total_m_im[rr]);
                        wx[yn] = {complex_type}(total_n_re[rr], total_n_im[rr]);
                    }}
                }}
            }}
        }}
    }}
}}
"""
    return cupy.RawKernel(source, name)


@dataclass
class CuPyPairwiseCouplingOperator:
    """Direct GPU pairwise coupling with exact forward and adjoint actions."""

    lmax: int
    k: float
    positions: np.ndarray
    ab5: np.ndarray
    radial_lut: RadialLUT
    dtype: np.dtype = COMPLEX128_DTYPE
    _positions_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_re_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_im_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _plm_coeff_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_re_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_im_gpu: Any | None = field(default=None, init=False, repr=False)
    _mode_m_gpu: Any | None = field(default=None, init=False, repr=False)
    _pair_offset_gpu: Any | None = field(default=None, init=False, repr=False)
    _pair_pmin_gpu: Any | None = field(default=None, init=False, repr=False)
    _pair_pcount_gpu: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise TypeError(
                "CuPy pairwise coupling supports only complex64 and complex128. "
                f"Got {self.dtype!r}."
            )

    @property
    def n_modes(self) -> int:
        return n_modes(self.lmax)

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def real_dtype(self) -> np.dtype:
        return np.dtype(np.float32 if self.dtype == np.dtype(np.complex64) else np.float64)

    def _raw_kernel_resources(self, *, rhs_tile_size: int = 1):
        cupy, _ = import_cupy()
        real_dtype = self.real_dtype
        if self._positions_gpu is None:
            self._positions_gpu = cupy.asarray(
                np.ascontiguousarray(self.positions, dtype=np.float64).reshape(-1)
            )
        if self._compact_re_ab_gpu is None or self._compact_im_ab_gpu is None:
            re_ab, im_ab = _translation_ab5_compact_tables(self.lmax, dtype=self.dtype)
            self._compact_re_ab_gpu = cupy.asarray(re_ab, dtype=real_dtype)
            self._compact_im_ab_gpu = cupy.asarray(im_ab, dtype=real_dtype)
        if self._plm_coeff_gpu is None:
            coeff = _translation_plm_coeff_table(self.lmax, dtype=real_dtype).reshape(-1)
            self._plm_coeff_gpu = cupy.asarray(coeff, dtype=real_dtype)
        if self._lut_re_gpu is None or self._lut_im_gpu is None:
            lut = np.asarray(self.radial_lut.h, dtype=self.dtype).T.copy()
            self._lut_re_gpu = cupy.asarray(
                np.ascontiguousarray(lut.real.reshape(-1)), dtype=real_dtype
            )
            self._lut_im_gpu = cupy.asarray(
                np.ascontiguousarray(lut.imag.reshape(-1)), dtype=real_dtype
            )
        if self._mode_m_gpu is None:
            _, _, mode_m = mode_metadata_tables(self.lmax)
            self._mode_m_gpu = cupy.asarray(mode_m)
        if (
            self._pair_offset_gpu is None
            or self._pair_pmin_gpu is None
            or self._pair_pcount_gpu is None
        ):
            pair_offset, pair_pmin, pair_pcount = mode_pair_p_range_tables(self.lmax)
            self._pair_offset_gpu = cupy.asarray(pair_offset.reshape(-1))
            self._pair_pmin_gpu = cupy.asarray(pair_pmin.reshape(-1))
            self._pair_pcount_gpu = cupy.asarray(pair_pcount.reshape(-1))
        return (
            _source_parallel_kernel(self.lmax, self.dtype.str, False, rhs_tile_size),
            self._positions_gpu,
            self._lut_re_gpu,
            self._lut_im_gpu,
            self._plm_coeff_gpu,
            self._compact_re_ab_gpu,
            self._compact_im_ab_gpu,
            self._mode_m_gpu,
            self._pair_offset_gpu,
            self._pair_pmin_gpu,
            self._pair_pcount_gpu,
        )

    def _array(self, x: np.ndarray | object):
        cupy, _ = import_cupy()
        raw = coerce_array(x, dtype=self.dtype, prefer_cupy=True)
        expected = self.n_particles * self.n_modes
        if int(raw.ndim) == 1:
            if int(raw.size) != expected:
                raise ValueError(f"Input length must be {expected}; got {int(raw.size)}.")
            return cupy.ascontiguousarray(raw.reshape(self.n_particles, self.n_modes, 1)), True
        if int(raw.ndim) == 2:
            if int(raw.shape[0]) != expected:
                raise ValueError(
                    f"Input first dimension must be {expected}; got {int(raw.shape[0])}."
                )
            return cupy.ascontiguousarray(
                raw.reshape(self.n_particles, self.n_modes, int(raw.shape[1]))
            ), False
        raise ValueError(f"Input must be 1D or 2D. Got shape {tuple(raw.shape)}.")

    def _launch_config(self, nrhs: int) -> tuple[int, int, int, int]:
        cupy, _ = import_cupy()
        props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
        max_threads = int(props["maxThreadsPerBlock"])
        target = 64 if self.dtype == np.dtype(np.complex64) else 32
        threads = min(max_threads, target)
        blocks_x = min(
            int(props["maxGridSize"][0]),
            (self.n_particles + threads - 1) // threads,
        )
        grid_y = self.n_modes // 2
        rhs_tile = 1 if nrhs == 1 else min(4, nrhs)
        grid_z = min(
            int(props["maxGridSize"][2]),
            (nrhs + rhs_tile - 1) // rhs_tile,
        )
        return int(blocks_x), int(threads), int(grid_y), int(grid_z)

    def _apply_gpu(self, x: np.ndarray | object, *, adjoint: bool = False):
        cupy, _ = import_cupy()
        arr, squeezed = self._array(x)
        nrhs = int(arr.shape[2])
        rhs_tile = 1 if nrhs == 1 else min(4, nrhs)
        # Each destination is written once when at least one non-self source
        # exists.  A one-particle system has no launched interaction, so keep
        # the mathematically required zero output as well.
        output = cupy.zeros((self.n_particles, self.n_modes, nrhs), dtype=self.dtype)
        (
            _forward_kernel,
            positions,
            lut_re,
            lut_im,
            plm_coeff,
            re_ab,
            im_ab,
            mode_m,
            pair_offset,
            pair_pmin,
            pair_pcount,
        ) = self._raw_kernel_resources(rhs_tile_size=rhs_tile)
        if self.n_particles and nrhs:
            kernel = _source_parallel_kernel(self.lmax, self.dtype.str, adjoint, rhs_tile)
            blocks_x, threads, grid_y, grid_z = self._launch_config(nrhs)
            inv_dr = self.real_dtype.type(self.radial_lut._inv_dr)
            kernel(
                (blocks_x, grid_y, grid_z),
                (threads,),
                (
                    np.int32(self.n_particles),
                    np.int32(nrhs),
                    positions,
                    lut_re,
                    lut_im,
                    inv_dr,
                    np.int32(self.radial_lut._last_index),
                    plm_coeff,
                    re_ab,
                    im_ab,
                    mode_m,
                    pair_offset,
                    pair_pmin,
                    pair_pcount,
                    arr.reshape(-1),
                    output.reshape(-1),
                ),
            )
        if squeezed:
            return output.reshape(self.n_particles * self.n_modes)
        return output.reshape(self.n_particles * self.n_modes, nrhs)

    def apply(self, x: np.ndarray | object) -> np.ndarray | object:
        out = self._apply_gpu(x)
        return out if is_cupy_array(x) else asnumpy(out)

    def apply_adjoint(self, x: np.ndarray | object) -> np.ndarray | object:
        out = self._apply_gpu(x, adjoint=True)
        return out if is_cupy_array(x) else asnumpy(out)


__all__ = ["CuPyPairwiseCouplingOperator", "_source_parallel_kernel"]
