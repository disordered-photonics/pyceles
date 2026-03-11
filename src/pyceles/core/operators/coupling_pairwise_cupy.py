from __future__ import annotations

"""Fused CuPy pairwise coupling for direct GPU matvecs.

We explicitly ship the CuPy direct backend as a fused RawKernel rather than as
a composition of higher-level CuPy operations. That higher-level decomposition
was explored separately during development, but for the direct `O(N^2)` pairwise
matvec it produced too many small GPU operations, too many temporary arrays,
too much Python-side orchestration, and too much kernel-launch overhead to stay
competitive.

The shipped CuPy path therefore keeps the translation tables on device and
accumulates the pairwise coupling inside the fused kernel.
"""

from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import asnumpy, coerce_array, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes
from pyceles.core.translation import (
    RadialLUT,
    _translation_ab5_compact_tables,
    _translation_plm_coeff_table,
)


@cache
def _translation_matvec_raw_kernel(lmax: int, dtype_str: str):
    cupy, _ = import_cupy()
    lmax = int(lmax)
    dtype = np.dtype(dtype_str)
    if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"Unsupported CuPy raw-kernel dtype {dtype!r}.")

    real_type = "float" if dtype == np.dtype(np.complex64) else "double"
    math = {
        "atan2": "atan2f" if real_type == "float" else "atan2",
        "cos": "cosf" if real_type == "float" else "cos",
        "floor": "floorf" if real_type == "float" else "floor",
        "max": "fmaxf" if real_type == "float" else "fmax",
        "pow": "powf" if real_type == "float" else "pow",
        "sin": "sinf" if real_type == "float" else "sin",
        "sqrt": "sqrtf" if real_type == "float" else "sqrt",
    }
    kernel_name = (
        "translation_matrix_product_c64"
        if real_type == "float"
        else "translation_matrix_product_c128"
    )

    n_scalar = lmax * (lmax + 2)
    n_modes_total = 2 * n_scalar
    n_orders = 2 * lmax + 1
    n_p_pdm = n_orders * (n_orders + 1) // 2
    n_phase = 2 * n_orders - 1
    source = f"""
    extern "C" __device__ {real_type} assoc_legendre_function(
        const int l,
        const int m,
        const {real_type} ct,
        const {real_type} st,
        const {real_type}* plm_coeffs
    ) {{
        {real_type} plm = 0;
        {real_type} st_pow = (m == 0) ? ({real_type})1 : {math["pow"]}(st, ({real_type})m);
        int jj = 0;
        for (int lambda = l - m; lambda >= 0; lambda -= 2) {{
            const int idx = jj * ({n_orders} * {n_orders}) + m * {n_orders} + l;
            plm += st_pow * {math["pow"]}(ct, ({real_type})lambda) * plm_coeffs[idx];
            jj += 1;
        }}
        return plm;
    }}

    extern "C" __device__ {real_type} hankel_lookup_linear(
        const int p,
        const {real_type} r,
        const {real_type}* table,
        const {real_type} inv_dr,
        const int last_index
    ) {{
        if (r <= ({real_type})0) {{
            return table[p];
        }}
        {real_type} t = r * inv_dr;
        int i0 = (int){math["floor"]}(t);
        {real_type} frac = t - ({real_type})i0;
        if (i0 < 0) {{
            i0 = 0;
            frac = ({real_type})0;
        }}
        if (i0 >= last_index) {{
            i0 = last_index - 1;
            frac = ({real_type})1;
        }}
        const int base0 = i0 * {n_orders} + p;
        const int base1 = (i0 + 1) * {n_orders} + p;
        return (({real_type})1 - frac) * table[base0] + frac * table[base1];
    }}

    extern "C" __global__ void {kernel_name}(
        const int s2,
        const int ns,
        const {real_type}* positions,
        const {real_type}* re_h,
        const {real_type}* im_h,
        const {real_type} inv_dr,
        const int last_index,
        const {real_type}* plm_coeffs,
        const {real_type}* re_ab,
        const {real_type}* im_ab,
        const {real_type}* re_x,
        const {real_type}* im_x,
        {real_type}* re_wx,
        {real_type}* im_wx
    ) {{
        const int s1 = blockDim.x * blockIdx.x + threadIdx.x;
        if (s1 >= ns || s1 == s2) {{
            return;
        }}

        const {real_type} x21 = positions[3 * s1] - positions[3 * s2];
        const {real_type} y21 = positions[3 * s1 + 1] - positions[3 * s2 + 1];
        const {real_type} z21 = positions[3 * s1 + 2] - positions[3 * s2 + 2];
        const {real_type} r = {math["sqrt"]}(x21 * x21 + y21 * y21 + z21 * z21);
        const {real_type} ct = z21 / r;
        const {real_type} st = {math["sqrt"]}({math["max"]}(({real_type})0, ({real_type})1 - ct * ct));
        const {real_type} phi = {math["atan2"]}(y21, x21);

        {real_type} re_h_local[{n_orders}];
        {real_type} im_h_local[{n_orders}];
        {real_type} p_pdm[{n_p_pdm}];
        {real_type} cos_mphi[{n_phase}];
        {real_type} sin_mphi[{n_phase}];

        for (int p = 0; p < {n_orders}; ++p) {{
            re_h_local[p] = hankel_lookup_linear(p, r, re_h, inv_dr, last_index);
            im_h_local[p] = hankel_lookup_linear(p, r, im_h, inv_dr, last_index);
            for (int absdm = 0; absdm <= p; ++absdm) {{
                p_pdm[p * (p + 1) / 2 + absdm] =
                    assoc_legendre_function(p, absdm, ct, st, plm_coeffs);
            }}
        }}

        for (int dm = -2 * {lmax}; dm <= 2 * {lmax}; ++dm) {{
            const int idx = dm + 2 * {lmax};
            cos_mphi[idx] = {math["cos"]}(({real_type})dm * phi);
            sin_mphi[idx] = {math["sin"]}(({real_type})dm * phi);
        }}

        int loop_counter = 0;
        for (int tau1 = 1; tau1 <= 2; ++tau1) {{
            const int temp1 = (tau1 - 1) * {n_scalar};
            for (int l1 = 1; l1 <= {lmax}; ++l1) {{
                const int coeff1 = temp1 + (l1 - 1) * (l1 + 1) + l1;
                for (int m1 = -l1; m1 <= l1; ++m1) {{
                    const int n1 = coeff1 + m1;
                    {real_type} re_incr = 0;
                    {real_type} im_incr = 0;

                    for (int tau2 = 1; tau2 <= 2; ++tau2) {{
                        const int temp2 = (tau2 - 1) * {n_scalar};
                        for (int l2 = 1; l2 <= {lmax}; ++l2) {{
                            const int coeff2 = temp2 + (l2 - 1) * (l2 + 1) + l2;
                            for (int m2 = -l2; m2 <= l2; ++m2) {{
                                const int n2 = coeff2 + m2;
                                const {real_type} re_x_tmp = re_x[s2 * {n_modes_total} + n2];
                                const {real_type} im_x_tmp = im_x[s2 * {n_modes_total} + n2];
                                const int delta_m = m2 - m1;
                                const int phase_idx = delta_m + 2 * {lmax};
                                const int p_min = max(abs(delta_m), abs(l1 - l2) + abs(tau1 - tau2));
                                for (int p = p_min; p <= l1 + l2; ++p) {{
                                    const {real_type} plm = p_pdm[p * (p + 1) / 2 + abs(delta_m)];
                                    const {real_type} re_abp = re_ab[loop_counter] * plm;
                                    const {real_type} im_abp = im_ab[loop_counter] * plm;
                                    const {real_type} re_abph =
                                        re_abp * re_h_local[p] - im_abp * im_h_local[p];
                                    const {real_type} im_abph =
                                        re_abp * im_h_local[p] + im_abp * re_h_local[p];
                                    const {real_type} re_phase =
                                        re_abph * cos_mphi[phase_idx] - im_abph * sin_mphi[phase_idx];
                                    const {real_type} im_phase =
                                        re_abph * sin_mphi[phase_idx] + im_abph * cos_mphi[phase_idx];
                                    re_incr += re_phase * re_x_tmp - im_phase * im_x_tmp;
                                    im_incr += re_phase * im_x_tmp + im_phase * re_x_tmp;
                                    loop_counter += 1;
                                }}
                            }}
                        }}
                    }}

                    re_wx[s1 * {n_modes_total} + n1] += re_incr;
                    im_wx[s1 * {n_modes_total} + n1] += im_incr;
                }}
            }}
        }}
    }}
    """
    return cupy.RawKernel(source, kernel_name)


@dataclass
class CuPyPairwiseCouplingOperator:
    """Direct GPU pairwise coupling using dtype-specific fused RawKernels.

    The free-space coupling `W` is geometry-only and does not depend on the
    particle single-body representation. The CuPy preparation path is limited
    by the available GPU `T` operator and therefore exposes only diagonal
    particle groups (spheres and layered spheres).
    """

    lmax: int
    k: float
    positions: np.ndarray
    ab5: np.ndarray
    radial_lut: RadialLUT
    dtype: np.dtype = np.dtype(np.complex128)
    _positions_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_re_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _compact_im_ab_gpu: Any | None = field(default=None, init=False, repr=False)
    _plm_coeff_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_re_gpu: Any | None = field(default=None, init=False, repr=False)
    _lut_im_gpu: Any | None = field(default=None, init=False, repr=False)

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

    def _raw_kernel_resources(self):
        cupy, _ = import_cupy()
        real_dtype = self.real_dtype
        if self._positions_gpu is None:
            self._positions_gpu = cupy.asarray(self.positions, dtype=real_dtype).reshape(-1)
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
                np.ascontiguousarray(lut.real.reshape(-1)),
                dtype=real_dtype,
            )
            self._lut_im_gpu = cupy.asarray(
                np.ascontiguousarray(lut.imag.reshape(-1)),
                dtype=real_dtype,
            )
        return (
            _translation_matvec_raw_kernel(self.lmax, self.dtype.str),
            self._positions_gpu,
            self._lut_re_gpu,
            self._lut_im_gpu,
            self._plm_coeff_gpu,
            self._compact_re_ab_gpu,
            self._compact_im_ab_gpu,
        )

    def _apply_gpu(self, x: np.ndarray | object):
        cupy, _ = import_cupy()
        real_dtype = cupy.float32 if self.dtype == np.dtype(np.complex64) else cupy.float64
        (
            kernel,
            positions_gpu,
            lut_re_gpu,
            lut_im_gpu,
            plm_coeff_gpu,
            compact_re_ab_gpu,
            compact_im_ab_gpu,
        ) = self._raw_kernel_resources()

        arr = coerce_array(x, dtype=self.dtype, prefer_cupy=True).reshape(
            self.n_particles, self.n_modes
        )
        x_re = cupy.ascontiguousarray(arr.real.reshape(-1).astype(real_dtype, copy=False))
        x_im = cupy.ascontiguousarray(arr.imag.reshape(-1).astype(real_dtype, copy=False))
        y_re = cupy.zeros((self.n_particles * self.n_modes,), dtype=real_dtype)
        y_im = cupy.zeros((self.n_particles * self.n_modes,), dtype=real_dtype)

        threads_per_block = 128
        blocks_per_grid = (self.n_particles + threads_per_block - 1) // threads_per_block
        inv_dr = self.real_dtype.type(self.radial_lut._inv_dr)
        for s2 in range(self.n_particles):
            kernel(
                (blocks_per_grid,),
                (threads_per_block,),
                (
                    np.int32(s2),
                    np.int32(self.n_particles),
                    positions_gpu,
                    lut_re_gpu,
                    lut_im_gpu,
                    inv_dr,
                    np.int32(self.radial_lut._last_index),
                    plm_coeff_gpu,
                    compact_re_ab_gpu,
                    compact_im_ab_gpu,
                    x_re,
                    x_im,
                    y_re,
                    y_im,
                ),
            )
        out = (y_re + 1j * y_im).astype(self.dtype, copy=False)
        return out.reshape(self.n_particles * self.n_modes)

    def apply(self, x: np.ndarray | object) -> np.ndarray | object:
        out = self._apply_gpu(x)
        return out if is_cupy_array(x) else asnumpy(out)


__all__ = ["CuPyPairwiseCouplingOperator"]
