"""Fused CuPy kernels for finite-cluster scattered near fields.

The public near-field API is intentionally backend-neutral.  This private owner
keeps the generated GPU implementation out of the NumPy reference module.
One CUDA block owns one field point, traverses source particles cooperatively,
and reduces the resulting Cartesian field inside the block.  That removes the
large ``(sphere, point, mode, xyz)`` temporaries and the many small CuPy
operations used by the previous general vectorized implementation.

The compiled specialization depends on ``lmax``, compute/accumulation dtypes,
and whether magnetic fields are requested.  Frequency, geometry, coefficients,
and radial-LUT values are launch data, so repeated broadband frequencies reuse
the in-process ``functools.cache`` entry rather than recompiling a kernel.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.indexing import index_vswf, n_modes
from pyceles.core.spherical import _legendre_scalar_tables

_BLOCK_SIZE = 128


def _real_literal(value: float) -> str:
    return f"({float(value):.17g})"


def _angular_recurrence_source(lmax: int) -> str:
    """Emit the CELES-normalized angular recurrence with scalar constants inlined."""
    lmax = int(lmax)
    stride = lmax + 1
    a0, b0, c_mm, a_lm, b_lm = _legendre_scalar_tables(lmax)
    lines = [
        f"REAL plm[{stride * stride}] = {{(REAL)0}};",
        f"REAL pi_lm[{stride * stride}] = {{(REAL)0}};",
        f"REAL tau_lm[{stride * stride}] = {{(REAL)0}};",
        f"REAL pprime0[{stride}] = {{(REAL)0}};",
        f"plm[0] = (REAL){_real_literal(np.sqrt(2.0) / 2.0)};",
    ]
    if lmax >= 1:
        lines.extend(
            [
                f"plm[{stride}] = (REAL){_real_literal(np.sqrt(3.0 / 2.0))} * ct;",
                f"pprime0[1] = (REAL){_real_literal(np.sqrt(3.0))} * plm[0];",
                f"tau_lm[{stride}] = -st * pprime0[1];",
            ]
        )
    for l in range(1, lmax):
        lp1 = l + 1
        idx = lp1 * stride
        prev = l * stride
        prev2 = (l - 1) * stride
        coeff = np.sqrt((2 * lp1 + 1.0) / (2 * lp1 - 1.0))
        lines.extend(
            [
                f"plm[{idx}] = (REAL){_real_literal(a0[l])} * ct * plm[{prev}] "
                f"- (REAL){_real_literal(b0[l])} * plm[{prev2}];",
                f"pprime0[{lp1}] = (REAL){_real_literal(lp1 * coeff)} * plm[{prev}] "
                f"+ (REAL){_real_literal(coeff)} * ct * pprime0[{l}];",
                f"tau_lm[{idx}] = -st * pprime0[{lp1}];",
            ]
        )
    lines.append("REAL st_pow_prev = (REAL)1;")
    lines.append("REAL st_pow = st;")
    for m in range(1, lmax + 1):
        idx = m * stride + m
        lines.extend(
            [
                f"plm[{idx}] = (REAL){_real_literal(c_mm[m])} * st_pow;",
                f"pi_lm[{idx}] = (REAL){_real_literal(c_mm[m])} * st_pow_prev;",
                f"tau_lm[{idx}] = (REAL){m} * ct * pi_lm[{idx}];",
            ]
        )
        for l in range(m, lmax):
            lp1 = l + 1
            idx_next = lp1 * stride + m
            idx_cur = l * stride + m
            idx_prev = (l - 1) * stride + m
            tau_coeff = (lp1 + m) * np.sqrt(
                (2 * lp1 + 1.0) * (lp1 - m) / ((2 * lp1 - 1.0) * (lp1 + m))
            )
            lines.extend(
                [
                    f"plm[{idx_next}] = (REAL){_real_literal(a_lm[l, m])} * ct * "
                    f"plm[{idx_cur}] - (REAL){_real_literal(b_lm[l, m])} * plm[{idx_prev}];",
                    f"pi_lm[{idx_next}] = (REAL){_real_literal(a_lm[l, m])} * ct * "
                    f"pi_lm[{idx_cur}] - (REAL){_real_literal(b_lm[l, m])} * pi_lm[{idx_prev}];",
                    f"tau_lm[{idx_next}] = (REAL){lp1} * ct * pi_lm[{idx_next}] "
                    f"- (REAL){_real_literal(tau_coeff)} * pi_lm[{idx_cur}];",
                ]
            )
        lines.append("st_pow_prev = st_pow;")
        lines.append("st_pow *= st;")
    return "\n                ".join(lines)


def _mode_accumulation_source(lmax: int, *, compute_magnetic: bool) -> str:
    stride = int(lmax) + 1
    lines: list[str] = []
    for l in range(1, int(lmax) + 1):
        pref = 1.0 / np.sqrt(2.0 * l * (l + 1.0))
        lines.extend(
            [
                "{",
                f"const int lut_base = {l} * n_r;",
                "const int lut_i0 = radial_i0;",
                "const ACC_COMPLEX h_l(",
                "    (ACC_REAL)(((REAL)1 - radial_frac) * h_re[lut_base + lut_i0] "
                "+ radial_frac * h_re[lut_base + lut_i0 + 1]),",
                "    (ACC_REAL)(((REAL)1 - radial_frac) * h_im[lut_base + lut_i0] "
                "+ radial_frac * h_im[lut_base + lut_i0 + 1])",
                ");",
                "const ACC_COMPLEX d_l(",
                "    (ACC_REAL)(((REAL)1 - radial_frac) * d_re[lut_base + lut_i0] "
                "+ radial_frac * d_re[lut_base + lut_i0 + 1]),",
                "    (ACC_REAL)(((REAL)1 - radial_frac) * d_im[lut_base + lut_i0] "
                "+ radial_frac * d_im[lut_base + lut_i0 + 1])",
                ");",
                "const ACC_COMPLEX h_over_kr = h_l / (ACC_REAL)kr;",
                "const ACC_COMPLEX d_over_kr = d_l / (ACC_REAL)kr;",
            ]
        )
        for m in range(-l, l + 1):
            abs_m = abs(m)
            idx_ang = l * stride + abs_m
            idx_a = index_vswf(l, m, 1, int(lmax))
            idx_b = index_vswf(l, m, 2, int(lmax))
            tag = f"l{l}_{'n' + str(abs(m)) if m < 0 else 'p' + str(m)}"
            phase = (
                f"phase_pos[{m}]"
                if m >= 0
                else f"ACC_COMPLEX(phase_pos[{abs_m}].real(), -phase_pos[{abs_m}].imag())"
            )
            lines.extend(
                [
                    "{",
                    f"const ACC_REAL p = (ACC_REAL)plm[{idx_ang}];",
                    f"const ACC_REAL pi_v = (ACC_REAL)pi_lm[{idx_ang}];",
                    f"const ACC_REAL tau_v = (ACC_REAL)tau_lm[{idx_ang}];",
                    f"const ACC_COMPLEX phase = {phase};",
                    f"const ACC_COMPLEX impi((ACC_REAL)0, (ACC_REAL){m} * pi_v);",
                    f"const COMPUTE_COMPLEX a_in = coeffs[{idx_a} * n_sources + source_idx];",
                    f"const COMPUTE_COMPLEX b_in = coeffs[{idx_b} * n_sources + source_idx];",
                    "const ACC_COMPLEX a((ACC_REAL)a_in.real(), (ACC_REAL)a_in.imag());",
                    "const ACC_COMPLEX b((ACC_REAL)b_in.real(), (ACC_REAL)b_in.imag());",
                    f"const ACC_REAL pref = (ACC_REAL){_real_literal(pref)};",
                    "const ACC_COMPLEX m_theta = pref * h_l * impi * phase;",
                    "const ACC_COMPLEX m_phi = -pref * h_l * tau_v * phase;",
                    (
                        f"const ACC_COMPLEX n_r = pref * (ACC_REAL){l * (l + 1)} "
                        "* h_over_kr * p * phase;"
                    ),
                    "const ACC_COMPLEX n_theta = pref * d_over_kr * tau_v * phase;",
                    "const ACC_COMPLEX n_phi = pref * d_over_kr * impi * phase;",
                    "const ACC_COMPLEX mx = m_theta * (ACC_REAL)etx + m_phi * (ACC_REAL)epx;",
                    "const ACC_COMPLEX my = m_theta * (ACC_REAL)ety + m_phi * (ACC_REAL)epy;",
                    "const ACC_COMPLEX mz = m_theta * (ACC_REAL)etz;",
                    (
                        "const ACC_COMPLEX nx = n_r * (ACC_REAL)erx "
                        "+ n_theta * (ACC_REAL)etx + n_phi * (ACC_REAL)epx;"
                    ),
                    (
                        "const ACC_COMPLEX ny = n_r * (ACC_REAL)ery "
                        "+ n_theta * (ACC_REAL)ety + n_phi * (ACC_REAL)epy;"
                    ),
                    "const ACC_COMPLEX nz = n_r * (ACC_REAL)erz + n_theta * (ACC_REAL)etz;",
                    "ex += a * mx + b * nx;",
                    "ey += a * my + b * ny;",
                    "ez += a * mz + b * nz;",
                ]
            )
            if compute_magnetic:
                lines.extend(
                    [
                        "hx += medium_factor * (a * nx + b * mx);",
                        "hy += medium_factor * (a * ny + b * my);",
                        "hz += medium_factor * (a * nz + b * mz);",
                    ]
                )
            lines.extend([f"// end mode {tag}", "}"])
        lines.append("}")
    return "\n                ".join(lines)


@cache
def _scattered_field_raw_kernel(
    lmax: int,
    compute_dtype_name: str,
    accum_dtype_name: str,
    compute_magnetic: bool,
):
    cupy, _ = import_cupy()
    compute_dtype = np.dtype(compute_dtype_name)
    accum_dtype = np.dtype(accum_dtype_name)
    if compute_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"Unsupported compute dtype {compute_dtype!r}.")
    if accum_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"Unsupported accumulation dtype {accum_dtype!r}.")

    real_type = "float" if compute_dtype == np.dtype(np.complex64) else "double"
    compute_complex = f"complex<{real_type}>"
    accum_real = "float" if accum_dtype == np.dtype(np.complex64) else "double"
    accum_complex = f"complex<{accum_real}>"
    math = {
        "atan2": "atan2f" if real_type == "float" else "atan2",
        "floor": "floorf" if real_type == "float" else "floor",
        "max": "fmaxf" if real_type == "float" else "fmax",
        "min": "fminf" if real_type == "float" else "fmin",
        "sincos": "sincosf" if real_type == "float" else "sincos",
        "sqrt": "sqrtf" if real_type == "float" else "sqrt",
    }
    suffix = "eh" if compute_magnetic else "e"
    kernel_name = (
        f"nearfield_scattered_l{int(lmax)}_{compute_dtype.name}_{accum_dtype.name}_{suffix}"
    )
    magnetic_decl = (
        f"__shared__ ACC_COMPLEX shx[{_BLOCK_SIZE}], shy[{_BLOCK_SIZE}], shz[{_BLOCK_SIZE}];"
        if compute_magnetic
        else ""
    )
    magnetic_locals = (
        "ACC_COMPLEX hx((ACC_REAL)0, (ACC_REAL)0);\n"
        "        ACC_COMPLEX hy((ACC_REAL)0, (ACC_REAL)0);\n"
        "        ACC_COMPLEX hz((ACC_REAL)0, (ACC_REAL)0);\n"
        "        const ACC_COMPLEX medium_factor(medium_re, medium_im);"
        if compute_magnetic
        else ""
    )
    magnetic_store = "shx[tid] = hx; shy[tid] = hy; shz[tid] = hz;" if compute_magnetic else ""
    magnetic_reduce = (
        "shx[tid] += shx[tid + stride];\n"
        "                shy[tid] += shy[tid + stride];\n"
        "                shz[tid] += shz[tid + stride];"
        if compute_magnetic
        else ""
    )
    magnetic_output = (
        "out_h[3 * point + 0] = shx[0];\n"
        "            out_h[3 * point + 1] = shy[0];\n"
        "            out_h[3 * point + 2] = shz[0];"
        if compute_magnetic
        else ""
    )
    h_argument = ", ACC_COMPLEX* out_h" if compute_magnetic else ""

    source = f"""
    #include <cupy/complex.cuh>
    typedef {real_type} REAL;
    typedef {compute_complex} COMPUTE_COMPLEX;
    typedef {accum_real} ACC_REAL;
    typedef {accum_complex} ACC_COMPLEX;

    extern "C" __global__ void {kernel_name}(
        const int n_points,
        const int n_sources,
        const REAL k,
        const REAL inv_dr,
        const int n_r,
        const ACC_REAL medium_re,
        const ACC_REAL medium_im,
        const double* points,
        const double* positions,
        const COMPUTE_COMPLEX* coeffs,
        const REAL* h_re,
        const REAL* h_im,
        const REAL* d_re,
        const REAL* d_im,
        ACC_COMPLEX* out_e{h_argument}
    ) {{
        const int point = (int)blockIdx.x;
        const int tid = (int)threadIdx.x;
        if (point >= n_points) return;

        __shared__ ACC_COMPLEX sex[{_BLOCK_SIZE}], sey[{_BLOCK_SIZE}], sez[{_BLOCK_SIZE}];
        {magnetic_decl}
        ACC_COMPLEX ex((ACC_REAL)0, (ACC_REAL)0);
        ACC_COMPLEX ey((ACC_REAL)0, (ACC_REAL)0);
        ACC_COMPLEX ez((ACC_REAL)0, (ACC_REAL)0);
        {magnetic_locals}

        const double px = points[3 * point + 0];
        const double py = points[3 * point + 1];
        const double pz = points[3 * point + 2];

        for (int source_idx = tid; source_idx < n_sources; source_idx += {_BLOCK_SIZE}) {{
            const REAL dx = px - positions[source_idx];
            const REAL dy = py - positions[n_sources + source_idx];
            const REAL dz = pz - positions[2 * n_sources + source_idx];
            REAL r = {math["sqrt"]}(dx * dx + dy * dy + dz * dz);
            if (r <= (REAL)0) r = (REAL)1e-30;
            const REAL inv_r = (REAL)1 / r;
            const REAL erx = dx * inv_r;
            const REAL ery = dy * inv_r;
            const REAL erz = dz * inv_r;
            const REAL ct = {math["max"]}((REAL)-1, {math["min"]}((REAL)1, erz));
            const REAL rho = {math["sqrt"]}({math["max"]}((REAL)0, dx * dx + dy * dy));
            const REAL st = rho * inv_r;
            const REAL inv_rho = rho > (REAL)0 ? (REAL)1 / rho : (REAL)0;
            const REAL cos_phi = rho > (REAL)0 ? dx * inv_rho : (REAL)1;
            const REAL sin_phi = rho > (REAL)0 ? dy * inv_rho : (REAL)0;
            const REAL etx = ct * cos_phi;
            const REAL ety = ct * sin_phi;
            const REAL etz = -st;
            const REAL epx = -sin_phi;
            const REAL epy = cos_phi;

            const REAL radial_t_unclamped = r * inv_dr;
            int radial_i0 = (int){math["floor"]}(radial_t_unclamped);
            REAL radial_frac = radial_t_unclamped - (REAL)radial_i0;
            if (radial_i0 < 0) {{ radial_i0 = 0; radial_frac = (REAL)0; }}
            if (radial_i0 >= n_r - 1) {{ radial_i0 = n_r - 2; radial_frac = (REAL)1; }}
            const REAL kr = k * r;

            {_angular_recurrence_source(int(lmax))}

            ACC_COMPLEX phase_pos[{int(lmax) + 1}];
            phase_pos[0] = ACC_COMPLEX((ACC_REAL)1, (ACC_REAL)0);
            const ACC_COMPLEX phase_unit((ACC_REAL)cos_phi, (ACC_REAL)sin_phi);
            #pragma unroll
            for (int m = 1; m <= {int(lmax)}; ++m) {{
                phase_pos[m] = phase_pos[m - 1] * phase_unit;
            }}
            {_mode_accumulation_source(int(lmax), compute_magnetic=compute_magnetic)}
        }}

        sex[tid] = ex; sey[tid] = ey; sez[tid] = ez;
        {magnetic_store}
        __syncthreads();
        for (int stride = {_BLOCK_SIZE // 2}; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                sex[tid] += sex[tid + stride];
                sey[tid] += sey[tid + stride];
                sez[tid] += sez[tid + stride];
                {magnetic_reduce}
            }}
            __syncthreads();
        }}
        if (tid == 0) {{
            out_e[3 * point + 0] = sex[0];
            out_e[3 * point + 1] = sey[0];
            out_e[3 * point + 2] = sez[0];
            {magnetic_output}
        }}
    }}
    """
    return cupy.RawKernel(source, kernel_name)


def _packed_lut_arrays(lut: Any, *, real_dtype: np.dtype) -> tuple[np.ndarray, ...]:
    h = np.ascontiguousarray(np.stack(lut.h, axis=0), dtype=lut.dtype)
    d = np.ascontiguousarray(np.stack(lut.dxxz, axis=0), dtype=lut.dtype)
    return (
        np.ascontiguousarray(h.real, dtype=real_dtype).reshape(-1),
        np.ascontiguousarray(h.imag, dtype=real_dtype).reshape(-1),
        np.ascontiguousarray(d.real, dtype=real_dtype).reshape(-1),
        np.ascontiguousarray(d.imag, dtype=real_dtype).reshape(-1),
    )


def compute_scattered_field_cupy_fused(
    *,
    field_points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    n_medium: complex,
    lut: Any,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
    compute_magnetic: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate finite-cluster scattered fields with one block per output point."""
    if int(lmax) < 1:
        raise ValueError(f"Unsupported fused near-field lmax={lmax}.")
    cupy, _ = import_cupy()
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)
    real_dtype = np.dtype(np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64)
    accum_real_dtype = np.dtype(np.float32 if accum_dtype == np.dtype(np.complex64) else np.float64)

    # Geometry is independent of field compute precision: retain coordinates
    # through subtraction, then evaluate radial/angular functions in REAL.
    pts = np.asarray(field_points, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    coeff_arr = np.asarray(coeffs, dtype=compute_dtype).reshape(pos.shape[0], n_modes(int(lmax)))
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        empty = np.zeros((pts.shape[0], 3), dtype=accum_dtype)
        return empty, empty.copy() if compute_magnetic else None

    h_re, h_im, d_re, d_im = _packed_lut_arrays(lut, real_dtype=real_dtype)
    pts_cp = cupy.asarray(np.ascontiguousarray(pts).reshape(-1), dtype=cupy.float64)
    pos_cp = cupy.asarray(np.ascontiguousarray(pos.T).reshape(-1), dtype=cupy.float64)
    coeff_cp = cupy.asarray(np.ascontiguousarray(coeff_arr.T).reshape(-1), dtype=compute_dtype)
    h_re_cp = cupy.asarray(h_re, dtype=real_dtype)
    h_im_cp = cupy.asarray(h_im, dtype=real_dtype)
    d_re_cp = cupy.asarray(d_re, dtype=real_dtype)
    d_im_cp = cupy.asarray(d_im, dtype=real_dtype)
    out_e_cp = cupy.empty((pts.shape[0], 3), dtype=accum_dtype)
    out_h_cp = cupy.empty_like(out_e_cp) if compute_magnetic else None

    medium_factor = -1j * complex(n_medium)
    scalar_real = np.float32 if real_dtype == np.dtype(np.float32) else np.float64
    scalar_acc = np.float32 if accum_real_dtype == np.dtype(np.float32) else np.float64
    args: list[Any] = [
        np.int32(pts.shape[0]),
        np.int32(pos.shape[0]),
        scalar_real(k),
        scalar_real(1.0 / float(lut.dr)),
        np.int32(lut.ri.size),
        scalar_acc(np.real(medium_factor)),
        scalar_acc(np.imag(medium_factor)),
        pts_cp,
        pos_cp,
        coeff_cp,
        h_re_cp,
        h_im_cp,
        d_re_cp,
        d_im_cp,
        out_e_cp,
    ]
    if out_h_cp is not None:
        args.append(out_h_cp)
    kernel = _scattered_field_raw_kernel(
        int(lmax), compute_dtype.name, accum_dtype.name, bool(compute_magnetic)
    )
    kernel((int(pts.shape[0]),), (_BLOCK_SIZE,), tuple(args))
    e = asnumpy(out_e_cp).astype(accum_dtype, copy=False)
    if out_h_cp is None:
        return e, None
    return e, asnumpy(out_h_cp).astype(accum_dtype, copy=False)


__all__ = ["compute_scattered_field_cupy_fused"]
