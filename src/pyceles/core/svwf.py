"""SVWF angular tensors, normalized radial factors, and mode indexing."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy.special import spherical_jn

from pyceles._arrays import expose_read_only_view
from pyceles._cupy_memory import cupy_allocator_snapshot
from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._optional import asnumpy, import_cupy, is_cupy_array
from pyceles.core.indexing import index_vswf
from pyceles.core.radial import dx_xz_hankel1, sph_hankel1
from pyceles.core.spherical import spherical_functions_trigon


@cache
def mode_indices_by_l(
    lmax: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """Cache per-l arrays for vectorized mode contractions."""
    lmax = int(lmax)
    out = []
    for l in range(1, lmax + 1):
        m_vals = np.arange(-l, l + 1, dtype=np.int32)
        abs_m = np.abs(m_vals).astype(np.int32)
        n1_idx = np.array([index_vswf(l, int(m), 1, lmax) for m in m_vals], dtype=np.int32)
        n2_idx = np.array([index_vswf(l, int(m), 2, lmax) for m in m_vals], dtype=np.int32)
        out.append(
            (
                expose_read_only_view(m_vals),
                expose_read_only_view(abs_m),
                expose_read_only_view(n1_idx),
                expose_read_only_view(n2_idx),
            )
        )
    return tuple(out)


def contract_modes(mode_coeffs: np.ndarray, mode_tensor: np.ndarray) -> np.ndarray:
    """Contract mode axis: (M,) x (B,M,3) -> (B,3)."""
    return cast(np.ndarray, np.matmul(np.transpose(mode_tensor, (0, 2, 1)), mode_coeffs))


def build_mode_tensors(
    *,
    l: int,
    m_vals: np.ndarray,
    abs_m: np.ndarray,
    phi: np.ndarray,
    e_r: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    pi_all: np.ndarray,
    tau_all: np.ndarray,
    p_all: np.ndarray,
    z_l: np.ndarray,
    z_over_kr: np.ndarray,
    dxxz_over_kr: np.ndarray,
    compute_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(batch, 2*l+1, 3)` M/N tensors from normalized radial factors."""
    if is_cupy_array(phi) or is_cupy_array(z_l):
        cupy, _ = import_cupy()
        xp = cupy
        dtype = cupy.dtype(compute_dtype)
        m_vals_arr = cupy.asarray(m_vals, dtype=cupy.int32)
        abs_m_arr = cupy.asarray(abs_m, dtype=cupy.int32)
    else:
        xp = np
        dtype = np.dtype(compute_dtype)
        m_vals_arr = np.asarray(m_vals, dtype=np.int32)
        abs_m_arr = np.asarray(abs_m, dtype=np.int32)

    real_scalar = np.float32 if np.dtype(compute_dtype) == np.dtype(np.complex64) else np.float64
    pref = real_scalar(1.0 / np.sqrt(2.0 * l * (l + 1.0)))
    ll = real_scalar(l * (l + 1.0))

    p_lm = p_all[l][abs_m_arr].T
    pi_lm = pi_all[l][abs_m_arr].T
    tau_lm = tau_all[l][abs_m_arr].T

    eimphi = xp.asarray(xp.exp(1j * phi[:, None] * m_vals_arr[None, :]), dtype=dtype)
    impi = xp.asarray((1j * m_vals_arr[None, :]) * pi_lm, dtype=dtype)

    theta_phi_m = impi[:, :, None] * e_theta[:, None, :] - tau_lm[:, :, None] * e_phi[:, None, :]
    m_all = pref * z_l[:, None, None] * theta_phi_m * eimphi[:, :, None]

    radial_er = (ll * z_over_kr)[:, None] * p_lm
    mix_theta_phi = tau_lm[:, :, None] * e_theta[:, None, :] + impi[:, :, None] * e_phi[:, None, :]
    n_all = (
        pref
        * (radial_er[:, :, None] * e_r[:, None, :] + dxxz_over_kr[:, None, None] * mix_theta_phi)
        * eimphi[:, :, None]
    )
    return m_all, n_all


def clear_caches() -> None:
    """Clear process-global SVWF helper caches."""

    mode_indices_by_l.cache_clear()


def regular_radial_factors(
    l: int, kr: np.ndarray, *, dtype: npt.DTypeLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample j_l, j_l/z and (z*j_l)'/z for l >= 1, including the origin.

    Native SciPy evaluation and normalization precede the field-compute cast.
    No nonzero radial argument is replaced by a length-dependent floor.
    """
    argument = np.asarray(kr, dtype=np.complex128)
    jl = np.asarray(spherical_jn(l, argument))
    jl_over_z = np.divide(
        jl,
        argument,
        out=np.full_like(jl, 1 / 3 if l == 1 else 0),
        where=argument != 0,
    )
    derivative_over_z = jl_over_z + spherical_jn(l, argument, derivative=True)
    return jl.astype(dtype), jl_over_z.astype(dtype), np.asarray(derivative_over_z, dtype=dtype)


def spherical_frame(
    displacements: np.ndarray,
    radii: np.ndarray,
    *,
    xp: Any,
    dtype: np.dtype,
) -> tuple[Any, ...]:
    """Use the z-axis at the origin; regular radial limits remove that choice."""
    rvec = xp.asarray(displacements, dtype=dtype)
    r = xp.asarray(radii, dtype=dtype)
    nonzero = r != 0
    safe_r = xp.where(nonzero, r, 1.0)
    ct = xp.where(nonzero, rvec[:, 2] / safe_r, 1.0)
    st = xp.hypot(rvec[:, 0], rvec[:, 1]) / safe_r
    phi = xp.where(nonzero, xp.arctan2(rvec[:, 1], rvec[:, 0]), 0.0)
    ca, sa = xp.cos(phi), xp.sin(phi)
    e_r = xp.stack((st * ca, st * sa, ct), axis=1)
    e_theta = xp.stack((ct * ca, ct * sa, -st), axis=1)
    e_phi = xp.stack((-sa, ca, xp.zeros_like(phi)), axis=1)
    return ct, st, phi, e_r, e_theta, e_phi


_SVWF_PAIR_BATCH_SIZE = 65_536
_SVWF_CUPY_WORKSPACE_HEADROOM_FRACTION = 0.5


def _svwf_cupy_workspace_bytes_per_pair(
    *,
    lmax: int,
    compute_dtype: np.dtype,
) -> int:
    """Conservatively estimate CuPy allocator footprint per radial pair."""
    lmax_i = max(1, int(lmax))
    complex_dtype = np.dtype(compute_dtype)
    real_itemsize = np.dtype(
        np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
    ).itemsize
    complex_itemsize = complex_dtype.itemsize

    # pi/tau/P grids plus geometry/angular scratch.
    angular_real_values = 4 * (lmax_i + 1) ** 2
    # Cover the layered path (two radial bases), mode-tensor temporaries,
    # einsum inputs/outputs, and CuPy pool retention across degrees. This is
    # intentionally a safe allocator-footprint estimate, not a live tensor
    # byte count.
    mode_complex_values = 28 * lmax_i * (lmax_i + 2)
    fixed_real_values = 24
    fixed_complex_values = 24
    return int(
        (angular_real_values + fixed_real_values) * real_itemsize
        + (mode_complex_values + fixed_complex_values) * complex_itemsize
    )


def _svwf_pair_batch_size_for_workspace(
    *,
    total_pairs: int,
    lmax: int,
    compute_dtype: np.dtype,
    workspace_bytes: int,
) -> int:
    """Resolve a nonzero radial-pair batch within one workspace budget."""
    total = max(1, int(total_pairs))
    fast_cap = min(total, _SVWF_PAIR_BATCH_SIZE)
    bytes_per_pair = _svwf_cupy_workspace_bytes_per_pair(
        lmax=int(lmax), compute_dtype=np.dtype(compute_dtype)
    )
    memory_cap = max(1, int(workspace_bytes) // max(1, int(bytes_per_pair)))
    return max(1, min(fast_cap, memory_cap))


def _cupy_svwf_pair_batch_size(
    *,
    cupy: object,
    total_pairs: int,
    lmax: int,
    compute_dtype: np.dtype,
) -> int:
    """Choose a scalable CuPy pair batch from current guarded headroom.

    Keep half of usable device headroom outside this postprocessing workspace
    for the solver state, allocator fragmentation, and other temporaries. The
    fixed 65,536-pair path remains the maximum and therefore stays intact
    whenever the estimated workspace fits.
    """
    snapshot = cupy_allocator_snapshot(cupy, apply_pool_limit=False)
    reusable_or_fresh = int(snapshot.raw_free_bytes) + int(snapshot.pool_free_bytes)
    usable_headroom = max(
        1,
        min(int(snapshot.active_headroom_bytes), int(reusable_or_fresh)),
    )
    workspace_bytes = max(
        1,
        int(float(usable_headroom) * _SVWF_CUPY_WORKSPACE_HEADROOM_FRACTION),
    )
    return _svwf_pair_batch_size_for_workspace(
        total_pairs=int(total_pairs),
        lmax=int(lmax),
        compute_dtype=np.dtype(compute_dtype),
        workspace_bytes=int(workspace_bytes),
    )


def iter_mode_tensors(
    points: np.ndarray,
    *,
    centre: np.ndarray,
    k: complex,
    lmax: int,
    outgoing: bool,
    compute_dtype: npt.DTypeLike = np.complex128,
    radial_scales: np.ndarray | None = None,
    xp: Any = np,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield angular indices and native M/N tensors one degree at a time."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    displacements = points - centre
    radii = np.linalg.norm(displacements, axis=1)
    compute_dtype = np.dtype(compute_dtype)
    real_dtype = np.dtype(np.float32 if compute_dtype == np.dtype(np.complex64) else np.float64)
    ct, st, phi, e_r, e_theta, e_phi = spherical_frame(
        displacements, radii, xp=xp, dtype=real_dtype
    )
    pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=xp, return_plm=True)
    kr = k * radii
    if outgoing and np.any(radii == 0):
        raise ValueError("An outgoing expansion is singular at its centre")
    for l, (m_vals, abs_m, m_indices, n_indices) in enumerate(mode_indices_by_l(lmax), 1):
        scale = 1.0 if radial_scales is None else radial_scales[l - 1]
        if scale == 0:
            continue
        if outgoing:
            z = sph_hankel1(l, kr)
            native = z, z / kr, dx_xz_hankel1(l, kr) / kr
        else:
            native = regular_radial_factors(l, kr, dtype=np.complex128)
        # Combine the degree amplitude with native radial values before the
        # compute cast: a large coefficient times a tiny Bessel factor can
        # be representable even when the separate factors are not.
        radial = tuple(xp.asarray(value * scale, dtype=compute_dtype) for value in native)
        m_all, n_all = build_mode_tensors(
            l=l,
            m_vals=m_vals,
            abs_m=abs_m,
            phi=phi,
            e_r=e_r,
            e_theta=e_theta,
            e_phi=e_phi,
            pi_all=pi_all,
            tau_all=tau_all,
            p_all=p_all,
            z_l=radial[0],
            z_over_kr=radial[1],
            dxxz_over_kr=radial[2],
            compute_dtype=compute_dtype,
        )
        yield m_indices, n_indices, m_all, n_all


def expansion_fields(
    points: np.ndarray,
    *,
    centre: np.ndarray,
    regular: np.ndarray,
    outgoing: np.ndarray,
    k: complex,
    n: complex,
    lmax: int,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
    backend: str = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate regular/outgoing SVWFs with native double-precision radial values.

    Regular origin limits and angular tensors are shared with package fields.
    There is no LUT interpolation in this SVWF evaluation. CuPy uses the same
    native host radial samples, with bounded device angular/mode contractions.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=compute_dtype, accum_dtype=accum_dtype
    )
    if backend not in {"numpy", "cupy"}:
        raise ValueError(f"Unknown SVWF backend: {backend!r}")
    xp = import_cupy()[0] if backend == "cupy" else np
    e = np.zeros((len(points), 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    expansions = []
    for is_outgoing, coefficients in ((False, regular), (True, outgoing)):
        units = np.array(coefficients, dtype=np.complex128, copy=True)
        scales = np.zeros(lmax, dtype=float)
        for degree, (_, _, m_columns, n_columns) in enumerate(mode_indices_by_l(lmax)):
            columns = np.r_[m_columns, n_columns]
            scales[degree] = np.max(np.abs(units[columns]))
            if scales[degree] != 0:
                units[columns] /= scales[degree]
        if np.any(units):
            # Normalize once for all point batches. Preserve the compute cast
            # before uploading promoted contraction operands to either backend.
            amplitudes = xp.asarray(units.astype(compute_dtype), dtype=accum_dtype)
            expansions.append((is_outgoing, amplitudes, scales))
    batch_size = (
        _cupy_svwf_pair_batch_size(
            cupy=xp,
            total_pairs=len(points),
            lmax=lmax,
            compute_dtype=np.dtype(np.result_type(compute_dtype, accum_dtype)),
        )
        if backend == "cupy" and len(points)
        else max(1, len(points))
    )
    for start in range(0, len(points), batch_size):
        stop = min(len(points), start + batch_size)
        e_batch, h_batch = _expansion_fields_batch(
            points[start:stop],
            centre=centre,
            expansions=expansions,
            k=k,
            n=n,
            lmax=lmax,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
            xp=xp,
        )
        e[start:stop], h[start:stop] = asnumpy(e_batch), asnumpy(h_batch)
    return e, h


def _expansion_fields_batch(
    points: np.ndarray,
    *,
    centre: np.ndarray,
    expansions: list[tuple[bool, Any, np.ndarray]],
    k: complex,
    n: complex,
    lmax: int,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
    xp: Any,
) -> tuple[Any, Any]:
    e = xp.zeros((len(points), 3), dtype=accum_dtype)
    h = xp.zeros_like(e)
    h_factor = xp.asarray(-1j * n, dtype=compute_dtype)
    for is_outgoing, coefficients, scales in expansions:
        for m_indices, n_indices, m_all, n_all in iter_mode_tensors(
            points,
            centre=centre,
            k=k,
            lmax=lmax,
            outgoing=is_outgoing,
            compute_dtype=compute_dtype,
            radial_scales=scales,
            xp=xp,
        ):
            # Widen operands before products when accumulation is promoted.
            a = coefficients[m_indices]
            b = coefficients[n_indices]
            m_all = xp.asarray(m_all, dtype=accum_dtype)
            n_all = xp.asarray(n_all, dtype=accum_dtype)
            e += xp.einsum("m,pmc->pc", a, m_all) + xp.einsum("m,pmc->pc", b, n_all)
            h += h_factor * (xp.einsum("m,pmc->pc", a, n_all) + xp.einsum("m,pmc->pc", b, m_all))
    return e, h
