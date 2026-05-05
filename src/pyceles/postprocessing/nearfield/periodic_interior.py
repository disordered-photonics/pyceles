from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

import numpy as np

from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.periodic import PeriodicSpec, plane_wave_k_parallel
from pyceles.core.periodic.ewald import default_ewald_eta
from pyceles.core.periodic.scalar import (
    factorial_int,
    real_integral_sequence,
    reciprocal_gamma,
    square_shell_indices,
    structural_sum_m_normalization,
    upper_gamma_sequence,
)
from pyceles.core.periodic.special import shifted_delta_sequence, shifted_reciprocal_regime
from pyceles.core.spherical import legendre_normalized_trigon
from pyceles.core.translation import translation_ab5_table

from .components import NearFieldComponents
from .periodic_exterior import _initial_plane_wave_field, _resolve_periodic_channel_payload
from .slice import reshape_field_points

if TYPE_CHECKING:
    from pyceles.simulation import SimulationResult


def _nearest_rectangular_image_distance(
    *,
    points: np.ndarray,
    center: np.ndarray,
    lattice_ax: float,
    lattice_ay: float,
) -> np.ndarray:
    """Return distance to the nearest rectangular-lattice image of one center."""
    dx = np.asarray(points[:, 0], dtype=float) - float(center[0])
    dy = np.asarray(points[:, 1], dtype=float) - float(center[1])
    dz = np.asarray(points[:, 2], dtype=float) - float(center[2])
    nx = np.rint(dx / float(lattice_ax))
    ny = np.rint(dy / float(lattice_ay))
    dx_wrap = dx - nx * float(lattice_ax)
    dy_wrap = dy - ny * float(lattice_ay)
    return np.sqrt(dx_wrap * dx_wrap + dy_wrap * dy_wrap + dz * dz)


def _inside_periodic_circumspheres(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    lattice_ax: float,
    lattice_ay: float,
    atol: float,
) -> np.ndarray:
    """Classify points that fall inside any periodic circumscribing-sphere image."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    rad = np.asarray(radii, dtype=float).reshape(-1)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return np.zeros((pts.shape[0],), dtype=bool)
    out = np.zeros((pts.shape[0],), dtype=bool)
    tol = float(max(0.0, atol))
    for j in range(pos.shape[0]):
        d = _nearest_rectangular_image_distance(
            points=pts,
            center=pos[j],
            lattice_ax=float(lattice_ax),
            lattice_ay=float(lattice_ay),
        )
        out |= d <= (float(rad[j]) + tol)
    return out


@cache
def _l1_projection_data(lmax: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Precompute the minimal contraction tensor for local `l=1` fields.

    Returns
    -------
    lmax_struct, m_offset, kernel, row_idx
        `kernel` has shape `(6, nm, 2*order+1, p_count)` and contracts the
        structural scalar table directly into the destination `l=1` sector,
        avoiding construction of the full `(nm x nm)` translation block.
    """
    lmax_i = int(lmax)
    if lmax_i < 1:
        raise ValueError(f"`lmax` must be >= 1. Got {lmax!r}.")
    nm = n_modes(lmax_i)
    row_idx: list[int] = []
    m_dst: list[int] = []
    m_src = np.zeros((nm,), dtype=np.int32)
    for _tau, l, m, idx in iter_modes(lmax_i):
        m_src[idx] = int(m)
        if l == 1:
            row_idx.append(int(idx))
            m_dst.append(int(m))
    row_idx_arr = np.asarray(row_idx, dtype=np.int64)
    m_dst_arr = np.asarray(m_dst, dtype=np.int32)

    ab5 = np.asarray(translation_ab5_table(lmax_i, dtype=np.complex128), dtype=np.complex128)
    ab5_l1 = np.asarray(ab5[row_idx_arr, :, :], dtype=np.complex128)

    max_degree = int(lmax_i + 1)  # p_max = l_dst + l_src with l_dst = 1
    lmax_struct = int((max_degree + 1) // 2)  # ensure 2*lmax_struct >= max_degree
    order = 2 * lmax_struct
    p_count = max_degree + 1
    m_offset = order
    kernel = np.zeros((ab5_l1.shape[0], nm, 2 * order + 1, p_count), dtype=np.complex128)
    for row in range(ab5_l1.shape[0]):
        dm_idx = m_src - int(m_dst_arr[row]) + m_offset
        for col in range(nm):
            kernel[row, col, int(dm_idx[col]), :p_count] = ab5_l1[row, col, :p_count]
    return lmax_struct, m_offset, kernel, row_idx_arr


def _reduce_structural_sums_to_l1(
    structural_sums: np.ndarray,
    coeffs: np.ndarray,
    *,
    kernel: np.ndarray,
) -> np.ndarray:
    """Contract batched structural sums directly into local `l=1` coefficients."""
    sums = np.asarray(structural_sums, dtype=np.complex128)
    src_coeffs = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
    p_count = int(kernel.shape[3])
    # sums: (n_points, order+1, 2*order+1), kernel: (6, nm, 2*order+1, p_count)
    # Only `p<=l_dst+l_src` contributes; slice the structural table accordingly.
    return np.asarray(
        np.einsum("npm,rcmp,c->nr", sums[:, :p_count, :], kernel, src_coeffs, optimize=True)
    )


def _same_plane_reciprocal_sums_batch(
    degree: int,
    order: int,
    *,
    c_xy: np.ndarray,
    k: float,
    k_parallel: np.ndarray,
    lattice,
    eta: float,
    shells: int,
) -> np.ndarray:
    """Vectorized same-plane reciprocal Ewald sum for many in-plane shifts."""
    l = int(degree)
    m = int(order)
    n_points = int(c_xy.shape[0])
    out = np.zeros((n_points,), dtype=np.complex128)
    if n_points == 0:
        return out
    if (l - abs(m)) % 2:
        return out

    root = np.sqrt(2 * l + 1.0) * np.sqrt(factorial_int(l - m)) * np.sqrt(factorial_int(l + m))
    prefactor = (1j) ** m * root / (lattice.area * float(k) * (2.0 * float(k)) ** l)
    kp0 = np.asarray(k_parallel, dtype=float).reshape(2)
    n_vals = np.arange((l - abs(m)) // 2 + 1, dtype=np.int64)
    max_n = int(n_vals[-1]) if n_vals.size else 0

    for shell in range(int(shells) + 1):
        indices = square_shell_indices(shell)
        reciprocal = np.asarray([p * lattice.b1 + q * lattice.b2 for p, q in indices], dtype=float)
        kgt = kp0[None, :] + reciprocal
        rho = np.linalg.norm(kgt, axis=1)
        phi = np.arctan2(kgt[:, 1], kgt[:, 0])
        gamma = reciprocal_gamma(float(k), rho)
        gamma_arg = -(gamma * gamma) / (4.0 * float(eta) * float(eta))
        gamma_fun = upper_gamma_sequence(max_n, gamma_arg)
        inner = np.zeros_like(gamma, dtype=np.complex128)
        for n in n_vals:
            denom = (
                factorial_int(n) * factorial_int((l + m) // 2 - n) * factorial_int((l - m) // 2 - n)
            )
            inner += (
                gamma_fun[:, int(n)] * gamma ** (2 * int(n) - 1) * rho ** (l - 2 * int(n)) / denom
            )
        phase = np.exp(-1j * (np.asarray(c_xy, dtype=float) @ kgt.T))
        out += phase @ (np.exp(1j * m * phi) * inner)
    return np.asarray(prefactor * out, dtype=np.complex128)


def _shifted_reciprocal_sums_batch(
    degree: int,
    order: int,
    *,
    c_vec: np.ndarray,
    k: float,
    k_parallel: np.ndarray,
    lattice,
    eta: float,
    shells: int,
) -> np.ndarray:
    """Vectorized shifted reciprocal Ewald sum for many point offsets."""
    l = int(degree)
    m = int(order)
    c = np.asarray(c_vec, dtype=float).reshape(-1, 3)
    n_points = int(c.shape[0])
    out = np.zeros((n_points,), dtype=np.complex128)
    if n_points == 0:
        return out

    same_plane = np.isclose(c[:, 2], 0.0, atol=0.0, rtol=0.0)
    if np.any(same_plane):
        out[same_plane] = _same_plane_reciprocal_sums_batch(
            l,
            m,
            c_xy=np.asarray(c[same_plane, :2], dtype=float),
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=float(eta),
            shells=int(shells),
        )
    if np.all(same_plane):
        return out

    root = np.sqrt(2 * l + 1.0) * np.sqrt(factorial_int(l - m)) * np.sqrt(factorial_int(l + m))
    prefactor = (-1j) ** m * root / (((-2.0) ** l) * lattice.area * float(k) * float(k))
    kp0 = np.asarray(k_parallel, dtype=float).reshape(2)
    n_vals = np.arange(0, l - abs(m) + 1, dtype=np.int64)
    if n_vals.size == 0:
        return out

    work_idx = np.flatnonzero(~same_plane)
    cz_vals = c[work_idx, 2]
    # Group repeated slice z-levels. For generic point clouds there may be no
    # repeats; the code still behaves correctly and avoids any cross-point cache.
    unique_cz, inverse = np.unique(cz_vals, return_inverse=True)

    for shell in range(int(shells) + 1):
        indices = square_shell_indices(shell)
        reciprocal = np.asarray([p * lattice.b1 + q * lattice.b2 for p, q in indices], dtype=float)
        kgt = kp0[None, :] + reciprocal
        rho = np.linalg.norm(kgt, axis=1)
        phi = np.arctan2(kgt[:, 1], kgt[:, 0])
        gamma = reciprocal_gamma(float(k), rho)
        phase_all = np.exp(-1j * (np.asarray(c[work_idx, :2], dtype=float) @ kgt.T))

        for group_id, cz in enumerate(unique_cz):
            idx = work_idx[inverse == group_id]
            regime = shifted_reciprocal_regime(gamma, float(cz))
            if regime == "same_plane":
                # Guarded above, but keep a safe fallback for exact-zero groups.
                out[idx] += _same_plane_reciprocal_sums_batch(
                    l,
                    m,
                    c_xy=np.asarray(c[idx, :2], dtype=float),
                    k=float(k),
                    k_parallel=k_parallel,
                    lattice=lattice,
                    eta=float(eta),
                    shells=0,
                )
                continue
            if regime == "small_shift":
                raise ValueError(
                    "Periodic interior evaluator hit the unresolved small-shift reciprocal regime; "
                    "add a stabilized small-shift path before evaluating these points."
                )

            inner = np.zeros((rho.size, n_vals.size), dtype=np.complex128)
            for n in n_vals:
                s_vals = np.arange(int(n), min(l - abs(m), 2 * int(n)) + 1, dtype=np.int64)
                s_vals = s_vals[s_vals % 2 == 1] if (l - abs(m)) % 2 else s_vals[s_vals % 2 == 0]
                if s_vals.size == 0:
                    continue
                terms = np.zeros_like(rho, dtype=np.complex128)
                for s in s_vals:
                    denom = (
                        factorial_int(2 * int(n) - int(s))
                        * factorial_int(int(s) - int(n))
                        * factorial_int((l + abs(m) - int(s)) // 2)
                        * factorial_int((l - abs(m) - int(s)) // 2)
                    )
                    terms += (
                        (-float(k) * float(cz)) ** (2 * int(n) - int(s))
                        * (rho / float(k)) ** (l - int(s))
                        / denom
                    )
                inner[:, int(n)] = terms
            delta = shifted_delta_sequence(int(n_vals[-1]), gamma, float(cz), float(eta))
            vec = np.exp(1j * m * phi) * np.sum(
                (gamma / float(k))[:, None] ** (2 * n_vals - 1) * delta * inner, axis=1
            )
            out[idx] += phase_all[inverse == group_id] @ vec

    out[work_idx] *= prefactor
    return out


def _shifted_real_sums_batch(
    degree: int,
    order: int,
    *,
    c_vec: np.ndarray,
    k: float,
    k_parallel: np.ndarray,
    lattice,
    eta: float,
    shells: int,
) -> np.ndarray:
    """Vectorized real-space Ewald sum for many point offsets."""
    l = int(degree)
    m = int(order)
    c = np.asarray(c_vec, dtype=float).reshape(-1, 3)
    n_points = int(c.shape[0])
    out = np.zeros((n_points,), dtype=np.complex128)
    if n_points == 0:
        return out
    if np.all(np.isclose(c[:, 2], 0.0, atol=0.0, rtol=0.0)) and (l - abs(m)) % 2:
        return out

    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    for shell in range(int(shells) + 1):
        indices = square_shell_indices(shell)
        shifts = np.asarray([p * lattice.a1 + q * lattice.a2 for p, q in indices], dtype=float)
        shifted = -(shifts[None, :, :] + c[:, None, :])
        radii = np.linalg.norm(shifted, axis=2)
        mask = radii > 0.0
        if not np.any(mask):
            continue
        point_idx, shell_idx = np.nonzero(mask)
        shifted_valid = shifted[point_idx, shell_idx, :]
        radii_valid = radii[point_idx, shell_idx]
        ct = shifted_valid[:, 2] / radii_valid
        st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
        phi = np.arctan2(shifted_valid[:, 1], shifted_valid[:, 0])
        plm = np.asarray(legendre_normalized_trigon(ct, st, max(1, l)), dtype=np.float64)
        phase_shell = np.exp(1j * (np.asarray(shifts[shell_idx, :2], dtype=float) @ kp))
        integral = (0.5) ** (l + 1.5) * real_integral_sequence(l, float(eta), float(k), radii_valid)
        angular = plm[l, abs(m), :] * np.exp(1j * m * phi) / structural_sum_m_normalization(m)
        contrib = phase_shell * (float(k) * radii_valid) ** l * angular * integral
        np.add.at(out, point_idx, contrib)
    return np.asarray(-1j * np.sqrt(2.0 / np.pi) * out, dtype=np.complex128)


def _ewald_structural_sums_batch(
    *,
    lmax_struct: int,
    k: float,
    destinations: np.ndarray,
    source: np.ndarray,
    lattice,
    k_parallel: np.ndarray,
    eta: float,
    real_shells: int,
    reciprocal_shells: int,
) -> np.ndarray:
    """Vectorized batch of pyceles-normalized scalar structural sums."""
    dest = np.asarray(destinations, dtype=float).reshape(-1, 3)
    source_arr = np.asarray(source, dtype=float).reshape(3)
    n_points = int(dest.shape[0])
    order = 2 * int(lmax_struct)
    offset = order
    sums = np.zeros((n_points, order + 1, 2 * order + 1), dtype=np.complex128)
    if n_points == 0:
        return sums

    c = source_arr[None, :] - dest
    cxy = np.asarray(c[:, :2], dtype=float)
    cz = np.asarray(c[:, 2], dtype=float)
    kp0 = np.asarray(k_parallel, dtype=float).reshape(2)

    # Reciprocal-space part: reuse shell geometry, phases, delta-sequences and
    # incomplete-gamma tables across all `(L, M)` channels.
    max_same_n = max(0, order // 2)
    unique_cz, inverse_cz = np.unique(cz, return_inverse=True)
    for shell in range(int(reciprocal_shells) + 1):
        indices = square_shell_indices(shell)
        reciprocal = np.asarray([p * lattice.b1 + q * lattice.b2 for p, q in indices], dtype=float)
        kgt = kp0[None, :] + reciprocal
        rho = np.linalg.norm(kgt, axis=1)
        phi = np.arctan2(kgt[:, 1], kgt[:, 0])
        gamma = reciprocal_gamma(float(k), rho)
        phase_all = np.exp(-1j * (cxy @ kgt.T))
        exp_m_phi = {m: np.exp(1j * m * phi) for m in range(-order, order + 1)}

        xarg = -(gamma * gamma) / (4.0 * float(eta) * float(eta))
        gamma_fun = upper_gamma_sequence(max_same_n, xarg)

        for group_id, cz_val in enumerate(unique_cz):
            point_mask = inverse_cz == group_id
            if not np.any(point_mask):
                continue
            phase = phase_all[point_mask]
            cz_f = float(cz_val)
            same_plane = np.isclose(cz_f, 0.0, atol=0.0, rtol=0.0)
            if same_plane:
                for degree in range(order + 1):
                    for m in range(-degree, degree + 1):
                        if (degree - abs(m)) % 2:
                            continue
                        root = (
                            np.sqrt(2 * degree + 1.0)
                            * np.sqrt(factorial_int(degree - m))
                            * np.sqrt(factorial_int(degree + m))
                        )
                        prefactor = (
                            (1j) ** m
                            * root
                            / (lattice.area * float(k) * (2.0 * float(k)) ** degree)
                        )
                        n_vals = np.arange((degree - abs(m)) // 2 + 1, dtype=np.int64)
                        inner = np.zeros_like(gamma, dtype=np.complex128)
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
                        sums[point_mask, degree, m + offset] += (
                            structural_sum_m_normalization(m) * prefactor * (phase @ vec)
                        )
                continue

            regime = shifted_reciprocal_regime(gamma, cz_f)
            if regime == "small_shift":
                raise ValueError(
                    "Periodic interior evaluator hit the unresolved small-shift reciprocal regime; "
                    "add a stabilized small-shift path before evaluating these points."
                )
            delta_full = shifted_delta_sequence(order, gamma, cz_f, float(eta))
            gamma_over_k = gamma / float(k)
            for degree in range(order + 1):
                for m in range(-degree, degree + 1):
                    root = (
                        np.sqrt(2 * degree + 1.0)
                        * np.sqrt(factorial_int(degree - m))
                        * np.sqrt(factorial_int(degree + m))
                    )
                    prefactor = (
                        (-1j) ** m
                        * root
                        / (((-2.0) ** degree) * lattice.area * float(k) * float(k))
                    )
                    n_vals = np.arange(0, degree - abs(m) + 1, dtype=np.int64)
                    if n_vals.size == 0:
                        continue
                    inner = np.zeros((rho.size, n_vals.size), dtype=np.complex128)
                    for n in n_vals:
                        s_vals = np.arange(
                            int(n), min(degree - abs(m), 2 * int(n)) + 1, dtype=np.int64
                        )
                        s_vals = (
                            s_vals[s_vals % 2 == 1]
                            if (degree - abs(m)) % 2
                            else s_vals[s_vals % 2 == 0]
                        )
                        if s_vals.size == 0:
                            continue
                        terms = np.zeros_like(rho, dtype=np.complex128)
                        for s in s_vals:
                            denom = (
                                factorial_int(2 * int(n) - int(s))
                                * factorial_int(int(s) - int(n))
                                * factorial_int((degree + abs(m) - int(s)) // 2)
                                * factorial_int((degree - abs(m) - int(s)) // 2)
                            )
                            terms += (
                                (-float(k) * cz_f) ** (2 * int(n) - int(s))
                                * (rho / float(k)) ** (degree - int(s))
                                / denom
                            )
                        inner[:, int(n)] = terms
                    vec = exp_m_phi[m] * np.sum(
                        gamma_over_k[:, None] ** (2 * n_vals - 1) * delta_full[:, n_vals] * inner,
                        axis=1,
                    )
                    sums[point_mask, degree, m + offset] += (
                        structural_sum_m_normalization(m) * prefactor * (phase @ vec)
                    )

    # Real-space part: reuse shell phases and vectorized normalized Legendre values.
    for shell in range(int(real_shells) + 1):
        indices = square_shell_indices(shell)
        shifts = np.asarray([p * lattice.a1 + q * lattice.a2 for p, q in indices], dtype=float)
        shifted = -(shifts[None, :, :] + c[:, None, :])
        radii = np.linalg.norm(shifted, axis=2)
        mask = radii > 0.0
        if not np.any(mask):
            continue
        point_idx, shell_idx = np.nonzero(mask)
        shifted_valid = shifted[point_idx, shell_idx, :]
        radii_valid = radii[point_idx, shell_idx]
        ct = shifted_valid[:, 2] / radii_valid
        st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
        phi = np.arctan2(shifted_valid[:, 1], shifted_valid[:, 0])
        plm = np.asarray(legendre_normalized_trigon(ct, st, max(1, order)), dtype=np.float64)
        phase_shell = np.exp(1j * (np.asarray(shifts[shell_idx, :2], dtype=float) @ kp0))
        exp_m_phi = {m: np.exp(1j * m * phi) for m in range(-order, order + 1)}
        kz_r = float(k) * radii_valid
        for degree in range(order + 1):
            integral = (0.5) ** (degree + 1.5) * real_integral_sequence(
                degree, float(eta), float(k), radii_valid
            )
            radial = phase_shell * kz_r**degree * integral
            for m in range(-degree, degree + 1):
                if np.all(np.isclose(cz, 0.0, atol=0.0, rtol=0.0)) and (degree - abs(m)) % 2:
                    continue
                angular = plm[degree, abs(m), :] * exp_m_phi[m] / structural_sum_m_normalization(m)
                contrib = -1j * np.sqrt(2.0 / np.pi) * radial * angular
                np.add.at(
                    sums[:, degree, m + offset],
                    point_idx,
                    structural_sum_m_normalization(m) * contrib,
                )
    return sums


def _periodic_local_regular_l1_coeffs(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    periodic: PeriodicSpec,
    k_parallel: np.ndarray,
    point_batch_size: int = 128,
) -> np.ndarray:
    """Return point-local regular `l=1` coefficients for periodic in-slab points.

    The hot path contracts periodic structural sums directly into the destination
    `l=1` sector.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    lmax_i = int(lmax)
    nm = n_modes(lmax_i)
    out = np.zeros((pts.shape[0], 6), dtype=np.complex128)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return out

    coeff_arr = np.asarray(coeffs, dtype=np.complex128).reshape(pos.shape[0], nm)
    lmax_struct, _m_offset, kernel, _row_idx = _l1_projection_data(lmax_i)
    method = str(periodic.options.method)
    if method != "ewald":
        raise NotImplementedError(
            "Periodic in-slab near-field evaluation currently requires "
            "`periodic.options.method='ewald'`."
        )

    batch = max(1, int(point_batch_size))
    eta = (
        default_ewald_eta(periodic.lattice)
        if periodic.options.eta is None
        else float(periodic.options.eta)
    )
    for s in range(0, pts.shape[0], batch):
        e = min(pts.shape[0], s + batch)
        pts_batch = np.asarray(pts[s:e], dtype=float)
        acc = np.zeros((pts_batch.shape[0], 6), dtype=np.complex128)
        for j in range(pos.shape[0]):
            sums = _ewald_structural_sums_batch(
                lmax_struct=lmax_struct,
                k=float(k),
                destinations=pts_batch,
                source=pos[j],
                lattice=periodic.lattice,
                k_parallel=k_parallel,
                eta=float(eta),
                real_shells=int(periodic.options.real_shells),
                reciprocal_shells=int(periodic.options.reciprocal_shells),
            )
            acc += _reduce_structural_sums_to_l1(sums, coeff_arr[j], kernel=kernel)
        out[s:e, :] = acc
    return out


def _local_regular_l1_fields_at_center(
    *,
    local_l1_coeffs: np.ndarray,
    n_medium: complex,
    out_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate point-local `E/H` from CELES-order local `l=1` coefficients.

    The column ordering is the CELES-mode order restricted to `l=1`:
    `tau=1,m=-1..1`, then `tau=2,m=-1..1`.

    Only the `l=1` regular sector contributes to the field exactly at the
    point-centered expansion origin; higher regular orders vanish there.
    """
    coeffs = np.asarray(local_l1_coeffs, dtype=np.complex128).reshape(-1, 6)
    n_points = int(coeffs.shape[0])
    e = np.zeros((n_points, 3), dtype=np.complex128)
    h = np.zeros((n_points, 3), dtype=np.complex128)

    a_m1 = coeffs[:, 0]
    a_0 = coeffs[:, 1]
    a_p1 = coeffs[:, 2]
    b_m1 = coeffs[:, 3]
    b_0 = coeffs[:, 4]
    b_p1 = coeffs[:, 5]

    c12 = 1.0 / np.sqrt(12.0)
    c6 = 1.0 / np.sqrt(6.0)
    medium = complex(n_medium)

    e[:, 0] = c12 * (b_m1 + b_p1)
    e[:, 1] = 1j * c12 * (b_p1 - b_m1)
    e[:, 2] = c6 * b_0

    h[:, 0] = -1j * medium * c12 * (a_m1 + a_p1)
    h[:, 1] = medium * c12 * (a_p1 - a_m1)
    h[:, 2] = -1j * medium * c6 * a_0
    return np.asarray(e, dtype=out_dtype), np.asarray(h, dtype=out_dtype)


def compute_periodic_near_field_interior(
    run: SimulationResult,
    *,
    points: np.ndarray,
    channel: Literal["mixed", "te", "tm"] = "mixed",
) -> NearFieldComponents:
    """Evaluate periodic near fields for points inside the particle slab.

    Notes
    -----
    This implementation intentionally excludes points that lie inside any
    periodic circumscribing-sphere image. Those points are marked in
    `inside_mask` and their scattered/total fields are returned as non-finite
    values.
    """
    periodic = run.config.periodic
    if periodic is None:
        raise ValueError(
            "`compute_periodic_near_field` requires a periodic simulation result "
            "(`run.config.periodic` must be set)."
        )
    if not isinstance(periodic, PeriodicSpec):
        raise TypeError(
            "Periodic near-field evaluation requires `run.config.periodic` to be a PeriodicSpec."
        )

    channel_key = str(channel).lower()
    if channel_key not in {"mixed", "te", "tm"}:
        raise ValueError("`channel` must be one of {'mixed', 'te', 'tm'}.")
    channel_literal: Literal["mixed", "te", "tm"] = (
        "mixed" if channel_key == "mixed" else ("te" if channel_key == "te" else "tm")
    )
    coeffs, source = _resolve_periodic_channel_payload(run, channel=channel_literal)

    pts_flat, lead_shape = reshape_field_points(points)
    n_points = int(pts_flat.shape[0])
    out_dtype = np.dtype(run.accum_dtype)

    e_initial, h_initial = _initial_plane_wave_field(
        pts_flat,
        source=source,
        k=float(run.k),
        n_medium=run.config.n_medium,
    )
    e_scat = np.zeros((n_points, 3), dtype=np.complex128)
    h_scat = np.zeros((n_points, 3), dtype=np.complex128)
    e_internal = np.zeros((n_points, 3), dtype=np.complex128)
    h_internal = np.zeros((n_points, 3), dtype=np.complex128)
    inside_mask = np.zeros((n_points,), dtype=bool)

    if n_points > 0 and run.n_particles > 0:
        ax = float(periodic.lattice.ax)
        ay = float(periodic.lattice.ay)
        sphere_atol = max(1.0e-12, 64.0 * np.finfo(np.dtype(run.accum_dtype)).eps)
        inside_mask = _inside_periodic_circumspheres(
            points=pts_flat,
            positions=run.positions,
            radii=run.circumscribing_radii,
            lattice_ax=ax,
            lattice_ay=ay,
            atol=sphere_atol,
        )
        valid_idx = np.flatnonzero(~inside_mask)
        if valid_idx.size > 0:
            local_l1 = _periodic_local_regular_l1_coeffs(
                points=np.asarray(pts_flat[valid_idx], dtype=float),
                positions=run.positions,
                coeffs=coeffs,
                lmax=int(run.config.lmax),
                k=float(run.k),
                periodic=periodic,
                k_parallel=np.asarray(plane_wave_k_parallel(source), dtype=float),
            )
            e_valid, h_valid = _local_regular_l1_fields_at_center(
                local_l1_coeffs=local_l1,
                n_medium=run.config.n_medium,
                out_dtype=np.dtype(np.complex128),
            )
            e_scat[valid_idx, :] = e_valid
            h_scat[valid_idx, :] = h_valid

    if np.any(inside_mask):
        e_scat[inside_mask, :] = np.nan + 0.0j
        h_scat[inside_mask, :] = np.nan + 0.0j
    e_total = e_initial + e_scat
    h_total = h_initial + h_scat

    if lead_shape == ():
        vec_shape: tuple[int, ...] = (3,)
        mask_shape: tuple[int, ...] = ()
    else:
        vec_shape = (*lead_shape, 3)
        mask_shape = lead_shape
    return NearFieldComponents(
        E_initial=np.asarray(e_initial, dtype=out_dtype).reshape(vec_shape),
        H_initial=np.asarray(h_initial, dtype=out_dtype).reshape(vec_shape),
        E_scattered=np.asarray(e_scat, dtype=out_dtype).reshape(vec_shape),
        H_scattered=np.asarray(h_scat, dtype=out_dtype).reshape(vec_shape),
        E_internal=np.asarray(e_internal, dtype=out_dtype).reshape(vec_shape),
        H_internal=np.asarray(h_internal, dtype=out_dtype).reshape(vec_shape),
        E_total=np.asarray(e_total, dtype=out_dtype).reshape(vec_shape),
        H_total=np.asarray(h_total, dtype=out_dtype).reshape(vec_shape),
        inside_mask=np.asarray(inside_mask, dtype=bool).reshape(mask_shape),
    )


__all__ = ["compute_periodic_near_field_interior"]
