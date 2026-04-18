from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import spherical_jn, spherical_yn
from tqdm.auto import tqdm

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.particles import (
    LayeredSphere,
    Particle,
    Sphere,
    Spheroid,
    particle_contains_points,
)
from pyceles.core.spherical import spherical_functions_trigon
from pyceles.core.tmatrix import (
    _spheroid_internal_block,
    layered_internal_ab_ratios,
    sphere_internal_ratios,
)

from .classification import InternalPointClassification
from .common import build_internal_mode_tensors, contract_modes, mode_indices_by_l


def compute_internal_field(
    field_points: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    *,
    particles: Sequence[Particle],
    _point_classification: InternalPointClassification | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the physical total field inside particles."""
    return _compute_internal_field_particles(
        field_points,
        particles=particles,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        classification=_point_classification,
        n_medium=n_medium,
        show_progress=show_progress,
        backend=backend,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )


def _compute_internal_field_homogeneous_spheres(
    field_points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_particle: np.ndarray,
    inside_indices_by_sphere: Sequence[np.ndarray] | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical homogeneous-sphere internal-field kernel."""
    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    n_spheres = pos.shape[0]
    n_points = pts.shape[0]
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)

    rad = np.asarray(radii, dtype=float).reshape(-1)
    if rad.shape[0] != n_spheres:
        raise ValueError(f"radii must have length Ns={n_spheres}, got {rad.shape}")
    if inside_indices_by_sphere is not None and len(inside_indices_by_sphere) != n_spheres:
        raise ValueError(
            "`inside_indices_by_sphere` length must match number of spheres "
            f"({n_spheres}). Got {len(inside_indices_by_sphere)}."
        )

    if np.ndim(n_particle) == 0:
        n_particle_arr = np.full(n_spheres, complex(n_particle), dtype=complex)
    else:
        n_particle_arr = np.asarray(n_particle, dtype=complex).reshape(-1)
        if n_particle_arr.shape[0] != n_spheres:
            raise ValueError(
                f"n_particle must have length Ns={n_spheres}, got {n_particle_arr.shape}"
            )

    n_medium_c = complex(n_medium)
    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros((n_points, 3), dtype=accum_dtype)
    inside = np.zeros(n_points, dtype=bool)

    eps = 1e-12

    if str(backend).lower() == "cupy":
        return _compute_internal_field_homogeneous_spheres_cupy(
            pts,
            pos,
            rad,
            np.asarray(coeffs, dtype=compute_dtype),
            k=k,
            lmax=lmax,
            n_particle=n_particle_arr,
            inside_indices_by_sphere=inside_indices_by_sphere,
            n_medium=n_medium_c,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )

    sphere_iter: Iterable[int] = range(n_spheres)
    if show_progress:
        sphere_iter = tqdm(sphere_iter, desc="Internal field (spheres)", leave=True)

    mode_by_l = mode_indices_by_l(lmax)

    for j_sphere in sphere_iter:
        if inside_indices_by_sphere is None:
            r_full = pts - pos[j_sphere]
            r2_full = np.sum(r_full * r_full, axis=1)
            idx = np.flatnonzero(r2_full < (rad[j_sphere] ** 2))
        else:
            idx = np.asarray(inside_indices_by_sphere[j_sphere], dtype=np.intp).reshape(-1)
        if idx.size == 0:
            continue

        inside[idx] = True
        rvec = pts[idx] - pos[j_sphere]
        r2 = np.sum(rvec * rvec, axis=1)
        r = np.sqrt(r2)
        r_safe = np.where(r < eps, eps, r)

        x = rvec[:, 0]
        y = rvec[:, 1]
        z = rvec[:, 2]
        rho = np.sqrt(x * x + y * y)
        ct = z / r_safe
        st = rho / r_safe
        phi = np.arctan2(y, x)

        e_r = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
        e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
        pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)

        n_s = n_particle_arr[j_sphere]
        k_s = k * (n_s / n_medium_c)
        kr = k_s * r_safe
        ratios = sphere_internal_ratios(lmax, k, rad[j_sphere], n_s, n_medium_c)
        ratio_m = ratios[1]
        ratio_n = ratios[2]

        for l in range(1, lmax + 1):
            z_l = spherical_jn(l, kr)
            dz_l = spherical_jn(l, kr, derivative=True)
            dxxz = z_l + kr * dz_l

            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            m_all, n_all = build_internal_mode_tensors(
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
                z_l=np.asarray(z_l, dtype=compute_dtype),
                dxxz=np.asarray(dxxz, dtype=compute_dtype),
                kr=np.asarray(kr, dtype=compute_dtype),
                compute_dtype=compute_dtype,
            )
            a_int = coeffs[j_sphere, n1_idx].astype(compute_dtype, copy=False) * ratio_m[l]
            b_int = coeffs[j_sphere, n2_idx].astype(compute_dtype, copy=False) * ratio_n[l]

            e[idx] += contract_modes(a_int, m_all)
            e[idx] += contract_modes(b_int, n_all)
            h[idx] += (-1j * n_s) * contract_modes(a_int, n_all)
            h[idx] += (-1j * n_s) * contract_modes(b_int, m_all)

    return e, h, inside


def _compute_internal_field_homogeneous_spheres_cupy(
    field_points: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    n_particle: np.ndarray,
    inside_indices_by_sphere: Sequence[np.ndarray] | None,
    n_medium: complex,
    show_progress: bool,
    compute_dtype: np.dtype,
    accum_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CuPy homogeneous-sphere internal field path.

    The current GPU internal-field slice targets the dominant sphere case while
    keeping layered spheres and spheroids on the established CPU reference
    path. Mixed clusters therefore still work: sphere subsets can use CuPy
    while the more specialized particle families retain their validated NumPy
    kernels until a clearer hotspot justifies porting them.
    """
    cupy, _ = import_cupy()
    compute_dtype_cp = (
        cupy.complex64 if compute_dtype == np.dtype(np.complex64) else cupy.complex128
    )
    accum_dtype_cp = cupy.complex64 if accum_dtype == np.dtype(np.complex64) else cupy.complex128
    real_dtype_cp = cupy.float32 if compute_dtype == np.dtype(np.complex64) else cupy.float64
    eps = real_dtype_cp(1e-12)

    n_points = field_points.shape[0]
    n_spheres = positions.shape[0]
    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros((n_points, 3), dtype=accum_dtype)
    inside = np.zeros(n_points, dtype=bool)
    sphere_iter: Iterable[int] = range(n_spheres)
    if show_progress:
        sphere_iter = tqdm(sphere_iter, desc="Internal field (spheres)", leave=True)

    pts_gpu = cupy.asarray(field_points, dtype=real_dtype_cp)
    coeffs_gpu = cupy.asarray(np.asarray(coeffs, dtype=compute_dtype), dtype=compute_dtype_cp)
    mode_by_l = mode_indices_by_l(lmax)

    for j_sphere in sphere_iter:
        if inside_indices_by_sphere is None:
            r_full = field_points - positions[j_sphere]
            r2_full = np.sum(r_full * r_full, axis=1)
            idx = np.flatnonzero(r2_full < (radii[j_sphere] ** 2))
        else:
            idx = np.asarray(inside_indices_by_sphere[j_sphere], dtype=np.intp).reshape(-1)
        if idx.size == 0:
            continue

        inside[idx] = True
        idx_gpu = cupy.asarray(idx, dtype=cupy.int64)
        center_gpu = cupy.asarray(positions[j_sphere], dtype=real_dtype_cp)
        rvec = pts_gpu[idx_gpu] - center_gpu[None, :]
        r2 = cupy.sum(rvec * rvec, axis=1)
        r = cupy.sqrt(r2)
        r_safe = cupy.where(r < eps, eps, r)

        x = rvec[:, 0]
        y = rvec[:, 1]
        z = rvec[:, 2]
        rho = cupy.sqrt(x * x + y * y)
        ct = z / r_safe
        st = rho / r_safe
        phi = cupy.arctan2(y, x)

        e_r = cupy.stack([st * cupy.cos(phi), st * cupy.sin(phi), ct], axis=1)
        e_theta = cupy.stack([ct * cupy.cos(phi), ct * cupy.sin(phi), -st], axis=1)
        e_phi = cupy.stack([-cupy.sin(phi), cupy.cos(phi), cupy.zeros_like(phi)], axis=1)
        pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=cupy, return_plm=True)

        n_s = complex(n_particle[j_sphere])
        k_s = k * (n_s / n_medium)
        kr = compute_dtype_cp(k_s) * r_safe.astype(compute_dtype_cp, copy=False)
        ratios = sphere_internal_ratios(lmax, k, radii[j_sphere], n_s, n_medium)
        ratio_m = ratios[1]
        ratio_n = ratios[2]

        e_gpu = cupy.zeros((idx.size, 3), dtype=accum_dtype_cp)
        h_gpu = cupy.zeros_like(e_gpu)
        for l in range(1, lmax + 1):
            z_l = cupy.asarray(spherical_jn(l, asnumpy(kr)), dtype=compute_dtype_cp)
            dz_l = cupy.asarray(
                spherical_jn(l, asnumpy(kr), derivative=True), dtype=compute_dtype_cp
            )
            dxxz = z_l + kr * dz_l

            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            m_all, n_all = build_internal_mode_tensors(
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
                z_l=z_l,
                dxxz=dxxz,
                kr=kr,
                compute_dtype=compute_dtype,
            )
            a_int = coeffs_gpu[j_sphere, n1_idx].astype(compute_dtype_cp, copy=False) * ratio_m[l]
            b_int = coeffs_gpu[j_sphere, n2_idx].astype(compute_dtype_cp, copy=False) * ratio_n[l]
            e_gpu += contract_modes(a_int, m_all).astype(accum_dtype_cp, copy=False)
            e_gpu += contract_modes(b_int, n_all).astype(accum_dtype_cp, copy=False)
            h_gpu += (-1j * n_s) * contract_modes(a_int, n_all).astype(accum_dtype_cp, copy=False)
            h_gpu += (-1j * n_s) * contract_modes(b_int, m_all).astype(accum_dtype_cp, copy=False)

        e[idx] += asnumpy(e_gpu).astype(accum_dtype, copy=False)
        h[idx] += asnumpy(h_gpu).astype(accum_dtype, copy=False)

    return e, h, inside


def _compute_internal_field_particles(
    field_points: np.ndarray,
    particles: Sequence[Particle],
    coeffs: np.ndarray,
    *,
    k: float,
    lmax: int,
    classification: InternalPointClassification | None = None,
    n_medium: complex = 1.0 + 0j,
    show_progress: bool = False,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute internal fields for explicit particle descriptors."""
    pts = np.asarray(field_points, dtype=float).reshape(-1, 3)
    part = list(particles)
    n_particles = len(part)
    n_points = pts.shape[0]
    compute_dtype = np.dtype(compute_dtype)
    accum_dtype = np.dtype(accum_dtype)
    n_medium_c = complex(n_medium)
    lmax = int(lmax)
    n_modes_total = n_modes(lmax)

    e = np.zeros((n_points, 3), dtype=accum_dtype)
    h = np.zeros((n_points, 3), dtype=accum_dtype)
    inside = np.zeros(n_points, dtype=bool)
    if n_particles == 0:
        return e, h, inside

    if classification is not None:
        if classification.inside_any.shape != (n_points,):
            raise ValueError(
                "`classification.inside_any` must have shape "
                f"({n_points},). Got {classification.inside_any.shape}."
            )
        if len(classification.point_indices_by_particle) != n_particles:
            raise ValueError(
                "`classification.point_indices_by_particle` length must match "
                f"particle count ({n_particles}). Got {len(classification.point_indices_by_particle)}."
            )

    c = np.asarray(coeffs, dtype=compute_dtype)
    if c.size != n_particles * n_modes_total:
        raise ValueError(
            f"`coeffs` must have {n_particles * n_modes_total} entries for {n_particles} particles and lmax={lmax}. Got {c.size}."
        )
    c = c.reshape(n_particles, n_modes_total)

    spheres = [p for p in part if isinstance(p, Sphere)]
    if len(spheres) == n_particles:
        positions = np.asarray([sp.position for sp in spheres], dtype=float).reshape(n_particles, 3)
        radii = np.asarray([sp.radius for sp in spheres], dtype=float).reshape(n_particles)
        n_particle = np.asarray(
            [complex(sp.refractive_index) for sp in spheres], dtype=np.complex128
        )
        inside_idx = (
            [np.asarray(v, dtype=np.intp) for v in classification.point_indices_by_particle]
            if classification is not None
            else None
        )
        return _compute_internal_field_homogeneous_spheres(
            pts,
            positions,
            radii,
            c,
            k=float(k),
            lmax=lmax,
            n_particle=n_particle,
            inside_indices_by_sphere=inside_idx,
            n_medium=n_medium_c,
            show_progress=show_progress,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )

    supported = (Sphere, LayeredSphere, Spheroid)
    bad = [type(p).__name__ for p in part if not isinstance(p, supported)]
    if bad:
        raise TypeError(
            "compute_internal_field currently supports Sphere, LayeredSphere, and Spheroid in "
            f"particle-dispatch mode. Got {bad}."
        )

    sphere_idx = [j for j, p in enumerate(part) if isinstance(p, Sphere)]
    layered_idx = [j for j, p in enumerate(part) if isinstance(p, LayeredSphere)]
    spheroid_idx = [j for j, p in enumerate(part) if isinstance(p, Spheroid)]

    if sphere_idx:
        sphere_part = [p for p in part if isinstance(p, Sphere)]
        positions = np.asarray([sp.position for sp in sphere_part], dtype=float).reshape(-1, 3)
        radii = np.asarray([sp.radius for sp in sphere_part], dtype=float).reshape(-1)
        n_particle = np.asarray(
            [complex(sp.refractive_index) for sp in sphere_part], dtype=np.complex128
        )
        c_sphere = c[np.asarray(sphere_idx, dtype=int), :]
        inside_idx = (
            [
                np.asarray(classification.point_indices_by_particle[j], dtype=np.intp)
                for j in sphere_idx
            ]
            if classification is not None
            else None
        )
        e_s, h_s, inside_s = _compute_internal_field_homogeneous_spheres(
            pts,
            positions,
            radii,
            c_sphere,
            k=float(k),
            lmax=lmax,
            n_particle=n_particle,
            inside_indices_by_sphere=inside_idx,
            n_medium=n_medium_c,
            show_progress=show_progress,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        e += e_s
        h += h_s
        inside |= inside_s

    if spheroid_idx:
        eps = 1e-12
        mode_by_l = mode_indices_by_l(lmax)
        sph_iter: Iterable[int] = spheroid_idx
        if show_progress:
            sph_iter = tqdm(spheroid_idx, desc="Internal field (spheroids)", leave=True)
        internal_block_memo: dict[tuple[object, ...], np.ndarray] = {}

        for j_sphere in sph_iter:
            particle = part[j_sphere]
            if not isinstance(particle, Spheroid):
                continue
            if classification is None:
                idx = np.flatnonzero(particle_contains_points(particle, pts))
            else:
                idx = np.asarray(
                    classification.point_indices_by_particle[j_sphere], dtype=np.intp
                ).reshape(-1)
            if idx.size == 0:
                continue

            inside[idx] = True
            key = (
                type(particle),
                float(particle.equatorial_radius),
                float(particle.polar_radius),
                complex(particle.refractive_index),
                tuple(float(v) for v in particle.euler_angles),
                int(lmax),
                float(k),
                complex(n_medium_c),
            )
            internal_map = internal_block_memo.get(key)
            if internal_map is None:
                internal_map = _spheroid_internal_block(
                    lmax=lmax,
                    k_medium=float(k),
                    particle=particle,
                    n_medium=n_medium_c,
                )
                internal_block_memo[key] = internal_map

            c_internal = np.asarray(internal_map @ c[j_sphere], dtype=compute_dtype)
            center = np.asarray(particle.position, dtype=float).reshape(3)
            rvec = pts[idx] - center[None, :]
            r2 = np.sum(rvec * rvec, axis=1)
            r = np.sqrt(r2)
            r_safe = np.where(r < eps, eps, r)

            x = rvec[:, 0]
            y = rvec[:, 1]
            z = rvec[:, 2]
            rho = np.sqrt(x * x + y * y)
            ct = z / r_safe
            st = rho / r_safe
            phi = np.arctan2(y, x)

            e_r = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
            e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
            e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
            pi_all, tau_all, p_all = spherical_functions_trigon(
                ct, st, lmax, xp=np, return_plm=True
            )

            n_s = complex(particle.refractive_index)
            kr_full = float(k) * (n_s / n_medium_c) * r_safe

            for l in range(1, lmax + 1):
                m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
                a_int = c_internal[n1_idx].astype(compute_dtype, copy=False)
                b_int = c_internal[n2_idx].astype(compute_dtype, copy=False)
                jl = spherical_jn(l, kr_full)
                djl = spherical_jn(l, kr_full, derivative=True)
                m_reg, n_reg = build_internal_mode_tensors(
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
                    z_l=np.asarray(jl, dtype=compute_dtype),
                    dxxz=np.asarray(jl + kr_full * djl, dtype=compute_dtype),
                    kr=np.asarray(kr_full, dtype=compute_dtype),
                    compute_dtype=compute_dtype,
                )
                e[idx] += contract_modes(a_int, m_reg)
                e[idx] += contract_modes(b_int, n_reg)
                h[idx] += (-1j * n_s) * contract_modes(a_int, n_reg)
                h[idx] += (-1j * n_s) * contract_modes(b_int, m_reg)

    if not layered_idx:
        return e, h, inside

    eps = 1e-12
    mode_by_l = mode_indices_by_l(lmax)
    layer_iter: Iterable[int] = layered_idx
    if show_progress:
        layer_iter = tqdm(layered_idx, desc="Internal field (layered particles)", leave=True)

    for j_sphere in layer_iter:
        particle = part[j_sphere]
        if not isinstance(particle, LayeredSphere):
            continue
        center = np.asarray(particle.position, dtype=float).reshape(3)
        outer_radius = float(particle.circumscribing_radius())

        if classification is None:
            r_full = pts - center[None, :]
            r2_full = np.sum(r_full * r_full, axis=1)
            idx = np.flatnonzero(r2_full < (outer_radius**2))
        else:
            idx = np.asarray(
                classification.point_indices_by_particle[j_sphere], dtype=np.intp
            ).reshape(-1)
        if idx.size == 0:
            continue

        inside[idx] = True
        rvec = pts[idx] - center[None, :]
        r2 = np.sum(rvec * rvec, axis=1)
        r = np.sqrt(r2)
        r_safe = np.where(r < eps, eps, r)

        x = rvec[:, 0]
        y = rvec[:, 1]
        z = rvec[:, 2]
        rho = np.sqrt(x * x + y * y)
        ct = z / r_safe
        st = rho / r_safe
        phi = np.arctan2(y, x)

        e_r = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
        e_theta = np.stack([ct * np.cos(phi), ct * np.sin(phi), -st], axis=1)
        e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
        pi_all, tau_all, p_all = spherical_functions_trigon(ct, st, lmax, xp=np, return_plm=True)

        layer_radii = np.asarray(particle.layer_radii, dtype=float).reshape(-1)
        layer_n = np.asarray(particle.layer_refractive_indices, dtype=np.complex128).reshape(-1)
        layer_idx = np.searchsorted(layer_radii, r, side="right")
        layer_idx = np.clip(layer_idx, 0, layer_radii.size - 1)
        k_layers = float(k) * (layer_n / n_medium_c)
        layered_ratios = layered_internal_ab_ratios(
            lmax=lmax,
            k_medium=float(k),
            layer_radii=particle.layer_radii,
            layer_refractive_indices=particle.layer_refractive_indices,
            n_medium=n_medium_c,
        )
        a_m = layered_ratios[1]["A"]
        b_m = layered_ratios[1]["B"]
        a_n = layered_ratios[2]["A"]
        b_n = layered_ratios[2]["B"]

        for l in range(1, lmax + 1):
            m_vals, abs_m, n1_idx, n2_idx = mode_by_l[l - 1]
            a_out = c[j_sphere, n1_idx].astype(compute_dtype, copy=False)
            b_out = c[j_sphere, n2_idx].astype(compute_dtype, copy=False)

            for g in range(layer_radii.size):
                gmask = layer_idx == g
                if not np.any(gmask):
                    continue
                idx_g = idx[gmask]
                r_g = r_safe[gmask]
                phi_g = phi[gmask]
                kr = k_layers[g] * r_g

                jl = spherical_jn(l, kr)
                djl = spherical_jn(l, kr, derivative=True)
                use_h = not (
                    np.isclose(b_m[g, l], 0.0, rtol=0.0, atol=0.0)
                    and np.isclose(b_n[g, l], 0.0, rtol=0.0, atol=0.0)
                )
                if use_h:
                    yl = spherical_yn(l, kr)
                    hl = jl + 1j * yl
                    dyl = spherical_yn(l, kr, derivative=True)
                    dhl = djl + 1j * dyl
                else:
                    hl = np.zeros_like(jl, dtype=np.complex128)
                    dhl = np.zeros_like(djl, dtype=np.complex128)

                z_m = a_m[g, l] * jl + b_m[g, l] * hl
                dxxz_m = a_m[g, l] * (jl + kr * djl) + b_m[g, l] * (hl + kr * dhl)
                z_n = a_n[g, l] * jl + b_n[g, l] * hl
                dxxz_n = a_n[g, l] * (jl + kr * djl) + b_n[g, l] * (hl + kr * dhl)

                m_m, n_m = build_internal_mode_tensors(
                    l=l,
                    m_vals=m_vals,
                    abs_m=abs_m,
                    phi=phi_g,
                    e_r=e_r[gmask],
                    e_theta=e_theta[gmask],
                    e_phi=e_phi[gmask],
                    pi_all=pi_all[:, :, gmask],
                    tau_all=tau_all[:, :, gmask],
                    p_all=p_all[:, :, gmask],
                    z_l=np.asarray(z_m, dtype=compute_dtype),
                    dxxz=np.asarray(dxxz_m, dtype=compute_dtype),
                    kr=np.asarray(kr, dtype=compute_dtype),
                    compute_dtype=compute_dtype,
                )
                m_n, n_n = build_internal_mode_tensors(
                    l=l,
                    m_vals=m_vals,
                    abs_m=abs_m,
                    phi=phi_g,
                    e_r=e_r[gmask],
                    e_theta=e_theta[gmask],
                    e_phi=e_phi[gmask],
                    pi_all=pi_all[:, :, gmask],
                    tau_all=tau_all[:, :, gmask],
                    p_all=p_all[:, :, gmask],
                    z_l=np.asarray(z_n, dtype=compute_dtype),
                    dxxz=np.asarray(dxxz_n, dtype=compute_dtype),
                    kr=np.asarray(kr, dtype=compute_dtype),
                    compute_dtype=compute_dtype,
                )

                e[idx_g] += contract_modes(a_out, m_m)
                e[idx_g] += contract_modes(b_out, n_n)
                n_loc = complex(layer_n[g])
                h[idx_g] += (-1j * n_loc) * contract_modes(a_out, n_m)
                h[idx_g] += (-1j * n_loc) * contract_modes(b_out, m_n)

    return e, h, inside


__all__ = ["compute_internal_field"]
