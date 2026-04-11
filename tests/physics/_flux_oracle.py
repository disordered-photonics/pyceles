from __future__ import annotations

"""Validation-only circumsphere flux oracle for local absorption checks.

This helper is intentionally test-scoped and not part of the public
postprocessing surface.

Assumptions/limits:
- embedding medium must be non-absorbing (`Im(n_medium)=0`);
- integration surfaces are circumscribing spheres sampled just outside the
  particle (`radius_scale > 1`);
- surfaces must remain in host medium (best suited to isolated/sparse cases).
"""

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from pyceles.core.particles import Particle
from pyceles.postprocessing.nearfield import compute_total_field, poynting


def _validate_nonabsorbing_medium(n_medium: complex) -> float:
    """Return `Re(n_medium)` for non-absorbing host media."""
    n_c = complex(n_medium)
    if abs(n_c.imag) > 0.0:
        raise ValueError(
            "Circumsphere-flux absorption diagnostics require a non-absorbing host medium."
        )
    n_real = float(np.real(n_c))
    if n_real <= 0.0:
        raise ValueError("`n_medium` must be positive and real.")
    return n_real


def _midpoint_unit_sphere_grid(
    n_polar: int,
    n_azimuth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return midpoint normals and unit-sphere area weights."""
    nt = int(n_polar)
    na = int(n_azimuth)
    if nt <= 0 or na <= 0:
        raise ValueError("`n_polar` and `n_azimuth` must be positive integers.")

    theta = (np.arange(nt, dtype=float) + 0.5) * (np.pi / nt)
    phi = (np.arange(na, dtype=float) + 0.5) * (2.0 * np.pi / na)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    st = np.sin(tt)
    normals = np.stack((st * np.cos(pp), st * np.sin(pp), np.cos(tt)), axis=-1).reshape(-1, 3)
    unit_area_weights = (st * (np.pi / nt) * (2.0 * np.pi / na)).reshape(-1)
    return normals, unit_area_weights


def circumsphere_absorbed_power_quadrature(
    *,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    beam: Any,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    particles: Sequence[Particle],
    n_medium: complex,
    particle_indices: Sequence[int] | None = None,
    n_polar: int = 24,
    n_azimuth: int = 48,
    radius_scale: float = 1.05,
    batch_size: int = 2048,
    show_progress: bool = False,
    force_general_initial_field: bool = False,
    lut_dr: float = 1.0,
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> dict[str, Any]:
    """Estimate absorbed power via circumsphere Poynting-flux quadrature."""
    _validate_nonabsorbing_medium(n_medium)
    part = tuple(particles)
    n_particles = len(part)
    if n_particles == 0:
        return {
            "particle_indices": np.zeros((0,), dtype=np.int64),
            "n_surface_points_per_particle": int(n_polar) * int(n_azimuth),
            "P_outward_surface_particles": np.zeros((0,), dtype=np.float64),
            "P_abs_surface_particles": np.zeros((0,), dtype=np.float64),
            "P_outward_surface_total": 0.0,
            "P_abs_surface_total": 0.0,
        }

    if radius_scale <= 1.0:
        raise ValueError("`radius_scale` must be > 1.0 to stay outside circumscribing spheres.")

    if particle_indices is None:
        selected = np.arange(n_particles, dtype=np.int64)
    else:
        selected = np.asarray(particle_indices, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            return {
                "particle_indices": selected,
                "n_surface_points_per_particle": int(n_polar) * int(n_azimuth),
                "P_outward_surface_particles": np.zeros((0,), dtype=np.float64),
                "P_abs_surface_particles": np.zeros((0,), dtype=np.float64),
                "P_outward_surface_total": 0.0,
                "P_abs_surface_total": 0.0,
            }
        if np.any(selected < 0) or np.any(selected >= n_particles):
            raise IndexError(
                f"`particle_indices` must lie in [0, {n_particles - 1}] for this particle set."
            )

    normals_unit, unit_area_weights = _midpoint_unit_sphere_grid(
        n_polar=n_polar, n_azimuth=n_azimuth
    )
    q = int(normals_unit.shape[0])

    centers = np.asarray(
        [np.asarray(part[int(i)].position, dtype=float) for i in selected], dtype=float
    )
    radii = np.asarray([float(part[int(i)].circumscribing_radius()) for i in selected], dtype=float)
    eval_radii = radius_scale * radii

    points = centers[:, None, :] + eval_radii[:, None, None] * normals_unit[None, :, :]
    normals = np.broadcast_to(normals_unit[None, :, :], points.shape)
    area_weights = (eval_radii[:, None] ** 2) * unit_area_weights[None, :]
    owner = np.broadcast_to(np.arange(selected.size, dtype=np.int64)[:, None], (selected.size, q))

    points_flat = points.reshape(-1, 3)
    normals_flat = normals.reshape(-1, 3)
    weights_flat = area_weights.reshape(-1)
    owner_flat = owner.reshape(-1)

    e_total, h_total, inside_mask = compute_total_field(
        points_flat,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        beam=beam,
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        particles=part,
        n_medium=n_medium,
        batch_size=batch_size,
        show_progress=show_progress,
        force_general_initial_field=force_general_initial_field,
        lut_dr=lut_dr,
        backend=backend,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    if np.any(inside_mask):
        n_inside = int(np.count_nonzero(inside_mask))
        raise ValueError(
            f"Circumsphere sampling produced {n_inside} points classified as inside a particle. "
            "Increase `radius_scale` to keep the integration surface in the host medium."
        )

    s_vec = poynting(e_total, h_total)
    radial_outward = np.einsum("ij,ij->i", s_vec, normals_flat)
    outward_by_particle = np.bincount(
        owner_flat,
        weights=(radial_outward * weights_flat),
        minlength=selected.size,
    ).astype(np.float64, copy=False)
    absorbed_by_particle = -outward_by_particle

    return {
        "particle_indices": selected.astype(np.int64, copy=False),
        "n_surface_points_per_particle": q,
        "P_outward_surface_particles": outward_by_particle,
        "P_abs_surface_particles": absorbed_by_particle,
        "P_outward_surface_total": float(np.sum(outward_by_particle)),
        "P_abs_surface_total": float(np.sum(absorbed_by_particle)),
    }


__all__ = ["circumsphere_absorbed_power_quadrature"]
