from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyceles.core.particles import Particle

from .nearfield_kernels import (
    compute_initial_field,
    compute_internal_field,
    compute_scattered_field,
)


@dataclass(frozen=True)
class NearFieldComponents:
    """Decomposed complex near fields evaluated at the same spatial points.

    Components follow standard multiple-scattering bookkeeping:
    - `initial`: incident source field in the host medium
    - `scattered`: re-radiated field from all particles
    - `internal`: regular field valid inside particles
    - `total`: outside `initial+scattered`, inside replaced by `internal`
    """

    E_initial: np.ndarray
    H_initial: np.ndarray
    E_scattered: np.ndarray
    H_scattered: np.ndarray
    E_internal: np.ndarray
    H_internal: np.ndarray
    E_total: np.ndarray
    H_total: np.ndarray
    inside_mask: np.ndarray


def compute_total_field(
    field_points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    beam,
    *,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    particles: Sequence[Particle] | None = None,
    n_medium: complex = 1.0 + 0j,
    batch_size: int = 2048,
    show_progress: bool = False,
    force_general_initial_field: bool = False,
    lut_dr: float = 1.0,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute total fields (everywhere) in CELES convention.

    Outside spheres:
        E_total = E_initial + E_scattered

    Inside spheres:
        E_total is replaced by the internal (regular) field evaluated with the
        internal coefficients (i.e. the physical total field inside the particle).

    This mirrors the logic in `celes_output.totalEField/totalHField`.

    Parameters
    ----------
    polar_angles, azimuthal_angles:
        Source-projection angular quadrature grids used for the incident
        wavebundle contribution.

    Returns
    -------
    E_total, H_total, inside_mask
        `inside_mask` is True where the point was inside any sphere.
    """

    out = compute_near_field_components(
        field_points,
        positions=positions,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        beam=beam,
        polar_angles=np.asarray(polar_angles, float),
        azimuthal_angles=np.asarray(azimuthal_angles, float),
        particles=particles,
        n_medium=n_medium,
        batch_size=batch_size,
        show_progress=show_progress,
        force_general_initial_field=force_general_initial_field,
        lut_dr=lut_dr,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    return out.E_total, out.H_total, out.inside_mask


def compute_near_field_components(
    field_points: np.ndarray,
    *,
    positions: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    beam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    particles: Sequence[Particle] | None = None,
    n_medium: complex = 1.0 + 0j,
    batch_size: int = 2048,
    show_progress: bool = False,
    force_general_initial_field: bool = False,
    lut_dr: float = 1.0,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> NearFieldComponents:
    """Compute full near-field decomposition in one pass over shared points.

    This is the canonical near-field evaluator used by higher-level workflows.
    It returns physically separated components so users can inspect incident,
    scattering, and internal contributions before/after the inside replacement.
    The angular grids are the source-projection quadrature nodes for the
    incident field, not generic far-field display bins.
    Compared to the baseline CELES workflow, scattered-field evaluation here
    skips points known to be inside spheres, because those samples are replaced
    by internal fields in the physical total-field definition.
    """

    pts = np.asarray(field_points, dtype=float)

    Ei, Hi = compute_initial_field(
        field_points,
        k=k,
        n_medium=n_medium,
        beam=beam,
        polar_angles=np.asarray(polar_angles, float),
        azimuthal_angles=np.asarray(azimuthal_angles, float),
        batch_size=batch_size,
        show_progress=show_progress,
        force_general_initial_field=force_general_initial_field,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )

    inside_hint = np.zeros(pts.shape[0], dtype=bool)
    if particles is not None:
        for p in particles:
            center = np.asarray(p.position, dtype=float).reshape(3)
            rr = float(p.circumscribing_radius())
            R = pts - center[None, :]
            # TODO(spheroids): circumscribing-radius masking is exact for
            # spherical particle families only. Introduce particle-native
            # point-containment capability before enabling non-spherical
            # internal-field replacement here.
            inside_hint |= np.sum(R * R, axis=1) < (rr**2)
    Es, Hs = compute_scattered_field(
        field_points,
        positions,
        coeffs,
        k=k,
        lmax=lmax,
        n_medium=n_medium,
        show_progress=show_progress,
        particle_distance_resolution=lut_dr,
        active_mask=(~inside_hint) if np.any(inside_hint) else None,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )

    Eint = np.zeros_like(Ei)
    Hint = np.zeros_like(Hi)
    inside = np.zeros(pts.shape[0], dtype=bool)

    Et = Ei + Es
    Ht = Hi + Hs

    if particles is not None and len(particles) > 0:
        Eint, Hint, inside = compute_internal_field(
            field_points,
            coeffs,
            k=k,
            lmax=lmax,
            particles=particles,
            n_medium=n_medium,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        # Exterior scattered-field expansions are not physically valid inside
        # particles and can diverge at exact sphere centers; sanitize these
        # entries so downstream storage/downcasting remains stable.
        Es[inside] = 0
        Hs[inside] = 0
        Et[inside] = Eint[inside]
        Ht[inside] = Hint[inside]

    return NearFieldComponents(
        E_initial=Ei,
        H_initial=Hi,
        E_scattered=Es,
        H_scattered=Hs,
        E_internal=Eint,
        H_internal=Hint,
        E_total=Et,
        H_total=Ht,
        inside_mask=inside,
    )


def poynting(E: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Time-averaged Poynting vector S = 0.5 * Re(E x H*)."""
    return 0.5 * np.real(np.cross(E, np.conj(H)))
