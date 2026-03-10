from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyceles.core.particles import Particle

from .classification import classify_internal_points
from .initial import compute_initial_field
from .internal import compute_internal_field
from .scattered import compute_scattered_field


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


def _positions_from_particles(particles: Sequence[Particle]) -> np.ndarray:
    """Return `(Ns,3)` centers derived from canonical particle descriptors."""
    part = tuple(particles)
    if len(part) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray([np.asarray(p.position, dtype=float) for p in part], dtype=float).reshape(
        -1, 3
    )


def compute_total_field(
    field_points: np.ndarray,
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    beam,
    *,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    particles: Sequence[Particle],
    n_medium: complex = 1.0 + 0j,
    batch_size: int = 2048,
    show_progress: bool = False,
    force_general_initial_field: bool = False,
    lut_dr: float = 1.0,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute total fields everywhere in CELES convention."""
    out = compute_near_field_components(
        field_points,
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
    coeffs: np.ndarray,
    k: float,
    lmax: int,
    beam,
    polar_angles: np.ndarray,
    azimuthal_angles: np.ndarray,
    particles: Sequence[Particle],
    n_medium: complex = 1.0 + 0j,
    batch_size: int = 2048,
    show_progress: bool = False,
    force_general_initial_field: bool = False,
    lut_dr: float = 1.0,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> NearFieldComponents:
    """Compute full near-field decomposition in one pass over shared points."""
    pts = np.asarray(field_points, dtype=float)
    pos = _positions_from_particles(particles)

    ei, hi = compute_initial_field(
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

    classification = None
    inside_hint = np.zeros(pts.shape[0], dtype=bool)
    if len(particles) > 0:
        classification = classify_internal_points(pts, particles, n_medium=n_medium)
        inside_hint = np.asarray(classification.inside_any, dtype=bool)
    es, hs = compute_scattered_field(
        field_points,
        pos,
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

    eint = np.zeros_like(ei)
    hint = np.zeros_like(hi)
    inside = np.zeros(pts.shape[0], dtype=bool)
    et = ei + es
    ht = hi + hs

    if len(particles) > 0:
        eint, hint, inside = compute_internal_field(
            field_points,
            coeffs,
            k=k,
            lmax=lmax,
            particles=particles,
            _point_classification=classification,
            n_medium=n_medium,
            show_progress=show_progress,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        es[inside] = 0
        hs[inside] = 0
        et[inside] = eint[inside]
        ht[inside] = hint[inside]

    return NearFieldComponents(
        E_initial=ei,
        H_initial=hi,
        E_scattered=es,
        H_scattered=hs,
        E_internal=eint,
        H_internal=hint,
        E_total=et,
        H_total=ht,
        inside_mask=inside,
    )


def poynting(e: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Time-averaged Poynting vector S = 0.5 * Re(E x H*)."""
    return 0.5 * np.real(np.cross(e, np.conj(h)))


__all__ = [
    "NearFieldComponents",
    "compute_near_field_components",
    "compute_total_field",
    "poynting",
]
