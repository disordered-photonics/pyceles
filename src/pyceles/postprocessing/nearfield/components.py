from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from pyceles.core.particles import Particle, ParticleCollection

from .classification import classify_internal_points
from .initial import compute_initial_field
from .internal import compute_internal_field
from .scattered import compute_scattered_electric_field, compute_scattered_field


class _NearFieldStageProgress:
    """Single user-facing progress bar for high-level finite near fields.

    Backend kernels expose very different natural work units: the NumPy
    reference path may iterate over particles, while the CuPy scattered-field
    path is intentionally one fused launch. A stage bar keeps the public
    workflow honest without splitting fused device work just to manufacture
    progress updates.
    """

    def __init__(self, *, enabled: bool, total: int) -> None:
        self._bar = (
            tqdm(total=int(total), desc="Near field", unit="stage", leave=True) if enabled else None
        )

    def status(self, label: str) -> None:
        if self._bar is not None:
            self._bar.set_postfix_str(str(label), refresh=True)

    def set_total(self, total: int) -> None:
        if self._bar is not None:
            self._bar.total = int(total)
            self._bar.refresh()

    def advance(self) -> None:
        if self._bar is not None:
            self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


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


@dataclass(frozen=True)
class ElectricFieldComponents:
    """Electric-only near-field decomposition on a common point set.

    This compact result is intended for dense visualization and spectral
    databases that never consume magnetic fields. It preserves the same
    initial/scattered/internal/total bookkeeping as :class:`NearFieldComponents`.
    """

    E_initial: np.ndarray
    E_scattered: np.ndarray
    E_internal: np.ndarray
    E_total: np.ndarray
    inside_mask: np.ndarray


def _positions_from_particles(particles: Sequence[Particle]) -> np.ndarray:
    """Return `(Ns,3)` centers derived from canonical particle descriptors."""
    if isinstance(particles, ParticleCollection):
        return particles.positions
    if len(particles) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(
        [np.asarray(p.position, dtype=float) for p in particles], dtype=float
    ).reshape(-1, 3)


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
    backend: str = "numpy",
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
        backend=backend,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    return out.E_total, out.H_total, out.inside_mask


def _compute_near_field_components(
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
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
    compute_magnetic: bool = True,
) -> NearFieldComponents | ElectricFieldComponents:
    """Compute near-field components through one shared classification workflow."""
    pts = np.asarray(field_points, dtype=float)
    pos = _positions_from_particles(particles)
    has_particles = len(particles) > 0
    progress = _NearFieldStageProgress(enabled=show_progress, total=3 if has_particles else 1)
    try:
        progress.status("initial")
        ei, hi = compute_initial_field(
            field_points,
            k=k,
            n_medium=n_medium,
            beam=beam,
            polar_angles=np.asarray(polar_angles, float),
            azimuthal_angles=np.asarray(azimuthal_angles, float),
            batch_size=batch_size,
            # High-level calls report physical stages on one stable bar. The
            # low-level helpers retain their granular progress when invoked
            # directly.
            show_progress=False,
            force_general_initial_field=force_general_initial_field,
            backend=backend,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
        )
        progress.advance()

        classification = None
        inside_hint = np.zeros(pts.shape[0], dtype=bool)
        if has_particles:
            progress.status("classifying points")
            classification = classify_internal_points(pts, particles, n_medium=n_medium)
            inside_hint = np.asarray(classification.inside_any, dtype=bool)
            if not np.any(inside_hint):
                progress.set_total(2)
            progress.status("scattered")

        if compute_magnetic:
            es, hs = compute_scattered_field(
                field_points,
                pos,
                coeffs,
                k=k,
                lmax=lmax,
                n_medium=n_medium,
                show_progress=False,
                backend=backend,
                particle_distance_resolution=lut_dr,
                active_mask=(~inside_hint) if np.any(inside_hint) else None,
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
            )
        else:
            es = compute_scattered_electric_field(
                field_points,
                pos,
                coeffs,
                k=k,
                lmax=lmax,
                n_medium=n_medium,
                show_progress=False,
                backend=backend,
                particle_distance_resolution=lut_dr,
                active_mask=(~inside_hint) if np.any(inside_hint) else None,
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
            )
            hs = None
        if has_particles:
            progress.advance()

        eint = np.zeros_like(ei)
        hint = np.zeros_like(hi) if compute_magnetic else None
        inside = inside_hint.copy()
        et = ei + es
        ht = hi + hs if hs is not None else None

        if has_particles and np.any(inside_hint):
            progress.status("internal")
            eint, hint_eval, inside = compute_internal_field(
                field_points,
                coeffs,
                k=k,
                lmax=lmax,
                particles=particles,
                _point_classification=classification,
                n_medium=n_medium,
                show_progress=False,
                backend=backend,
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
            )
            if compute_magnetic:
                if hint is None or hs is None or ht is None:
                    raise RuntimeError("Magnetic near-field components were not initialized.")
                hint[:] = hint_eval
            progress.advance()

        if np.any(inside):
            if compute_magnetic:
                if hint is None or hs is None or ht is None:
                    raise RuntimeError("Magnetic near-field components were not initialized.")
                hs[inside] = 0
                ht[inside] = hint[inside]
            es[inside] = 0
            et[inside] = eint[inside]

        if not compute_magnetic:
            return ElectricFieldComponents(
                E_initial=ei,
                E_scattered=es,
                E_internal=eint,
                E_total=et,
                inside_mask=inside,
            )
        if hint is None or hs is None or ht is None:
            raise RuntimeError("Magnetic near-field components were not initialized.")
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
    finally:
        progress.close()


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
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> NearFieldComponents:
    """Compute the full near-field decomposition on shared points."""
    result = _compute_near_field_components(
        field_points,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        beam=beam,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        particles=particles,
        n_medium=n_medium,
        batch_size=batch_size,
        show_progress=show_progress,
        force_general_initial_field=force_general_initial_field,
        lut_dr=lut_dr,
        backend=backend,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        compute_magnetic=True,
    )
    if not isinstance(result, NearFieldComponents):
        raise RuntimeError("Full near-field evaluation returned electric-only components.")
    return result


def compute_electric_field_components(
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
    backend: str = "numpy",
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> ElectricFieldComponents:
    """Compute only electric near-field components."""
    result = _compute_near_field_components(
        field_points,
        coeffs=coeffs,
        k=k,
        lmax=lmax,
        beam=beam,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
        particles=particles,
        n_medium=n_medium,
        batch_size=batch_size,
        show_progress=show_progress,
        force_general_initial_field=force_general_initial_field,
        lut_dr=lut_dr,
        backend=backend,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        compute_magnetic=False,
    )
    if not isinstance(result, ElectricFieldComponents):
        raise RuntimeError("Electric near-field evaluation returned full components.")
    return result


def poynting(e: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Time-averaged Poynting vector S = 0.5 * Re(E x H*)."""
    return 0.5 * np.real(np.cross(e, np.conj(h)))


__all__ = [
    "ElectricFieldComponents",
    "NearFieldComponents",
    "compute_electric_field_components",
    "compute_near_field_components",
    "compute_total_field",
    "poynting",
]
