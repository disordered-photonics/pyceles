"""Regular reaction fields and work delivered by prescribed electric dipoles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._optional import asnumpy, import_cupy
from pyceles.core.lattice import periodic_image_mask
from pyceles.core.particles import LayeredSphere, ParticleCollection, Sphere
from pyceles.core.periodic.types import periodic_source_k_parallel
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.core.sources.common import _validated_int
from pyceles.core.sources.embedded import PreparedDipoleSource, dipole_arrays, prepare_dipole_source
from pyceles.postprocessing.nearfield.excitation import (
    exciting_coefficients,
    spherical_exciting_fields,
)
from pyceles.postprocessing.nearfield.scattered import compute_scattered_field

if TYPE_CHECKING:
    from pyceles.simulation import ChannelResult


@dataclass(frozen=True)
class DipoleSourceWork:
    """Per-emitter work relative to its own homogeneous local material.

    `E_reaction_at_dipoles` subtracts each emitter's entire singular bulk field,
    retaining interface response and fields from all other coherent emitters.
    The lossless bulk self-power is restored in `power_total`. Individual
    coherent emitters can receive work (`power_total < 0`); only their sum is
    the total delivered power. `power_ratio` is not a collection LDOS.
    """

    dipole_positions: np.ndarray
    dipole_moments: np.ndarray
    local_refractive_indices: np.ndarray
    E_reaction_at_dipoles: np.ndarray
    delta_power: np.ndarray
    power_homogeneous: np.ndarray
    power_total: np.ndarray
    power_ratio: np.ndarray
    reaction_lmax: int


def _prepare_diagnostic(
    run: ChannelResult, reaction_lmax: int | None
) -> tuple[PreparedDipoleSource, np.ndarray, np.ndarray, int]:
    if not isinstance(run.source, (DipoleSource, DipoleCollection)):
        raise TypeError("Dipole diagnostics require a DipoleSource or DipoleCollection channel")
    order = (
        run.config.lmax
        if reaction_lmax is None
        else _validated_int("reaction_lmax", reaction_lmax, minimum=run.config.lmax)
    )
    periodic = run.config.periodic
    prepared = run._dipole_source
    if prepared is None or prepared.lmax != order:
        prepared = prepare_dipole_source(
            ParticleCollection.from_particles(run.particles),
            run.source,
            lmax=order,
            lattice=None if periodic is None else periodic.lattice,
            k_parallel=None
            if periodic is None
            else periodic_source_k_parallel(periodic, run.source),
            placement=None if prepared is None else prepared.placement,
        )
    positions, moments = prepared.placement.positions, prepared.placement.moments
    active_positions = positions[np.any(moments != 0, axis=1)]
    if periodic is None:
        coincident = len(np.unique(active_positions, axis=0)) != len(active_positions)
    else:
        coincident = any(
            np.any(
                periodic_image_mask(
                    active_positions[i + 1 :],
                    position,
                    ax=periodic.lattice.ax,
                    ay=periodic.lattice.ay,
                )
            )
            for i, position in enumerate(active_positions)
        )
    if coincident:
        raise ValueError(
            "Source-work decomposition requires distinct active emitter positions; combine coincident moments first"
        )
    return prepared, positions, moments, order


def _host_mutual_fields(
    run: ChannelResult,
    prepared: PreparedDipoleSource,
    positions: np.ndarray,
    moments: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Direct mutual dyadic Green fields, excluding each emitter's own bulk.

    The degree-one Green tensor has an elementary exact expression. Geometry
    differences precede the compute cast; accumulate in the requested dtype.
    Use the selected backend without a radial LUT or a source-distance floor.
    """
    compute, accum = resolve_compute_accum_dtypes(
        compute_dtype=run.compute_dtype, accum_dtype=run.accum_dtype
    )
    host_emitters = np.flatnonzero((prepared.particle_indices == -1) & np.any(moments != 0, axis=1))
    if len(host_emitters) < 2:
        return np.zeros((len(targets), 3), dtype=accum)
    xp = import_cupy()[0] if run.config.resolved_postprocessing_backend() == "cupy" else np
    real = np.float32 if compute == np.dtype(np.complex64) else np.float64
    fields = xp.zeros((len(targets), 3), dtype=accum)
    k = xp.asarray(run.k, dtype=compute)
    k0 = xp.asarray(run.k0, dtype=real)
    for emitter in host_emitters:
        selected = np.flatnonzero(targets != emitter)
        if not len(selected):
            continue
        displacement = positions[targets[selected]] - positions[emitter]
        radii = np.linalg.norm(displacement, axis=1)
        direction = xp.asarray(displacement / radii[:, None], dtype=real)
        r = xp.asarray(radii, dtype=real)
        moment = xp.asarray(moments[emitter], dtype=compute)
        longitudinal = direction * (direction @ moment)[:, None]
        scalar = xp.exp(1j * k * r) / (4 * np.pi * r)
        value = (k0**2 * scalar)[:, None] * (
            moment
            - longitudinal
            + (1 / (k * r) ** 2 - 1j / (k * r))[:, None] * (3 * longitudinal - moment)
        )
        fields[selected] += value
    return np.asarray(asnumpy(fields), dtype=accum)


def _reaction_field(
    run: ChannelResult,
    prepared: PreparedDipoleSource,
    positions: np.ndarray,
    moments: np.ndarray,
    *,
    show_progress: bool,
) -> np.ndarray:
    compute, accum = resolve_compute_accum_dtypes(
        compute_dtype=run.compute_dtype, accum_dtype=run.accum_dtype
    )
    active = np.flatnonzero(np.any(moments != 0, axis=1))
    field = np.zeros((len(positions), 3), dtype=accum)
    host_targets = active[prepared.particle_indices[active] == -1]
    if len(host_targets):
        if run.config.periodic is not None:
            from .nearfield.periodic_dipoles import periodic_outgoing_fields

            scattered, _ = periodic_outgoing_fields(
                run,
                positions[host_targets],
                run.positions,
                run.coeffs,
                lmax=run.config.lmax,
                circumscribing_radii=run.circumscribing_radii,
            )
            host = prepared.host_source
            mutual, _ = periodic_outgoing_fields(
                run,
                positions[host_targets],
                host.source_positions(),
                host.outgoing_coeffs(),
                lmax=1,
                exclude_self=True,
            )
            field[host_targets] = scattered + mutual
        else:
            scattered, _ = compute_scattered_field(
                positions[host_targets],
                run.positions,
                run.coeffs,
                k=run.k,
                lmax=run.config.lmax,
                n_medium=run.config.n_medium,
                particle_distance_resolution=run.config.radial_lut_dr,
                backend=run.config.resolved_postprocessing_backend(),
                compute_dtype=compute,
                accum_dtype=accum,
                show_progress=show_progress,
            )
            field[host_targets] = scattered + _host_mutual_fields(
                run, prepared, positions, moments, host_targets
            )
    if not prepared.embedded:
        return field
    backend = run.config.resolved_postprocessing_backend()
    periodic = run.config.periodic
    q = None if periodic is None else periodic_source_k_parallel(periodic, run.source)
    local_phase = np.ones(len(positions), dtype=np.complex128)
    if periodic is not None:
        positions = positions.copy()
        for placed in prepared.placement.embedded:
            index = placed.emitter_index
            local_phase[index] = placed.reference_phase
            positions[index] = placed.local_source.source_positions()[0]
    particles = ParticleCollection.from_particles(run.particles)
    owners = np.unique(prepared.particle_indices[active])
    owners = owners[owners >= 0]
    incoming = exciting_coefficients(
        owners,
        particles=particles,
        host_source=prepared.host_source,
        coeffs=run.coeffs,
        k=run.k,
        lmax=run.config.lmax,
        lut_dr=run.config.radial_lut_dr,
        compute_dtype=compute,
        accum_dtype=accum,
        backend=backend,
        periodic=periodic,
        k_parallel=q,
    )
    for local, owner in enumerate(owners):
        targets = active[prepared.particle_indices[active] == owner]
        particle = cast(Sphere | LayeredSphere, particles[int(owner)])
        e, _ = spherical_exciting_fields(
            positions[targets],
            particle=particle,
            incident=incoming[local],
            wavelength=run.config.wavelength,
            n_medium=run.config.n_medium,
            lmax=run.config.lmax,
            compute_dtype=compute,
            accum_dtype=accum,
            backend=backend,
        )
        field[targets] = e
        for emitter in prepared.embedded:
            if emitter.particle_index != owner:
                continue
            response = emitter.response
            regions = np.searchsorted(
                response.radii,
                np.linalg.norm(positions[targets] - response.centre, axis=1),
                "right",
            )
            for region in np.unique(regions):
                for self_term in (False, True):
                    rows = targets[
                        (regions == region) & ((targets == emitter.emitter_index) == self_term)
                    ]
                    if not len(rows):
                        continue
                    e, _ = response.region_fields(
                        positions[rows],
                        int(region),
                        include_bulk=not self_term,
                        compute_dtype=compute,
                        accum_dtype=accum,
                        backend=backend,
                    )
                    field[rows] += e
    return field * local_phase[:, None]


def compute_dipole_reaction_field(
    run: ChannelResult,
    *,
    reaction_lmax: int | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Return regular local-bulk-subtracted E at each active emitter.

    Absorbing emitter materials are allowed: no absolute bulk self-power is
    inferred. Zero-moment rows are zero. `reaction_lmax` increases the native
    source-interface response independently of the solved coupling order;
    global multiple-scattering convergence remains controlled by config.lmax.
    """
    prepared, positions, moments, _ = _prepare_diagnostic(run, reaction_lmax)
    return _reaction_field(run, prepared, positions, moments, show_progress=show_progress)


def compute_dipole_source_work(
    run: ChannelResult,
    *,
    reaction_lmax: int | None = None,
    show_progress: bool = False,
) -> DipoleSourceWork:
    """Work delivered by prescribed coherent electric dipoles in lossless local media.

    P_i = P0_i + (k0/2)*Im(conj(p_i).E_reaction_i), with
    P0_i = n_local_i*k0**4*|p_i|**2/(12*pi). Other layers and the exterior host
    may absorb, provided every active emitter's own material is lossless.
    Direct mutual fields are included; per-emitter work need not be positive.
    For periodic runs this is work per reduced cell, including self images
    and the prescribed Bloch phase, without plane-wave R/T normalization.
    """
    prepared, positions, moments, order = _prepare_diagnostic(run, reaction_lmax)
    local_n = prepared.placement.local_refractive_indices
    active = np.any(moments != 0, axis=1)
    if np.any(local_n[active].imag != 0):
        raise ValueError(
            "Point-emitter source work/self-power/LDOS requires a nonabsorbing local material"
        )
    reaction = _reaction_field(run, prepared, positions, moments, show_progress=show_progress)
    correction = run.k0 / 2 * np.imag(np.sum(moments.conj() * reaction, axis=1))
    p0 = local_n.real * run.k0**4 * np.sum(np.abs(moments) ** 2, axis=1) / (12 * np.pi)
    total = p0 + correction
    ratio = np.divide(total, p0, out=np.full_like(p0, np.nan), where=p0 != 0)
    return DipoleSourceWork(
        positions, moments, local_n, reaction, correction, p0, total, ratio, order
    )


def compute_dipole_ldos_enhancement(
    run: ChannelResult,
    *,
    reaction_lmax: int | None = None,
    show_progress: bool = False,
    squeeze: bool = True,
) -> float | np.ndarray:
    """Projected LDOS enhancement for a single active electric emitter.

    A coherent multi-emitter source has source work, not an individual LDOS
    inferred from a simultaneous excitation. Solve separate source channels
    for individual LDOS values, or use compute_dipole_source_work.
    """
    if not isinstance(run.source, (DipoleSource, DipoleCollection)):
        raise TypeError("LDOS enhancement requires an electric dipole channel")
    if run.config.periodic is not None:
        raise ValueError(
            "A fixed-Bloch dipole array has collective source work, not isolated-emitter LDOS; "
            "use compute_dipole_source_work"
        )
    _, moments = dipole_arrays(run.source)
    if np.count_nonzero(np.any(moments != 0, axis=1)) > 1:
        raise ValueError(
            "LDOS enhancement requires one active emitter; use source work for coherent collections"
        )
    ratio = compute_dipole_source_work(
        run, reaction_lmax=reaction_lmax, show_progress=show_progress
    ).power_ratio
    return float(ratio[0]) if squeeze and len(ratio) == 1 else ratio
