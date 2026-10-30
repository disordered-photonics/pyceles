"""Periodic Green fields and native internal fields of driven dipole arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyceles.core.lattice import periodic_image_mask
from pyceles.core.particles import LayeredSphere, ParticleCollection, Sphere
from pyceles.core.periodic.reception import apply_periodic_reception
from pyceles.core.periodic.types import periodic_source_k_parallel
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.core.sources.embedded import prepare_dipole_source
from pyceles.postprocessing.farfield.periodic import periodic_outgoing_amplitudes

from .classification import classify_periodic_internal_points
from .components import NearFieldComponents
from .excitation import exciting_coefficients, spherical_exciting_fields
from .internal import compute_internal_field
from .periodic_exterior import _accumulate_order_field
from .periodic_interior import _local_regular_l1_fields_at_center
from .slice import reshape_field_points

if TYPE_CHECKING:
    from pyceles.simulation import ChannelResult


def periodic_outgoing_fields(
    run: ChannelResult,
    points: np.ndarray,
    positions: np.ndarray,
    coefficients: np.ndarray,
    *,
    lmax: int,
    exclude_self: bool = False,
    field_bmax: float | None = None,
    circumscribing_radii: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate outgoing arrays at any height; optionally subtract own bulk fields.

    Self subtraction keeps the entire regular image field. Ordinary field
    queries instead mark an active source's true position as NaN.
    """
    periodic = run.config.periodic
    assert periodic is not None
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    coefficients = np.asarray(coefficients)
    active = np.any(coefficients != 0, axis=1)
    if not np.all(active):
        positions, coefficients = positions[active], coefficients[active]
        if circumscribing_radii is not None:
            circumscribing_radii = np.asarray(circumscribing_radii)[active]
    singular = np.zeros(len(points), dtype=bool)
    if not exclude_self:
        for center, coeff in zip(positions, coefficients, strict=True):
            if not np.any(coeff):
                continue
            singular |= periodic_image_mask(
                points, center, ax=periodic.lattice.ax, ay=periodic.lattice.ay
            )
    q = periodic_source_k_parallel(periodic, run.source)
    regular_points = ~singular
    e = np.full((len(points), 3), np.nan, dtype=run.accum_dtype)
    h = np.full_like(e, np.nan)
    if (
        field_bmax is not None
        and periodic.options.method != "directsum"
        and len(positions)
        and not exclude_self
    ):
        upper, lower = float(positions[:, 2].max()), float(positions[:, 2].min())
        above = regular_points & (points[:, 2] > upper)
        below = regular_points & (points[:, 2] < lower)
        if np.any(above | below):
            outgoing = periodic_outgoing_amplitudes(
                lattice=periodic.lattice,
                positions=positions,
                coeffs=np.asarray(coefficients, dtype=run.compute_dtype),
                lmax=lmax,
                k=run.k,
                k_parallel=q,
                output_bmax=field_bmax,
                reference_z=(upper, lower),
            )
            for rows, sign, amplitude, reference in (
                (above, 1, outgoing.up_amplitudes, upper),
                (below, -1, outgoing.down_amplitudes, lower),
            ):
                if not np.any(rows):
                    continue
                e[rows], h[rows] = _accumulate_order_field(
                    points[rows],
                    k=run.k,
                    n_medium=run.config.n_medium,
                    order_k_parallel=outgoing.orders.k_parallel,
                    order_kz=sign * outgoing.orders.kz,
                    amplitudes=amplitude,
                    reference_z=reference,
                )
            regular_points &= ~(above | below)
    local_l1 = apply_periodic_reception(
        points[regular_points],
        positions,
        coefficients,
        lmax_out=1,
        lmax_in=lmax,
        k=run.k,
        periodic=periodic,
        k_parallel=q,
        exclude_self=exclude_self,
        compute_dtype=run.compute_dtype,
        accum_dtype=run.accum_dtype,
        backend=run.config.resolved_postprocessing_backend(),
        circumscribing_radii=circumscribing_radii,
    )
    e[regular_points], h[regular_points] = _local_regular_l1_fields_at_center(
        local_l1_coeffs=local_l1,
        n_medium=run.config.n_medium,
        out_dtype=np.dtype(run.accum_dtype),
    )
    return e, h


def compute_periodic_dipole_field(
    run: ChannelResult,
    points: np.ndarray,
    *,
    field_bmax: float | None = None,
) -> NearFieldComponents:
    """Reconstruct total fields without treating a source plane as particle slab."""
    periodic = run.config.periodic
    assert periodic is not None
    assert isinstance(run.source, (DipoleSource, DipoleCollection))
    points, lead = reshape_field_points(points)
    q = periodic_source_k_parallel(periodic, run.source)
    particles = ParticleCollection.from_particles(run.particles)
    prepared = run._dipole_source or prepare_dipole_source(
        particles,
        run.source,
        lmax=run.config.lmax,
        lattice=periodic.lattice,
        k_parallel=q,
    )
    host = prepared.host_source
    field_bmax = periodic.options.output_bmax if field_bmax is None else field_bmax
    ei, hi = periodic_outgoing_fields(
        run,
        points,
        host.source_positions(),
        host.outgoing_coeffs(),
        lmax=1,
        field_bmax=field_bmax,
    )
    classification, wrapped, phase = classify_periodic_internal_points(
        points,
        particles,
        lattice_ax=periodic.lattice.ax,
        lattice_ay=periodic.lattice.ay,
        k_parallel=q,
        n_medium=run.config.n_medium,
    )
    inside = classification.inside_any
    for emitter in prepared.embedded:
        singular = periodic_image_mask(
            points,
            run.source.source_positions()[emitter.emitter_index],
            ax=periodic.lattice.ax,
            ay=periodic.lattice.ay,
        )
        # Preserve an exact source query through the wrapped-coordinate subtraction.
        wrapped[singular] = emitter.response.local_source.source_positions()[0]
    es, hs = np.zeros_like(ei), np.zeros_like(hi)
    es[~inside], hs[~inside] = periodic_outgoing_fields(
        run,
        points[~inside],
        run.positions,
        run.coeffs,
        lmax=run.config.lmax,
        field_bmax=field_bmax,
        circumscribing_radii=run.circumscribing_radii,
    )
    eint, hint = np.zeros_like(ei), np.zeros_like(hi)
    owners = classification.active_particle_indices
    incoming = exciting_coefficients(
        owners,
        particles=particles,
        host_source=host,
        coeffs=run.coeffs,
        k=run.k,
        lmax=run.config.lmax,
        lut_dr=run.config.radial_lut_dr,
        compute_dtype=run.compute_dtype,
        accum_dtype=run.accum_dtype,
        backend=run.config.resolved_postprocessing_backend(),
        periodic=periodic,
        k_parallel=q,
    )
    for local, index in enumerate(owners):
        owner = int(index)
        rows = classification.points_for_particle(owner)
        particle = particles[owner]
        if isinstance(particle, (Sphere, LayeredSphere)):
            eint[rows], hint[rows] = spherical_exciting_fields(
                wrapped[rows],
                particle=particle,
                incident=incoming[local],
                wavelength=run.config.wavelength,
                n_medium=run.config.n_medium,
                lmax=run.config.lmax,
                compute_dtype=run.compute_dtype,
                accum_dtype=run.accum_dtype,
                backend=run.config.resolved_postprocessing_backend(),
                source_responses=(
                    e.response for e in prepared.embedded if e.particle_index == owner
                ),
            )
        else:
            eint[rows], hint[rows], _ = compute_internal_field(
                wrapped[rows],
                run.coeffs[owner : owner + 1],
                k=run.k,
                lmax=run.config.lmax,
                particles=(particle,),
                n_medium=run.config.n_medium,
                compute_dtype=run.compute_dtype,
                accum_dtype=run.accum_dtype,
                backend=run.config.resolved_postprocessing_backend(),
                show_progress=False,
            )
    eint *= phase[:, None]
    hint *= phase[:, None]
    et, ht = ei + es, hi + hs
    et[inside], ht[inside] = eint[inside], hint[inside]
    shape = (*lead, 3)
    return NearFieldComponents(
        E_initial=ei.reshape(shape),
        H_initial=hi.reshape(shape),
        E_scattered=es.reshape(shape),
        H_scattered=hs.reshape(shape),
        E_internal=eint.reshape(shape),
        H_internal=hint.reshape(shape),
        E_total=et.reshape(shape),
        H_total=ht.reshape(shape),
        inside_mask=inside.reshape(lead),
    )
