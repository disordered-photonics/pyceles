"""External regular excitation and native spherical interior reception."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._optional import asnumpy
from pyceles.core.geometry_bounds import conservative_set_diameter
from pyceles.core.operators.coupling_pairwise_cupy import CuPyPairwiseCouplingOperator
from pyceles.core.particles import LayeredSphere, ParticleCollection, Sphere
from pyceles.core.periodic.reception import apply_periodic_reception
from pyceles.core.periodic.types import PeriodicSpec
from pyceles.core.sources import DipoleCollection, DipoleSource, Source
from pyceles.core.sources.embedded import EmbeddedDipoleResponse
from pyceles.core.spherical_response import (
    spherical_material_profile,
    spherical_response_coefficients,
)
from pyceles.core.svwf import expansion_fields
from pyceles.core.translation import RadialLUT, translation_ab5_table, translation_block


def exciting_coefficients(
    owners: np.ndarray,
    *,
    particles: ParticleCollection,
    host_source: Source,
    coeffs: np.ndarray,
    k: complex,
    lmax: int,
    lut_dr: float,
    compute_dtype: npt.DTypeLike,
    accum_dtype: npt.DTypeLike,
    backend: str = "numpy",
    periodic: PeriodicSpec | None = None,
    k_parallel: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct b_host + W*a only at the requested particle centres."""
    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=compute_dtype, accum_dtype=accum_dtype
    )
    positions = particles.positions
    if periodic is not None:
        assert k_parallel is not None
        if not isinstance(host_source, (DipoleSource, DipoleCollection)):
            raise TypeError("Rectangular source reception requires electric dipoles")
        incoming = apply_periodic_reception(
            positions[owners],
            host_source.source_positions(),
            host_source.outgoing_coeffs(),
            lmax_out=lmax,
            lmax_in=1,
            k=k,
            periodic=periodic,
            k_parallel=k_parallel,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
            backend=backend,
        )
        incoming += apply_periodic_reception(
            positions[owners],
            positions,
            coeffs,
            lmax_out=lmax,
            lmax_in=lmax,
            k=k,
            periodic=periodic,
            k_parallel=k_parallel,
            exclude_self=True,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
            backend=backend,
            circumscribing_radii=particles.circumscribing_radii,
        )
        return incoming
    incoming = host_source.incident_coeffs(positions[owners], lmax, dtype=compute_dtype).astype(
        accum_dtype
    )
    if len(particles) < 2 or not len(owners):
        return incoming
    lut = RadialLUT(
        lmax=lmax,
        k=k,
        r_max=conservative_set_diameter(positions),
        dr=1e-2 / abs(k) if lut_dr == 0 else lut_dr,
        dtype=compute_dtype,
    )
    # The outgoing table may contain a padded singular prefix. Validate the
    # physical distances used by these rows before either unchecked backend
    # samples it; unrelated source-source pairs are not part of this action.
    distances, _ = cKDTree(positions).query(positions[owners], k=2)
    lut.validate_interval(float(np.min(distances[:, 1])), conservative_set_diameter(positions))
    if backend == "cupy":
        coupling = CuPyPairwiseCouplingOperator(
            lmax=lmax,
            k=k,
            positions=positions,
            ab5=None,
            radial_lut=lut,
            dtype=compute_dtype,
        )
        received = coupling.apply_rows(
            np.asarray(coeffs).reshape(-1), owners, accum_dtype=accum_dtype
        )
        incoming += np.asarray(asnumpy(received)).reshape(len(owners), -1)
        return incoming
    ab5 = translation_ab5_table(lmax, dtype=compute_dtype)
    outgoing = np.asarray(coeffs, dtype=compute_dtype).astype(accum_dtype, copy=False)
    for local, owner in enumerate(owners):
        for emitter in range(len(particles)):
            if emitter != owner:
                block = translation_block(
                    lmax, k, positions[owner] - positions[emitter], ab5=ab5, radial_lut=lut
                )
                incoming[local] += block.astype(accum_dtype) @ outgoing[emitter]
    return incoming


def spherical_exciting_fields(
    points: np.ndarray,
    *,
    particle: Sphere | LayeredSphere,
    incident: np.ndarray,
    wavelength: float,
    n_medium: complex,
    lmax: int,
    compute_dtype: npt.DTypeLike,
    accum_dtype: npt.DTypeLike,
    backend: str = "numpy",
    source_responses: Iterable[EmbeddedDipoleResponse] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Physical internal response to regular excitation and native emission.

    Match the incident field directly, including at material matching and
    scattering zeros. No scattered-to-internal T-entry ratio is required.
    """
    radii, indices = spherical_material_profile(particle, n_medium)
    regular, outgoing = spherical_response_coefficients(
        wavelength=wavelength,
        radii=radii,
        indices=indices,
        lmax=lmax,
        incident=incident,
    )
    regions = np.searchsorted(radii, np.linalg.norm(points - particle.position, axis=1), "right")
    e = np.zeros((len(points), 3), dtype=accum_dtype)
    h = np.zeros_like(e)
    for region in np.unique(regions):
        selected = regions == region
        e[selected], h[selected] = expansion_fields(
            points[selected],
            centre=np.asarray(particle.position),
            regular=regular[region],
            outgoing=outgoing[region],
            k=2 * np.pi / wavelength * indices[region],
            n=indices[region],
            lmax=lmax,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
            backend=backend,
        )
    for response in source_responses:
        regions = np.searchsorted(
            response.radii, np.linalg.norm(points - response.centre, axis=1), "right"
        )
        singular = np.all(points == response.local_source.source_positions()[0], axis=1)
        e[singular], h[singular] = np.nan, np.nan
        for region in np.unique(regions):
            rows = (regions == region) & ~singular
            if not np.any(rows):
                continue
            local_e, local_h = response.region_fields(
                points[rows],
                int(region),
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
                backend=backend,
            )
            e[rows] += local_e
            h[rows] += local_h
    return e, h
