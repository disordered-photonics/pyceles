"""Dipole placement, particle-local emission and spherical interface response."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from pyceles._arrays import expose_read_only_view, owned_read_only_view

from ..indexing import n_modes
from ..lattice import RectangularLattice2D, nearest_image_points, periodic_image_mask
from ..particles import LayeredSphere, ParticleCollection, Sphere, particle_contains_points
from ..spherical_response import spherical_material_profile, spherical_response_coefficients
from ..svwf import expansion_fields, iter_mode_tensors
from .dipole import DipoleCollection, DipoleSource


@dataclass(frozen=True)
class PlacedDipole:
    """One native emitter represented beside its particle in the reference cell."""

    emitter_index: int
    particle_index: int
    particle: Sphere | LayeredSphere
    local_source: DipoleSource
    reference_phase: complex


@dataclass(frozen=True)
class DipolePlacement:
    """Order-independent geometry and local materials of an electric source.

    Host-only consumers need no material-interface solve. The original source
    retains its exterior-host convention; local source phases account for a
    periodic representative beside the containing particle.
    """

    source: DipoleSource | DipoleCollection
    host_source: DipoleSource | DipoleCollection
    positions: np.ndarray
    moments: np.ndarray
    particle_indices: np.ndarray
    local_refractive_indices: np.ndarray
    embedded: tuple[PlacedDipole, ...]
    nonreceiving_particles: tuple[int, ...]


@dataclass(frozen=True)
class EmbeddedDipole:
    """Bind an order-dependent native response to its source and particle."""

    emitter_index: int
    particle_index: int
    response: EmbeddedDipoleResponse


@dataclass(frozen=True)
class PreparedDipoleSource:
    """One scattering-order response over reusable source placement."""

    placement: DipolePlacement
    embedded: tuple[EmbeddedDipole, ...]
    lmax: int

    @property
    def host_source(self) -> DipoleSource | DipoleCollection:
        return self.placement.host_source

    @property
    def particle_indices(self) -> np.ndarray:
        return self.placement.particle_indices

    @property
    def nonreceiving_particles(self) -> tuple[int, ...]:
        return self.placement.nonreceiving_particles


def dipole_arrays(source: DipoleSource | DipoleCollection) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and amplitude-scaled physical electric moments."""
    moments = (
        source.dipole_moments()
        if isinstance(source, DipoleSource)
        else source.dipole_moments_array()
    )
    return source.source_positions(), moments


def locate_dipole_source(
    particles: ParticleCollection,
    source: DipoleSource | DipoleCollection,
    *,
    lattice: RectangularLattice2D | None = None,
    k_parallel: np.ndarray | None = None,
) -> DipolePlacement:
    """Classify active emitters without constructing any radial response.

    Zero moments impose neither a singularity nor an internal-source capability.
    Non-emitting particles retain their ordinary T matrices. Matched native
    bodies remain host material, including a source at their centre.
    """
    positions, moments = dipole_arrays(source)
    # Collections already own sealed geometry; preserve it rather than copying
    # all exterior emitters merely to classify a small embedded subset.
    if positions.flags.writeable:
        positions = owned_read_only_view(positions)
    moments = owned_read_only_view(moments)
    owners = np.full(len(positions), -1, dtype=np.intp)
    local_indices = np.full(len(positions), source.medium_n, dtype=np.complex128)
    candidates: np.ndarray | list[Iterable[int]]
    if not len(particles):
        candidates = [[] for _ in positions]
    elif lattice is None:
        candidates = cKDTree(particles.positions).query_ball_point(
            positions, float(particles.circumscribing_radii.max())
        )
    else:
        candidates = []
        for position in positions:
            nearest, _ = nearest_image_points(
                particles.positions, position, ax=lattice.ax, ay=lattice.ay
            )
            distances = np.linalg.norm(nearest - position, axis=1)
            candidates.append(np.flatnonzero(distances <= particles.circumscribing_radii))
    embedded = []
    nonreceiving = set()
    profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for emitter, possible in enumerate(candidates):
        if not np.any(moments[emitter]):
            continue
        owner = None
        local_position = positions[emitter]
        reference_phase = 1.0 + 0j
        for raw_index in possible:
            index = int(raw_index)
            particle = particles[index]
            point, shift = positions[emitter], np.zeros(3)
            if lattice is not None:
                wrapped, shifts = nearest_image_points(
                    positions[emitter : emitter + 1],
                    np.asarray(particle.position),
                    ax=lattice.ax,
                    ay=lattice.ay,
                )
                point, shift = wrapped[0], shifts[0]
            distance = np.linalg.norm(point - particle.position)
            if distance > particle.circumscribing_radius():
                continue
            if isinstance(particle, (Sphere, LayeredSphere)):
                if index not in profiles:
                    profiles[index] = spherical_material_profile(particle, source.medium_n)
                radii, indices = profiles[index]
                if np.all(indices == source.medium_n):
                    coincident = distance == 0
                    if lattice is not None:
                        # Reconstruct an exact image: subtracting its lattice
                        # vector need not recover a fractional centre bitwise.
                        coincident = bool(
                            periodic_image_mask(
                                positions[emitter : emitter + 1],
                                np.asarray(particle.position),
                                ax=lattice.ax,
                                ay=lattice.ay,
                            )[0]
                        )
                    if coincident:
                        nonreceiving.add(index)
                    continue
                real_interfaces = indices[:-1] != indices[1:]
                if np.any(distance == radii[real_interfaces]):
                    raise ValueError("An emitter on a material interface needs a side prescription")
            if particle_contains_points(particle, point[None, :])[0]:
                if owner is not None:
                    raise ValueError("A dipole cannot belong to overlapping particle interiors")
                owner = index
                local_position = point
                reference_phase = (
                    1.0 + 0j
                    if k_parallel is None
                    else complex(np.exp(1j * (shift[:2] @ k_parallel)))
                )
            elif (
                not isinstance(particle, (Sphere, LayeredSphere))
                and distance < particle.circumscribing_radius()
            ):
                raise ValueError(
                    f"A host dipole inside the circumscribing sphere of {type(particle).__name__} "
                    "cannot illuminate the full particle with a regular expansion"
                )
        if owner is None:
            continue
        container = particles[owner]
        if not isinstance(container, (Sphere, LayeredSphere)):
            raise NotImplementedError(
                f"Embedded electric dipoles are unavailable inside {type(container).__name__}"
            )
        radii, indices = profiles[owner]
        region = np.searchsorted(
            radii, np.linalg.norm(local_position - container.position), "right"
        )
        local_indices[emitter] = indices[region]
        local_source = DipoleSource(
            source.wavelength,
            medium_n=source.medium_n,
            position=tuple(local_position),
            dipole_moment=tuple(moments[emitter] * reference_phase.conjugate()),
        )
        embedded.append(PlacedDipole(emitter, owner, container, local_source, reference_phase))
        owners[emitter] = owner
    host_source: DipoleSource | DipoleCollection = source
    if embedded:
        host = owners == -1
        host_source = DipoleCollection(
            source.wavelength,
            medium_n=source.medium_n,
            positions=positions[host],
            dipole_moments=moments[host],
            radial_lut_dr=source.radial_lut_dr,
        )
    return DipolePlacement(
        source,
        host_source,
        positions,
        moments,
        expose_read_only_view(owners),
        expose_read_only_view(local_indices),
        tuple(embedded),
        tuple(sorted(nonreceiving)),
    )


def prepare_dipole_source(
    particles: ParticleCollection,
    source: DipoleSource | DipoleCollection,
    *,
    lmax: int,
    lattice: RectangularLattice2D | None = None,
    k_parallel: np.ndarray | None = None,
    placement: DipolePlacement | None = None,
) -> PreparedDipoleSource:
    """Match native emission at one order, reusing geometry when available."""
    if placement is None:
        placement = locate_dipole_source(particles, source, lattice=lattice, k_parallel=k_parallel)
    embedded = tuple(
        EmbeddedDipole(
            emitter.emitter_index,
            emitter.particle_index,
            prepare_embedded_dipole(emitter.particle, emitter.local_source, lmax=lmax),
        )
        for emitter in placement.embedded
    )
    return PreparedDipoleSource(placement, embedded, lmax)


def source_expansion_coefficients(
    source: DipoleSource, *, centre: np.ndarray, lmax: int, inward: bool
) -> np.ndarray:
    """Green-function addition theorem for a dipole inside/outside a sphere.

    The outgoing expansion outside the emitter radius uses regular SVWFs at
    the emitter; the inward regular expansion uses outgoing SVWFs there.
    Angular duality reverses m, without conjugating the complex radial values.
    Angular modes have norm pi in the package convention, giving i*k*k0**2/pi.
    This is O(lmax**2) and needs no general mode-to-mode translation table.
    """
    k0 = 2 * np.pi / source.wavelength
    k = k0 * source.medium_n
    prefactor = 1j * k * k0**2 / np.pi
    moment = source.dipole_moments()[0]
    coefficients = np.zeros(n_modes(lmax), dtype=np.complex128)
    if not np.any(moment):
        return coefficients
    for m_indices, n_indices, m_all, n_all in iter_mode_tensors(
        source.source_positions(), centre=centre, k=k, lmax=lmax, outgoing=inward
    ):
        coefficients[m_indices] = prefactor * (m_all[0, ::-1] @ moment)
        coefficients[n_indices] = prefactor * (n_all[0, ::-1] @ moment)
    return coefficients


@dataclass(frozen=True)
class EmbeddedDipoleResponse:
    """Homogeneous interface response plus a particular dipole in its own layer."""

    centre: np.ndarray
    radii: np.ndarray
    indices: np.ndarray
    local_source: DipoleSource
    emitter_layer: int
    lmax: int
    regular: np.ndarray
    outgoing: np.ndarray

    @property
    def exterior_coefficients(self) -> np.ndarray:
        return cast(np.ndarray, self.outgoing[-1])

    def region_fields(
        self,
        points: np.ndarray,
        region: int,
        *,
        include_bulk: bool = True,
        compute_dtype: npt.DTypeLike = np.complex128,
        accum_dtype: npt.DTypeLike = np.complex128,
        backend: str = "numpy",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one region, including either side of its bounding interfaces.

        Callers own this domain choice; `fields` classifies ordinary queries.
        `include_bulk=False` selects only the homogeneous interface response,
        which is regular at the emitter and supplies its reaction field.
        """
        n = self.indices[region]
        k = 2 * np.pi / self.local_source.wavelength * n
        e, h = expansion_fields(
            points,
            centre=self.centre,
            regular=self.regular[region],
            outgoing=self.outgoing[region],
            k=k,
            n=n,
            lmax=self.lmax,
            compute_dtype=compute_dtype,
            accum_dtype=accum_dtype,
            backend=backend,
        )
        if include_bulk and region == self.emitter_layer:
            bulk_e, bulk_h = expansion_fields(
                points,
                centre=np.asarray(self.local_source.position),
                regular=np.zeros(n_modes(1), dtype=np.complex128),
                outgoing=self.local_source.outgoing_coeffs()[0],
                k=k,
                n=n,
                lmax=1,
                compute_dtype=compute_dtype,
                accum_dtype=accum_dtype,
                backend=backend,
            )
            e += bulk_e
            h += bulk_h
        return e, h

    def fields(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct finite-position E/Z0*H, retaining the emitter singularity."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        regions = np.searchsorted(self.radii, np.linalg.norm(points - self.centre, axis=1), "right")
        e = np.zeros((len(points), 3), dtype=np.complex128)
        h = np.zeros_like(e)
        for region in np.unique(regions):
            selected = regions == region
            e[selected], h[selected] = self.region_fields(points[selected], int(region))
        return e, h


def prepare_embedded_dipole(
    particle: Sphere | LayeredSphere, source: DipoleSource, *, lmax: int
) -> EmbeddedDipoleResponse:
    """Infer local material and solve nonmagnetic spherical interface equations.

    `source.medium_n` is the exterior host, as in the existing source API.
    Only the prepared particular field uses the emitter's inferred material.
    The banded interface system does not divide by a scattering coefficient.
    """
    if not isinstance(lmax, int) or lmax < 1:
        raise ValueError("lmax must be a positive integer")
    if not isinstance(particle, (Sphere, LayeredSphere)):
        raise TypeError("Embedded emitters require a Sphere or LayeredSphere")
    radii, indices = spherical_material_profile(particle, source.medium_n)
    centre = np.asarray(particle.position)
    displacement = centre - np.asarray(source.position)
    distance = float(np.linalg.norm(displacement))
    if np.any(distance == radii):
        raise ValueError("An emitter on a material interface needs a side prescription")
    emitter_layer = int(np.searchsorted(radii, distance, "right"))
    if emitter_layer == len(radii):
        raise ValueError("This local response requires an emitter inside the particle")
    local_source = replace(source, medium_n=complex(indices[emitter_layer]))
    source_outgoing = source_expansion_coefficients(
        local_source, centre=centre, lmax=lmax, inward=False
    )
    # The inward expansion is needed only at an emitter shell's inner boundary.
    # In the core it would be both unnecessary and singular for a centred source.
    source_regular = (
        source_expansion_coefficients(local_source, centre=centre, lmax=lmax, inward=True)
        if emitter_layer > 0
        else np.zeros(n_modes(lmax), dtype=np.complex128)
    )

    regular, outgoing = spherical_response_coefficients(
        wavelength=source.wavelength,
        radii=radii,
        indices=indices,
        lmax=lmax,
        source_layer=emitter_layer,
        source_regular=source_regular,
        source_outgoing=source_outgoing,
    )
    return EmbeddedDipoleResponse(
        centre, radii, indices, local_source, emitter_layer, lmax, regular, outgoing
    )
