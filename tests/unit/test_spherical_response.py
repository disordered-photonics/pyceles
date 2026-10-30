"""Direct native interface responses and the existing scattered-mode interiors."""

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.particles import LayeredSphere, Sphere
from pyceles.core.sources import DipoleSource, PlaneWave
from pyceles.core.sources.embedded import prepare_embedded_dipole
from pyceles.core.spherical_response import (
    spherical_material_profile,
    spherical_response_coefficients,
)
from pyceles.core.svwf import expansion_fields
from pyceles.core.tmatrix import particle_T_matrix_block
from pyceles.postprocessing.nearfield.initial import compute_initial_field
from pyceles.postprocessing.nearfield.internal import compute_internal_field


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.15j])
@pytest.mark.parametrize(
    "particle",
    [Sphere((13, -24, 5), 72, 2 + 0.1j), LayeredSphere((13, -24, 5), (36, 72), (1.7, 2 + 0.1j))],
    ids=["sphere", "layered"],
)
def test_exciting_and_scattered_mode_representations_give_the_same_fields(host, particle):
    lmax, wavelength = 3, 550
    rng = np.random.default_rng(913)
    incident = rng.normal(size=n_modes(lmax)) + 1j * rng.normal(size=n_modes(lmax))
    radii, indices = spherical_material_profile(particle, host)
    regular, outgoing = spherical_response_coefficients(
        wavelength=wavelength, radii=radii, indices=indices, lmax=lmax, incident=incident
    )
    k = 2 * np.pi / wavelength * host
    scattered = particle_T_matrix_block(lmax, k, particle, host) @ incident
    np.testing.assert_allclose(outgoing[-1], scattered, rtol=3e-12, atol=0)

    points = np.asarray(particle.position) + np.array(
        [[0, 0, 0], [10, 3, 1], [42, 2, 0], [68, 0, 0]]
    )
    expected_e, expected_h, inside = compute_internal_field(
        points, scattered[None, :], k, lmax, particles=[particle], n_medium=host
    )
    assert inside.all()
    regions = np.searchsorted(radii, np.linalg.norm(points - particle.position, axis=1), "right")
    for region in np.unique(regions):
        selected = regions == region
        actual = expansion_fields(
            points[selected],
            centre=np.asarray(particle.position),
            regular=regular[region],
            outgoing=outgoing[region],
            k=2 * np.pi / wavelength * indices[region],
            n=indices[region],
            lmax=lmax,
        )
        for value, expected in zip(
            actual, (expected_e[selected], expected_h[selected]), strict=True
        ):
            assert np.linalg.norm(value - expected) / np.linalg.norm(expected) < 3e-12


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.15j])
def test_matched_interfaces_retain_regular_excitation_without_dividing_by_scattering(host):
    particle = LayeredSphere((13, -24, 5), (36, 72), (host, host))
    source = PlaneWave(550, medium_n=host, polar_angle=0.4, azimuthal_angle=0.7)
    centre = np.asarray(particle.position)[None, :]
    incident = source.incident_coeffs(centre, 3)[0]
    radii, indices = spherical_material_profile(particle, host)
    regular, outgoing = spherical_response_coefficients(
        wavelength=550, radii=radii, indices=indices, lmax=3, incident=incident
    )
    np.testing.assert_allclose(regular[0], incident, rtol=2e-14, atol=0)
    assert np.linalg.norm(outgoing[-1]) < 2e-14 * np.linalg.norm(incident)
    actual = expansion_fields(
        centre,
        centre=centre[0],
        regular=regular[0],
        outgoing=outgoing[0],
        k=2 * np.pi / 550 * host,
        n=host,
        lmax=3,
    )
    expected = compute_initial_field(
        centre,
        k=2 * np.pi / 550 * host,
        n_medium=host,
        beam=source,
        polar_angles=np.empty(0),
        azimuthal_angles=np.empty(0),
    )
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=3e-14, atol=3e-14)


def test_thin_absorbing_interfaces_preserve_tangential_e_and_h():
    particle = LayeredSphere((0, 0, 0), (50, 50.5, 80), (1.5, 2.4 + 0.2j, 1.7))
    radii, indices = spherical_material_profile(particle, 1.33 + 0.08j)
    rng = np.random.default_rng(315)
    incident = rng.normal(size=n_modes(8)) + 1j * rng.normal(size=n_modes(8))
    regular, outgoing = spherical_response_coefficients(
        wavelength=550, radii=radii, indices=indices, lmax=8, incident=incident
    )
    directions = np.array([[1, 2, 3], [2, -3, 1], [-2, 1, -3]], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for boundary, radius in enumerate(radii):
        sides = [
            expansion_fields(
                radius * directions,
                centre=np.zeros(3),
                regular=regular[region],
                outgoing=outgoing[region],
                k=2 * np.pi / 550 * indices[region],
                n=indices[region],
                lmax=8,
            )
            for region in (boundary, boundary + 1)
        ]
        for inner, outer in zip(*sides, strict=True):
            residual = np.linalg.norm(np.cross(directions, inner - outer))
            assert residual / max(np.linalg.norm(inner), np.linalg.norm(outer)) < 2e-12


@pytest.mark.reference
@pytest.mark.parametrize(
    "particle,host,reaction,emission",
    [
        (
            Sphere((0, 0, 0), 80, 1.7 + 0.15j),
            1.0,
            -0.5099078332831605 + 0.437554423287635j,
            0.485228587436979 + 0.04176721488535621j,
        ),
        (
            LayeredSphere((0, 0, 0), (50, 80), (1.5, 1.7 + 0.15j)),
            1.0,
            -0.17896314007090514 + 0.10003119905083056j,
            0.5766771232449084 + 0.10159549988957534j,
        ),
        (
            LayeredSphere((0, 0, 0), (50, 80), (1.6 + 0.05j, 1.9 + 0.12j)),
            1.33 + 0.08j,
            -0.05144180040121768 - 0.23032222494507004j,
            0.8039792947296216 + 0.2128952260740263j,
        ),
    ],
    ids=["sphere", "layered", "absorbing-host"],
)
def test_centered_emission_and_reaction_match_external_transfer_factors(
    particle, host, reaction, emission
):
    # STRATIFY t_mat (Octave 10.2): electric l=1 ratios T12/T22 and 1/T22,
    # normalized to the local outgoing source coefficient at 550 nm.
    source = DipoleSource(
        550, medium_n=host, position=particle.position, dipole_moment=(1, 0.3j, -0.2)
    )
    response = prepare_embedded_dipole(particle, source, lmax=1)
    local = response.local_source.outgoing_coeffs()[0]
    np.testing.assert_allclose(response.regular[0], reaction * local, rtol=3e-14, atol=0)
    np.testing.assert_allclose(response.exterior_coefficients, emission * local, rtol=3e-14, atol=0)
