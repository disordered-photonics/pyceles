"""Symmetry-enforced end-to-end scattering checks.

These tests distinguish three different physical statements:

- global translation changes far-field phase but not integrated cross sections;
- spatial inversion relates opposite illumination for centrosymmetric particles;
- Lorentz reciprocity transposes the complex bistatic scattering matrix after
  accounting for the TE/TM basis signs under direction reversal;
- losslessness makes total scattering equal for exact time-reversed channels.

The reciprocity statement is tested at amplitude level. The time-reversal
statement is tested at cross-section level: reciprocity does not imply equal
total cross sections for opposite illumination of an arbitrary asymmetric
cluster unless the illumination channels are correctly paired by time reversal
and the scatterer is lossless.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.sources import PolarizationInput
from pyceles.postprocessing.farfield import scattered_field_plane_wave_pattern

_WAVELENGTH = 550.0
_N_MEDIUM = 1.0 + 0.0j
_DEFAULT_POLAR = np.linspace(0.0, np.pi, 81, endpoint=True)
_DEFAULT_AZIMUTHAL = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
_CROSS_SECTION_FIELDS = (
    "extinction",
    "scattering",
    "local_absorption",
    "absorption_by_difference",
    "closure_error",
)


def _source(
    *,
    polar_angle: float = 0.63,
    azimuthal_angle: float = 0.41,
    polarization: PolarizationInput = (1.0 + 0.2j, 0.35 - 0.1j),
) -> pcl.PlaneWave:
    return pcl.PlaneWave(
        wavelength=_WAVELENGTH,
        medium_n=_N_MEDIUM,
        polarization=polarization,
        polar_angle=float(polar_angle),
        azimuthal_angle=float(azimuthal_angle),
        amplitude=1.0,
    )


def _run(
    particles: list[pcl.Particle],
    source: pcl.PlaneWave,
    *,
    lmax: int,
    polar_angles: np.ndarray = _DEFAULT_POLAR,
    azimuthal_angles: np.ndarray = _DEFAULT_AZIMUTHAL,
) -> pcl.SimulationResult:
    config = pcl.SimulationConfig(
        wavelength=_WAVELENGTH,
        n_medium=_N_MEDIUM,
        lmax=int(lmax),
        source=source,
        polar_angles=np.asarray(polar_angles, dtype=float),
        azimuthal_angles=np.asarray(azimuthal_angles, dtype=float),
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )
    return pcl.Simulation(config, particles=particles).run()


def _cross_sections(run: pcl.SimulationResult) -> pcl.CrossSectionBalance:
    if run.cross_sections is None:
        raise AssertionError("Plane-wave simulations must expose cross sections.")
    return run.cross_sections


def _assert_cross_sections_close(
    first: pcl.CrossSectionBalance,
    second: pcl.CrossSectionBalance,
    *,
    rtol: float,
    atol: float,
) -> None:
    for field in _CROSS_SECTION_FIELDS:
        np.testing.assert_allclose(
            getattr(second, field),
            getattr(first, field),
            rtol=rtol,
            atol=atol,
            err_msg=f"Cross-section symmetry failed for {field}.",
        )


_PARTICLE_CASES = (
    pytest.param(
        pcl.Sphere(
            position=(20.0, -30.0, 40.0),
            radius=85.0,
            refractive_index=1.55 + 0.02j,
        ),
        4,
        id="sphere",
    ),
    pytest.param(
        pcl.PECSphere(position=(20.0, -30.0, 40.0), radius=85.0),
        4,
        id="pec-sphere",
    ),
    pytest.param(
        pcl.LayeredSphere(
            position=(20.0, -30.0, 40.0),
            layer_radii=(45.0, 90.0),
            layer_refractive_indices=(1.80 + 0.01j, 1.35 + 0.03j),
        ),
        4,
        id="layered-sphere",
    ),
    pytest.param(
        pcl.Spheroid(
            position=(20.0, -30.0, 40.0),
            equatorial_radius=70.0,
            polar_radius=110.0,
            refractive_index=1.55 + 0.02j,
            euler_angles=(0.3, 0.5, 0.2),
        ),
        5,
        id="spheroid",
    ),
)


def _mixed_particle_cluster() -> list[pcl.Particle]:
    return [
        pcl.Sphere(
            position=(-210.0, -80.0, 25.0),
            radius=55.0,
            refractive_index=1.50 + 0.02j,
        ),
        pcl.PECSphere(position=(155.0, -95.0, 80.0), radius=50.0),
        pcl.LayeredSphere(
            position=(55.0, 170.0, -75.0),
            layer_radii=(25.0, 48.0),
            layer_refractive_indices=(1.80 + 0.01j, 1.30 + 0.04j),
        ),
        pcl.Spheroid(
            position=(-85.0, 135.0, 155.0),
            equatorial_radius=40.0,
            polar_radius=60.0,
            refractive_index=1.60 + 0.015j,
            euler_angles=(0.25, 0.55, 0.35),
        ),
    ]


def _centrosymmetric_mixed_cluster() -> list[pcl.Particle]:
    """Return a mixed cluster invariant under inversion through the origin."""
    particles = [
        pcl.Sphere(
            position=(230.0, 20.0, 40.0),
            radius=45.0,
            refractive_index=1.50 + 0.02j,
        ),
        pcl.PECSphere(position=(20.0, 240.0, -80.0), radius=42.0),
        pcl.LayeredSphere(
            position=(-220.0, -180.0, 110.0),
            layer_radii=(24.0, 45.0),
            layer_refractive_indices=(1.80 + 0.01j, 1.30 + 0.04j),
        ),
        pcl.Spheroid(
            position=(130.0, -210.0, -150.0),
            equatorial_radius=34.0,
            polar_radius=50.0,
            refractive_index=1.60 + 0.015j,
            euler_angles=(0.25, 0.55, 0.35),
        ),
    ]
    mirrored = [
        replace(
            particle,
            position=tuple(-np.asarray(particle.position, dtype=float)),
        )
        for particle in particles
    ]
    return particles + mirrored


@pytest.mark.parametrize(("particle", "lmax"), _PARTICLE_CASES)
def test_plane_wave_cross_sections_are_invariant_to_particle_translation(
    particle: pcl.Particle,
    lmax: int,
) -> None:
    source = _source()
    shift = np.array([137.0, -211.0, 83.0], dtype=float)
    shifted = replace(
        particle,
        position=tuple(np.asarray(particle.position, dtype=float) + shift),
    )

    reference = _cross_sections(_run([particle], source, lmax=lmax))
    translated = _cross_sections(_run([shifted], source, lmax=lmax))

    _assert_cross_sections_close(reference, translated, rtol=5e-12, atol=1e-9)


def test_mixed_cluster_farfield_obeys_global_translation_phase_covariance() -> None:
    source = _source(polar_angle=0.67, azimuthal_angle=0.43)
    particles = _mixed_particle_cluster()
    shift = np.array([137.0, -211.0, 83.0], dtype=float)
    shifted_particles = [
        replace(
            particle,
            position=tuple(np.asarray(particle.position, dtype=float) + shift),
        )
        for particle in particles
    ]
    polar_angles = np.linspace(0.0, np.pi, 41, endpoint=True)
    azimuthal_angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)

    reference = _run(
        particles,
        source,
        lmax=4,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
    )
    translated = _run(
        shifted_particles,
        source,
        lmax=4,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
    )

    sin_beta = np.sin(source.polar_angle)
    incident_k = reference.k * np.array(
        [
            sin_beta * np.cos(source.azimuthal_angle),
            sin_beta * np.sin(source.azimuthal_angle),
            np.cos(source.polar_angle),
        ],
        dtype=float,
    )
    pattern = reference.farfield.scattered_te
    outgoing_phase = (
        np.asarray(pattern["kx"]) * shift[0]
        + np.asarray(pattern["ky"]) * shift[1]
        + np.asarray(pattern["kz"]) * shift[2]
    )
    phase = np.exp(1j * (float(incident_k @ shift) - outgoing_phase))

    for field in ("scattered_te", "scattered_tm"):
        expected = phase * np.asarray(getattr(reference.farfield, field)["coeff"])
        actual = np.asarray(getattr(translated.farfield, field)["coeff"])
        np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)

    _assert_cross_sections_close(
        _cross_sections(reference),
        _cross_sections(translated),
        rtol=5e-12,
        atol=1e-8,
    )


def test_mixed_cluster_observables_are_invariant_to_particle_permutation() -> None:
    source = _source(polar_angle=0.67, azimuthal_angle=0.43)
    particles = _mixed_particle_cluster()
    order = np.array([2, 0, 3, 1], dtype=int)
    polar_angles = np.linspace(0.0, np.pi, 41, endpoint=True)
    azimuthal_angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)

    reference = _run(
        particles,
        source,
        lmax=4,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
    )
    permuted = _run(
        [particles[int(index)] for index in order],
        source,
        lmax=4,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
    )

    inverse_order = np.argsort(order)
    np.testing.assert_allclose(
        permuted.coeffs[inverse_order],
        reference.coeffs,
        rtol=5e-12,
        atol=5e-14,
    )
    for field in ("scattered_te", "scattered_tm"):
        np.testing.assert_allclose(
            getattr(permuted.farfield, field)["coeff"],
            getattr(reference.farfield, field)["coeff"],
            rtol=5e-12,
            atol=5e-14,
        )
    _assert_cross_sections_close(
        _cross_sections(reference),
        _cross_sections(permuted),
        rtol=5e-12,
        atol=2e-8,
    )


_RADIAL_PARTICLE_CASES = (
    pytest.param(
        pcl.Sphere(
            position=(0.0, 0.0, 0.0),
            radius=85.0,
            refractive_index=1.55 + 0.02j,
        ),
        id="sphere",
    ),
    pytest.param(
        pcl.PECSphere(position=(0.0, 0.0, 0.0), radius=85.0),
        id="pec-sphere",
    ),
    pytest.param(
        pcl.LayeredSphere(
            position=(0.0, 0.0, 0.0),
            layer_radii=(45.0, 90.0),
            layer_refractive_indices=(1.80 + 0.01j, 1.35 + 0.03j),
        ),
        id="layered-sphere",
    ),
)


@pytest.mark.parametrize("particle", _RADIAL_PARTICLE_CASES)
def test_radially_symmetric_particles_are_incidence_and_polarization_isotropic(
    particle: pcl.Particle,
) -> None:
    polar_angles = np.linspace(0.0, np.pi, 161, endpoint=True)
    azimuthal_angles = np.linspace(0.0, 2.0 * np.pi, 192, endpoint=False)
    first = _cross_sections(
        _run(
            [particle],
            _source(polar_angle=0.22, azimuthal_angle=0.14, polarization="TE"),
            lmax=4,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
        )
    )
    second = _cross_sections(
        _run(
            [particle],
            _source(polar_angle=2.13, azimuthal_angle=1.72, polarization="TM"),
            lmax=4,
            polar_angles=polar_angles,
            azimuthal_angles=azimuthal_angles,
        )
    )

    np.testing.assert_allclose(second.extinction, first.extinction, rtol=5e-12, atol=1e-9)
    np.testing.assert_allclose(
        second.local_absorption,
        first.local_absorption,
        rtol=5e-12,
        atol=1e-9,
    )
    # The continuous scattering integral is exactly isotropic. The fixed
    # latitude/longitude quadrature is not rotation-covariant, so this one
    # comparison intentionally carries the finite-grid error budget.
    np.testing.assert_allclose(second.scattering, first.scattering, rtol=5e-5, atol=0.0)


@pytest.mark.parametrize(("particle", "lmax"), _PARTICLE_CASES)
def test_centrosymmetric_single_particles_obey_opposite_incidence_parity(
    particle: pcl.Particle,
    lmax: int,
) -> None:
    polarization = (1.0 + 0.2j, 0.35 - 0.1j)
    source = _source(polarization=polarization)
    # At the antipodal propagation direction, e_TE changes sign while e_TM
    # does not. Spatial inversion of the electric field therefore keeps the
    # TE Jones weight and negates the TM weight.
    reversed_source = _source(
        polar_angle=np.pi - source.polar_angle,
        azimuthal_angle=source.azimuthal_angle + np.pi,
        polarization=(polarization[0], -polarization[1]),
    )

    forward = _cross_sections(_run([particle], source, lmax=lmax))
    backward = _cross_sections(_run([particle], reversed_source, lmax=lmax))

    _assert_cross_sections_close(forward, backward, rtol=5e-12, atol=1e-9)


def test_centrosymmetric_mixed_cluster_obeys_opposite_incidence_parity() -> None:
    """Opposite-incidence equality requires object symmetry, not reciprocity alone."""
    polarization = (1.0 + 0.2j, 0.35 - 0.1j)
    source = _source(polarization=polarization)
    reversed_source = _source(
        polar_angle=np.pi - source.polar_angle,
        azimuthal_angle=source.azimuthal_angle + np.pi,
        polarization=(polarization[0], -polarization[1]),
    )

    forward = _cross_sections(_run(_centrosymmetric_mixed_cluster(), source, lmax=3))
    backward = _cross_sections(_run(_centrosymmetric_mixed_cluster(), reversed_source, lmax=3))

    _assert_cross_sections_close(forward, backward, rtol=5e-12, atol=1e-9)


def _scattered_intensity(run: pcl.SimulationResult) -> np.ndarray:
    return np.asarray(
        np.abs(run.farfield.scattered_te["coeff"]) ** 2
        + np.abs(run.farfield.scattered_tm["coeff"]) ** 2,
        dtype=float,
    )


def test_spheroid_axial_symmetry_rotates_the_farfield_without_changing_cross_sections() -> None:
    n_azimuthal = 72
    shift = 13
    delta = 2.0 * np.pi * shift / n_azimuthal
    particle = pcl.Spheroid(
        position=(45.0, -30.0, 20.0),
        equatorial_radius=70.0,
        polar_radius=110.0,
        refractive_index=1.55 + 0.02j,
        euler_angles=(0.0, 0.0, 0.0),
    )
    polar_angles = np.linspace(0.0, np.pi, 61, endpoint=True)
    azimuthal_angles = np.linspace(0.0, 2.0 * np.pi, n_azimuthal, endpoint=False)
    source = _source(
        polar_angle=0.7,
        azimuthal_angle=0.23,
        polarization=(1.0 + 0.1j, 0.25 - 0.2j),
    )
    rotated_source = _source(
        polar_angle=source.polar_angle,
        azimuthal_angle=source.azimuthal_angle + delta,
        polarization=source.polarization,
    )

    reference = _run(
        [particle],
        source,
        lmax=4,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
    )
    rotated = _run(
        [particle],
        rotated_source,
        lmax=4,
        polar_angles=polar_angles,
        azimuthal_angles=azimuthal_angles,
    )

    np.testing.assert_allclose(
        _scattered_intensity(rotated),
        np.roll(_scattered_intensity(reference), shift=shift, axis=0),
        rtol=5e-12,
        atol=1e-16,
    )
    _assert_cross_sections_close(
        _cross_sections(reference),
        _cross_sections(rotated),
        rtol=5e-12,
        atol=1e-9,
    )


def _antipodal_direction(polar_angle: float, azimuthal_angle: float) -> tuple[float, float]:
    return np.pi - float(polar_angle), (float(azimuthal_angle) + np.pi) % (2.0 * np.pi)


def _bistatic_scattering_matrix(
    particles: list[pcl.Particle],
    *,
    lmax: int,
    incident_direction: tuple[float, float],
    observation_direction: tuple[float, float],
) -> np.ndarray:
    incident_polar, incident_azimuthal = incident_direction
    observation_polar, observation_azimuthal = observation_direction
    sources = {
        "te": _source(
            polar_angle=incident_polar,
            azimuthal_angle=incident_azimuthal,
            polarization="TE",
        ),
        "tm": _source(
            polar_angle=incident_polar,
            azimuthal_angle=incident_azimuthal,
            polarization="TM",
        ),
    }
    config = pcl.SimulationConfig(
        wavelength=_WAVELENGTH,
        n_medium=_N_MEDIUM,
        lmax=int(lmax),
        source=sources["te"],
        solver_method="direct",
        verbose=False,
        compute_dtype="complex128",
        accum_dtype="complex128",
    )
    simulation = pcl.Simulation(config, particles=particles)
    solved = simulation.solve_sources(sources)

    matrix = np.empty((2, 2), dtype=np.complex128)
    for column, label in enumerate(("te", "tm")):
        scattered_te, scattered_tm = scattered_field_plane_wave_pattern(
            positions=simulation.positions,
            coeffs=solved.coeffs[label],
            k=solved.k,
            lmax=lmax,
            polar_angles=np.array([observation_polar], dtype=float),
            azimuthal_angles=np.array([observation_azimuthal], dtype=float),
            dtype=np.complex128,
        )
        matrix[:, column] = (
            complex(np.asarray(scattered_te["coeff"])[0, 0]),
            complex(np.asarray(scattered_tm["coeff"])[0, 0]),
        )
    return matrix


def _assert_lorentz_reciprocity(
    particles: list[pcl.Particle],
    *,
    lmax: int,
    rtol: float,
    atol: float,
) -> None:
    incident = (0.71, 0.38)
    observation = (1.26, 2.13)
    forward = _bistatic_scattering_matrix(
        particles,
        lmax=lmax,
        incident_direction=incident,
        observation_direction=observation,
    )
    reverse = _bistatic_scattering_matrix(
        particles,
        lmax=lmax,
        incident_direction=_antipodal_direction(*observation),
        observation_direction=_antipodal_direction(*incident),
    )

    # Under k -> -k, the spherical TE basis vector changes sign and the TM
    # basis vector does not. Lorentz reciprocity therefore reads
    # F(k_out, k_in) = D F(-k_in, -k_out)^T D in this basis.
    direction_reversal = np.diag(np.array([-1.0, 1.0], dtype=float))
    expected = direction_reversal @ reverse.T @ direction_reversal
    np.testing.assert_allclose(forward, expected, rtol=rtol, atol=atol)


def _asymmetric_radial_particle_cluster(*, lossless: bool) -> list[pcl.Particle]:
    return [
        pcl.Sphere(
            position=(-180.0, -70.0, 20.0),
            radius=60.0,
            refractive_index=1.50 + (0.0j if lossless else 0.02j),
        ),
        pcl.PECSphere(position=(130.0, -90.0, 75.0), radius=55.0),
        pcl.LayeredSphere(
            position=(40.0, 155.0, -65.0),
            layer_radii=(28.0, 50.0),
            layer_refractive_indices=(
                1.80 + (0.0j if lossless else 0.01j),
                1.30 + (0.0j if lossless else 0.04j),
            ),
        ),
    ]


def test_asymmetric_radial_particle_cluster_obeys_lorentz_reciprocity() -> None:
    particles = _asymmetric_radial_particle_cluster(lossless=False)
    _assert_lorentz_reciprocity(particles, lmax=3, rtol=2e-12, atol=2e-14)


def test_lossless_asymmetric_cluster_obeys_time_reversed_cross_section_symmetry() -> None:
    """A lossless reciprocal cluster scatters equally in time-reversed channels.

    Opposite propagation alone is not the symmetry operation. Time reversal
    also conjugates the electric-field phasor. In the spherical TE/TM basis,
    ``e_TE`` changes sign under direction reversal while ``e_TM`` does not, so
    an equivalent Jones pair is ``(conj(a_te), -conj(a_tm))`` up to a global
    phase.
    """
    polarization = (1.0 + 0.2j, 0.35 - 0.1j)
    source = _source(
        polar_angle=0.71,
        azimuthal_angle=0.38,
        polarization=polarization,
    )
    reversed_polar, reversed_azimuthal = _antipodal_direction(
        source.polar_angle,
        source.azimuthal_angle,
    )
    time_reversed_source = _source(
        polar_angle=reversed_polar,
        azimuthal_angle=reversed_azimuthal,
        polarization=(np.conj(polarization[0]), -np.conj(polarization[1])),
    )
    same_jones_opposite_source = _source(
        polar_angle=reversed_polar,
        azimuthal_angle=reversed_azimuthal,
        polarization=polarization,
    )
    particles = _asymmetric_radial_particle_cluster(lossless=True)
    polar_angles = np.linspace(0.0, np.pi, 121, endpoint=True)
    azimuthal_angles = np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False)

    def run_cross_sections(illumination: pcl.PlaneWave) -> pcl.CrossSectionBalance:
        return _cross_sections(
            _run(
                particles,
                illumination,
                lmax=3,
                polar_angles=polar_angles,
                azimuthal_angles=azimuthal_angles,
            )
        )

    forward = run_cross_sections(source)
    time_reversed = run_cross_sections(time_reversed_source)
    same_jones_opposite = run_cross_sections(same_jones_opposite_source)

    # Reciprocity preserves forward-scattering extinction for the paired
    # channels. Losslessness then makes their integrated scattering equal too;
    # the looser scattering tolerances are solely the fixed angular quadrature.
    np.testing.assert_allclose(
        time_reversed.extinction,
        forward.extinction,
        rtol=5e-12,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        time_reversed.scattering,
        forward.scattering,
        rtol=1e-5,
        atol=0.0,
    )
    for cross_sections in (forward, time_reversed):
        np.testing.assert_allclose(
            cross_sections.local_absorption,
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            cross_sections.scattering,
            cross_sections.extinction,
            rtol=5e-5,
            atol=0.0,
        )

    # Guard against accidentally testing a centrosymmetric or isotropic case:
    # the same Jones tuple from the opposite direction is not the time-reversed
    # channel and produces a measurably different response for this cluster.
    relative_same_jones_difference = abs(same_jones_opposite.extinction - forward.extinction) / abs(
        forward.extinction
    )
    assert relative_same_jones_difference > 1e-2


def test_rotated_spheroid_obeys_lorentz_reciprocity_at_ebcm_accuracy() -> None:
    particle = pcl.Spheroid(
        position=(40.0, -20.0, 30.0),
        equatorial_radius=45.0,
        polar_radius=65.0,
        refractive_index=1.60 + 0.015j,
        euler_angles=(0.25, 0.55, 0.35),
    )
    _assert_lorentz_reciprocity([particle], lmax=5, rtol=1e-5, atol=2e-9)
