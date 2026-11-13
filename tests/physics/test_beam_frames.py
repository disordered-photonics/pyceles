"""Beam polarization and mode phases under physical rotations."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyceles.core.sources import (
    BesselBeam,
    CartesianPolarizedBesselBeam,
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
    PlaneWave,
)
from pyceles.postprocessing.nearfield import compute_initial_field


def _rotation(theta, phi):
    return (Rotation.from_rotvec([0, 0, phi]) * Rotation.from_rotvec([0, theta, 0])).as_matrix()


def _cartesian_spectrum(spectrum):
    alpha, beta = np.meshgrid(spectrum.alpha, spectrum.beta, indexing="ij")
    ephi = np.stack([-np.sin(alpha), np.cos(alpha), np.zeros_like(alpha)], axis=-1)
    etheta = np.stack(
        [np.cos(beta) * np.cos(alpha), np.cos(beta) * np.sin(alpha), -np.sin(beta)], axis=-1
    )
    return spectrum.coeff_te[..., None] * ephi + spectrum.coeff_tm[..., None] * etheta


@pytest.mark.parametrize(
    "theta,phi", [(0, 0.4), (2e-12, 0.4), (0.2, 0.7853), (0.2, 0.7855), (np.pi, 0.4)]
)
@pytest.mark.parametrize(
    "beam",
    [
        GaussianBeam(550, beam_width=1000, polarization=(0.7 + 0.2j, -0.3j)),
        LaguerreGaussianBeam(
            550,
            radial_order_p=1,
            azimuthal_order_l=2,
            polarization=(0.7 + 0.2j, -0.3j),
            azimuthal_phase=0.3,
        ),
        FocusedLaguerreGaussianBeam(
            550, azimuthal_order_l=1, polarization=(0.7 + 0.2j, -0.3j), azimuthal_phase=0.3
        ),
        CartesianPolarizedFocusedLaguerreGaussianBeam(
            550, azimuthal_order_l=2, global_polarization=(1, 0.3j, 0.2), azimuthal_phase=0.3
        ),
    ],
    ids=["gaussian", "laguerre", "focused", "cartesian-focused"],
)
def test_beam_spectrum_rotates_polarization_and_vortex_phase(beam, theta, phi):
    rotation = _rotation(theta, phi)
    tilted = replace(beam, polar_angle=theta, azimuthal_angle=phi)
    if isinstance(beam, CartesianPolarizedFocusedLaguerreGaussianBeam):
        tilted = replace(tilted, global_polarization=tuple(rotation @ beam.global_polarization))
    k = 2 * np.pi / beam.wavelength
    # Compare the same physical rays, rather than two differently discretized
    # angular integrals. Nonzero mode orders check the complex phase reference.
    for alpha, beta in ((0.2, 0.07), (1.1, 0.13), (2.3, 0.3)):
        ray = rotation @ np.array(
            [np.sin(beta) * np.cos(alpha), np.sin(beta) * np.sin(alpha), np.cos(beta)]
        )
        lab_alpha, lab_beta = np.arctan2(ray[1], ray[0]), np.arctan2(np.hypot(*ray[:2]), ray[2])
        local = beam.angular_spectrum(
            k=k, polar_angles=np.array([beta]), azimuthal_angles=np.array([alpha])
        )
        lab = tilted.angular_spectrum(
            k=k, polar_angles=np.array([lab_beta]), azimuthal_angles=np.array([lab_alpha])
        )
        np.testing.assert_allclose(
            _cartesian_spectrum(lab),
            _cartesian_spectrum(local) @ rotation.T,
            rtol=1e-11,
            atol=1e-13,
        )


@pytest.mark.parametrize("source_type", [GaussianBeam, LaguerreGaussianBeam])
@pytest.mark.parametrize("polarization", ["TE", "TM", (0.7 + 0.2j, -0.3 + 0.5j)])
def test_collimated_beam_has_plane_wave_polarization_and_continuous_fields(
    source_type, polarization
):
    polar = np.linspace(0, np.pi, 1001)
    azimuthal = np.arange(128) * (2 * np.pi / 128)
    angles = [
        (0, 0.4),
        (5e-13, 0.4),
        (2e-12, 0.4),
        (0.2, 0.7853),
        (0.2, 0.7855),
        (0.6, 1.1),
        (np.pi - 2e-12, 0.4),
        (np.pi, 0.4),
    ]
    fields = []
    for theta, phi in angles:
        beam = source_type(
            550, beam_width=2000, polarization=polarization, polar_angle=theta, azimuthal_angle=phi
        )
        actual = compute_initial_field(
            np.zeros((1, 3)),
            k=2 * np.pi / 550,
            n_medium=1,
            beam=beam,
            polar_angles=polar,
            azimuthal_angles=azimuthal,
            show_progress=False,
        )
        plane = PlaneWave(550, polarization=polarization, polar_angle=theta, azimuthal_angle=phi)
        expected = compute_initial_field(
            np.zeros((1, 3)),
            k=2 * np.pi / 550,
            n_medium=1,
            beam=plane,
            polar_angles=polar,
            azimuthal_angles=azimuthal,
            show_progress=False,
        )
        for field, reference in zip(actual, expected, strict=True):
            # Normalize by positive real norms: a wrong complex phase still fails.
            np.testing.assert_allclose(
                field / np.linalg.norm(field),
                reference / np.linalg.norm(reference),
                rtol=1e-5,
                atol=1e-5,
            )
        fields.append(actual)
    for left, right in ((1, 2), (6, 7)):
        for before, after in zip(fields[left], fields[right], strict=True):
            np.testing.assert_allclose(before, after, rtol=5e-11, atol=5e-11)
    for before, after in zip(fields[3], fields[4], strict=True):
        assert np.linalg.norm(after - before) / np.linalg.norm(before) < 1e-3


@pytest.mark.parametrize("source_type", [BesselBeam, CartesianPolarizedBesselBeam])
def test_bessel_vortex_phase_uses_incident_plane_reference(source_type):
    beam = source_type(550, polar_angle=0.2, azimuthal_angle=0.4, cone_angle=0.4)
    k = 2 * np.pi / 550
    options = dict(
        k=k,
        polar_angles=np.linspace(0, np.pi, 181),
        azimuthal_angles=np.arange(64) * (2 * np.pi / 64),
    )
    plain = beam.angular_spectrum(**options)
    vortex = replace(beam, order_m=2).angular_spectrum(**options)
    rotation = _rotation(beam.polar_angle, beam.azimuthal_angle)
    rays = np.stack([plain.kx, plain.ky, plain.kz], axis=-1) / k
    local = rays @ rotation
    phase = np.exp(2j * np.arctan2(local[..., 1].real, local[..., 0].real))
    np.testing.assert_allclose(vortex.coeff_te, phase * plain.coeff_te, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(vortex.coeff_tm, phase * plain.coeff_tm, rtol=1e-13, atol=1e-12)
