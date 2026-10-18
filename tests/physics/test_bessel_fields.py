"""Finite-centre Bessel fields checked against cylindrical-wave integrals."""

import numpy as np
import pytest
from scipy.special import jv

from pyceles.core.sources import BesselBeam
from pyceles.postprocessing.nearfield import compute_initial_field


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.01j, 1.33 + 0.5j])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_finite_center_bessel_fields_match_cylindrical_integral(host, polarization):
    beam = BesselBeam(
        600,
        medium_n=host,
        cone_angle=0.4,
        polarization=polarization,
        center=(10, -20, 30),
        amplitude=0.7 - 0.2j,
    )
    relative = np.array([[20, 10, 30], [-15, 25, 50]], dtype=float)
    k = 2 * np.pi / 600 * host
    actual = compute_initial_field(
        relative + beam.center,
        k=k,
        n_medium=host,
        beam=beam,
        polar_angles=np.array([0, 0.4, np.pi / 2, np.pi]),
        azimuthal_angles=np.arange(32) * np.pi / 16,
        show_progress=False,
    )
    radius = np.linalg.norm(relative[:, :2], axis=1)
    er = np.column_stack((relative[:, :2] / radius[:, None], np.zeros(len(radius))))
    ep = np.column_stack((-er[:, 1], er[:, 0], np.zeros(len(radius))))
    q = k * np.sin(0.4) * radius
    phase = 2 * np.pi * beam.amplitude * np.exp(1j * k * np.cos(0.4) * relative[:, 2])
    azimuthal = 1j * jv(1, q)[:, None] * ep
    meridional = np.cos(0.4) * 1j * jv(1, q)[:, None] * er
    meridional[:, 2] = -np.sin(0.4) * jv(0, q)
    expected_e, expected_h = (
        (azimuthal, -host * meridional) if polarization == "TE" else (meridional, host * azimuthal)
    )
    for field, expected in zip(actual, (expected_e, expected_h), strict=True):
        np.testing.assert_allclose(field, phase[:, None] * expected, rtol=2e-13, atol=2e-13)
