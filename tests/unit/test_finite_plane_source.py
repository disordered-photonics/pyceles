import numpy as np
import pytest
from _source_contract_helper import assert_source_compliance

from pyceles.core.fourier_plane import FourierPlaneSpectrum
from pyceles.core.sources import FinitePlaneSource, FinitePlaneSpectrumSource


def test_plane_source_owns_its_boundary():
    q, alpha = np.array([0, 0.01]), np.array([0, np.pi])
    te, tm = np.ones((2, 2), dtype=complex), np.zeros((2, 2), dtype=complex)
    source = FinitePlaneSource(600, q, alpha, te, tm, medium_n=1.33 + 0.5j)
    assert_source_compliance(
        source,
        expect_angular_spectrum=False,
        expect_finite_incident_power=True,
        expect_jones_polarization=False,
        expect_local_expansion=False,
    )
    assert isinstance(source, FinitePlaneSpectrumSource)
    expected = source.incident_coeffs(np.zeros((1, 3)), 2)
    q[:] = 0
    alpha[:] = 0
    te[:] = 0
    tm[:] = 1
    np.testing.assert_array_equal(source.incident_coeffs(np.zeros((1, 3)), 2), expected)
    for values in (
        source.transverse_wavenumbers,
        source.spectrum_azimuths,
        source.coeff_te,
        source.coeff_tm,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)


@pytest.mark.parametrize(
    "alpha",
    [np.arange(8) * np.pi / 4, np.linspace(0, 2 * np.pi, 9), np.array([0.2, 0.6, 2.4, 4.1])],
)
def test_plane_measure_spans_full_period_on_open_closed_and_nonuniform_grids(alpha):
    q = np.array([0, 0.3, 0.8])
    spectrum = FourierPlaneSpectrum(
        alpha, q, 1.33 + 0.5j, np.ones((len(alpha), len(q))), np.ones((len(alpha), len(q)))
    )
    np.testing.assert_allclose(np.sum(spectrum.weights), np.pi * q[-1] ** 2, rtol=2e-16)
    vectors = spectrum.wavevectors()
    np.testing.assert_allclose(np.sum(vectors**2, axis=-1), spectrum.k**2, rtol=3e-16)
    np.testing.assert_allclose(
        np.sum(vectors * spectrum.electric_vectors(), axis=-1), 0, atol=3e-16
    )


def test_plane_boundary_rejects_underdetermined_grazing_mode():
    with pytest.raises(ValueError, match="grazing"):
        FinitePlaneSource(
            2 * np.pi, np.array([0, 1]), np.array([0, np.pi]), np.ones((2, 2)), np.ones((2, 2))
        )


@pytest.mark.parametrize(
    "q,alpha",
    [
        ([0, 0], [0, 1]),
        ([-1, 1], [0, 1]),
        ([0, np.inf], [0, 1]),
        ([0, 0.1], [0]),
        ([0, 0.1], [1, 0]),
        ([0, 0.1], [0, 7]),
        ([0, 0.1], [0, np.nan]),
    ],
)
def test_plane_boundary_rejects_invalid_sampling(q, alpha):
    with pytest.raises(ValueError):
        FourierPlaneSpectrum(np.array(alpha), np.array(q), 1.33, np.ones((2, 2)), np.ones((2, 2)))
