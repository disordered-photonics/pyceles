"""Outgoing angular amplitudes and their finite reference origin."""

import numpy as np
import pytest

from pyceles.core.sources import DipoleCollection
from pyceles.postprocessing.farfield import (
    compute_far_field_patterns,
    pwp_power_flux,
    scattered_field_plane_wave_pattern,
)


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.01j, 1.33 + 0.5j])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_outgoing_amplitudes_match_transverse_green_tensor_and_origin_shift(
    host, dtype, backend, request
):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    source = DipoleCollection(
        600,
        medium_n=host,
        positions=np.array([[20, -30, 10], [-40, 20, 60]]),
        dipole_moments=np.array([[1, 0.3j, -0.2], [-0.1j, 0.5, 0.2j]]),
        amplitude=0.7 - 0.2j,
    )
    k0 = 2 * np.pi / source.wavelength
    k = k0 * host
    alpha = np.array([0.1, 0.7, 3.1, 5.2])
    beta = np.array([0, 0.4, 1.2, 2.9, np.pi])
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")
    directions = np.stack((np.sin(bb) * np.cos(aa), np.sin(bb) * np.sin(aa), np.cos(bb)), axis=-1)
    e_phi = np.stack((-np.sin(aa), np.cos(aa), np.zeros_like(aa)), axis=-1)
    e_theta = np.stack((np.cos(bb) * np.cos(aa), np.cos(bb) * np.sin(aa), -np.sin(bb)), axis=-1)
    expected = np.zeros_like(directions, dtype=complex)
    for position, moment in zip(source.positions, source.dipole_moments_array(), strict=True):
        transverse = moment - directions * (directions @ moment)[..., None]
        expected += (
            k
            * k0**2
            / (4 * np.pi)
            * transverse
            * np.exp(-1j * k * (directions @ position))[..., None]
        )

    patterns = compute_far_field_patterns(
        np.zeros((1, 3)),
        np.zeros((1, 6)),
        k=k,
        lmax=1,
        polar_angles=beta,
        azimuthal_angles=alpha,
        source=source,
        backend=backend,
        dtype=dtype,
        show_progress=False,
    )
    initial = patterns.initial
    assert initial is not None and patterns.total is not None
    initial.require_homogeneous_wavevectors(k)
    np.testing.assert_array_equal(initial.coeff_te, patterns.total.coeff_te)
    np.testing.assert_array_equal(initial.coeff_tm, patterns.total.coeff_tm)
    actual = (
        -2
        * np.pi
        * 1j
        * (initial.coeff_te[..., None] * e_phi + initial.coeff_tm[..., None] * e_theta)
    )
    tolerance = 2e-6 if dtype == np.complex64 else 3e-14
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance

    shift = np.array([45, -20, 30])
    shifted = scattered_field_plane_wave_pattern(
        source.positions - shift,
        source.outgoing_coeffs(),
        k,
        source.outgoing_lmax,
        beta,
        alpha,
        backend=backend,
        dtype=dtype,
    )
    factor = np.exp(1j * k * (directions @ shift))
    for value, reference in (
        (shifted.coeff_te, initial.coeff_te),
        (shifted.coeff_tm, initial.coeff_tm),
    ):
        assert (
            np.linalg.norm(value - reference * factor) / np.linalg.norm(reference * factor)
            < tolerance
        )
    if complex(host).imag:
        # Neither an absorbing spectrum nor a complex normalization wavenumber
        # can be used in the lossless hemispheric power formula.
        for normalization in (k.real, k):
            with pytest.raises(ValueError, match="nonabsorbing host"):
                pwp_power_flux(initial, k0=k0, k_medium=normalization, direction="forward")


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_zero_outgoing_rows_do_not_overflow_the_angular_phase(backend, request):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    spectrum = scattered_field_plane_wave_pattern(
        np.array([[1e9, 1e9, 1e9]]),
        np.zeros((1, 6)),
        1.33 + 0.5j,
        1,
        np.array([0, np.pi / 2, np.pi]),
        np.array([0, np.pi]),
        backend=backend,
    )
    np.testing.assert_array_equal(spectrum.coeff_te, 0)
    np.testing.assert_array_equal(spectrum.coeff_tm, 0)
