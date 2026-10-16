import numpy as np
import pytest

from pyceles.core.angular import periodic_azimuthal_weights, trapezoidal_weights
from pyceles.core.conversions import pwp_to_svwf_regular
from pyceles.core.fields import FocusedLaguerreGaussianBeam, GaussianBeam, PlaneWave
from pyceles.core.plane_wave_spectrum import PlaneWaveSpectrum
from pyceles.postprocessing.nearfield import compute_initial_field


@pytest.mark.reference
def test_gaussian_initial_field_focus_amplitude_against_smuthi_style_case():
    """SMUTHI-style Gaussian focus sanity check (homogeneous medium)."""
    wavelength = 532.0
    amplitude = 1.0
    focal = np.array([[-100.0, 100.0, 200.0]])

    beam = GaussianBeam(
        wavelength=wavelength,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.2 * np.pi,
        beam_width=4000.0,
        focal_point=tuple(focal[0]),
        amplitude=amplitude,
    )
    k = 2 * np.pi / wavelength

    # Increased angular discretization to keep the focus-amplitude error <= 1e-3.
    polar = np.linspace(0.0, np.pi, 1400)
    azimuthal = np.linspace(0.0, 2 * np.pi, 1400, endpoint=False)

    E, _ = compute_initial_field(
        focal,
        k=k,
        n_medium=1.0 + 0j,
        beam=beam,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
        batch_size=1,
        show_progress=False,
    )
    normE = np.linalg.norm(E[0])
    assert abs(normE - amplitude) < 1e-3


def test_gaussian_beam_tilted_incidence_runs():
    beam = GaussianBeam(
        wavelength=532.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.3,
        azimuthal_angle=0.0,
        beam_width=1000.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    E, H = compute_initial_field(
        np.array([[0.0, 0.0, 0.0]]),
        k=2 * np.pi / 532.0,
        n_medium=1.0 + 0j,
        beam=beam,
        polar_angles=np.linspace(0.0, np.pi, 181),
        azimuthal_angles=np.linspace(0.0, 2 * np.pi, 121, endpoint=False),
        show_progress=False,
    )
    assert E.shape == (1, 3)
    assert H.shape == (1, 3)


def test_gaussian_beam_tilted_fast_vs_general_initial_field_agree():
    beam = GaussianBeam(
        wavelength=532.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.35,
        azimuthal_angle=0.9,
        beam_width=1500.0,
        focal_point=(20.0, -15.0, 10.0),
        amplitude=1.0,
    )
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 50.0, -20.0],
            [-80.0, 25.0, 30.0],
            [15.0, -45.0, 75.0],
        ],
        dtype=float,
    )
    polar = np.linspace(0.0, np.pi, 481)
    azimuthal = np.linspace(0.0, 2 * np.pi, 181, endpoint=False)

    E_fast, H_fast = compute_initial_field(
        pts,
        k=2 * np.pi / 532.0,
        n_medium=1.0 + 0j,
        beam=beam,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
        force_general_initial_field=False,
        show_progress=False,
    )
    E_gen, H_gen = compute_initial_field(
        pts,
        k=2 * np.pi / 532.0,
        n_medium=1.0 + 0j,
        beam=beam,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
        force_general_initial_field=True,
        show_progress=False,
    )

    np.testing.assert_allclose(E_fast, E_gen, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(H_fast, H_gen, rtol=1e-3, atol=1e-3)


def test_general_initial_field_uses_periodic_azimuth_weights():
    """General alpha-beta integration must treat endpoint=False azimuth as periodic."""

    class _SingleAlphaTMSource:
        def angular_spectrum(
            self,
            *,
            k: float,
            polar_angles: np.ndarray,
            azimuthal_angles: np.ndarray,
        ) -> PlaneWaveSpectrum:
            beta = np.asarray(polar_angles, dtype=float).reshape(-1)
            alpha = np.asarray(azimuthal_angles, dtype=float).reshape(-1)
            agrid, bgrid = np.meshgrid(alpha, beta, indexing="ij")
            sb = np.sin(bgrid)
            cb = np.cos(bgrid)
            ca = np.cos(agrid)
            sa = np.sin(agrid)
            kx = float(k) * sb * ca
            ky = float(k) * sb * sa
            kz = float(k) * cb

            coeff_te = np.zeros((alpha.size, beta.size), dtype=np.complex128)
            coeff_tm = np.zeros_like(coeff_te)
            coeff_tm[0, :] = 1.0 + 0.0j

            return PlaneWaveSpectrum(alpha, beta, kx, ky, kz, coeff_te, coeff_tm)

    k = 2.0
    beta = np.array([0.6, 1.1], dtype=float)
    alpha = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)

    e, h = compute_initial_field(
        pts,
        k=k,
        n_medium=1.0 + 0j,
        beam=_SingleAlphaTMSource(),
        polar_angles=beta,
        azimuthal_angles=alpha,
        force_general_initial_field=True,
        batch_size=1,
        show_progress=False,
    )

    wa0 = float(periodic_azimuthal_weights(alpha)[0])
    wb = trapezoidal_weights(beta)
    sinb = np.sin(beta)
    cosb = np.cos(beta)
    expected_e = np.array(
        [
            wa0 * np.sum((sinb * wb) * cosb),
            0.0,
            wa0 * np.sum((sinb * wb) * (-sinb)),
        ],
        dtype=np.complex128,
    )
    expected_h = np.array(
        [
            0.0,
            wa0 * np.sum(sinb * wb),
            0.0,
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(e[0], expected_e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h[0], expected_h, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.3j])
@pytest.mark.parametrize(
    ("compute_dtype", "accum_dtype"),
    [(np.complex64, np.complex64), (np.complex64, np.complex128), (np.complex128, np.complex128)],
)
@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_sampled_homogeneous_ray_matches_analytic_plane_wave(
    host, compute_dtype, accum_dtype, backend, request
):
    """A quadrature delta ray must give the same E/H and SVWFs as a plane wave."""
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    alpha = np.array([0.2, 1.1, 2.8, 4.6])
    beta = np.array([0.3, 0.9, 1.7])
    source = PlaneWave(
        wavelength=600,
        medium_n=host,
        polar_angle=beta[1],
        azimuthal_angle=alpha[1],
        polarization=(0.7 + 0.2j, -0.3 + 0.8j),
        amplitude=0.3 + 0.9j,
        focal_point=(20, -30, 15),
    )
    k = 2 * np.pi / source.wavelength * host
    kx = k * np.sin(beta[None, :]) * np.cos(alpha[:, None])
    ky = k * np.sin(beta[None, :]) * np.sin(alpha[:, None])
    kz = np.broadcast_to(k * np.cos(beta), kx.shape)
    reference_phase = np.exp(-1j * np.dot([kx[1, 1], ky[1, 1], kz[1, 1]], source.focal_point))
    measure = periodic_azimuthal_weights(alpha)[1] * trapezoidal_weights(beta)[1] * np.sin(beta[1])
    te = np.zeros(kx.shape, dtype=np.complex128)
    tm = np.zeros_like(te)
    a_te, a_tm = source.jones_coefficients()
    te[1, 1] = source.amplitude * a_te * reference_phase / measure
    tm[1, 1] = source.amplitude * a_tm * reference_phase / measure
    spectrum = PlaneWaveSpectrum(alpha, beta, kx, ky, kz, te, tm)

    class SampledRay:
        def angular_spectrum(self, **kwargs):
            return spectrum

    points = np.array([[0, 0, 0], [120, -75, 30], [-70, 90, -40]])
    options = dict(
        k=k,
        n_medium=host,
        polar_angles=beta,
        azimuthal_angles=alpha,
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
        show_progress=False,
    )
    expected = compute_initial_field(points, beam=source, **options)
    # The returned spectrum owns its quadrature, even when a source resamples
    # the requested grid. Both field and coefficient projection must use it.
    actual = compute_initial_field(
        points,
        beam=SampledRay(),
        backend=backend,
        **{**options, "azimuthal_angles": np.array([0, np.pi])},
    )
    tolerance = 3e-6 if compute_dtype == np.complex64 else 3e-13
    for got, ref in zip(actual, expected, strict=True):
        assert got.dtype == accum_dtype
        np.testing.assert_allclose(got, ref, rtol=tolerance, atol=tolerance)
    coefficients = pwp_to_svwf_regular(points, 3, k=k, spectrum=spectrum, dtype=compute_dtype)
    np.testing.assert_allclose(
        coefficients,
        source.incident_coeffs(points, 3, dtype=compute_dtype),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_plane_wave_finite_reference_attenuation_and_coordinate_translation(dtype):
    """Local amplitude decays along propagation and is invariant under origin shifts."""
    host = 1.33 + 0.5j
    k = 2 * np.pi / 600 * host
    source = PlaneWave(
        wavelength=600,
        medium_n=host,
        amplitude=0.3 + 0.9j,
        polarization=(1, 0.5j),
        focal_point=(3, -2, 5),
    )
    distances = np.array([-30, 0, 45])
    points = np.array(source.focal_point) + np.column_stack((np.zeros((3, 2)), distances))
    options = dict(
        k=k,
        n_medium=host,
        polar_angles=np.empty(0),
        azimuthal_angles=np.empty(0),
        accum_dtype=dtype,
        show_progress=False,
    )
    e, h = compute_initial_field(points, beam=source, **options)
    # At normal incidence TE=y, TM=x, and Z0*H=n*(z cross E).
    at_reference = source.amplitude * np.array([0.5j, 1, 0])
    tolerance = 3e-7 if dtype == np.complex64 else 3e-15
    np.testing.assert_allclose(
        e,
        np.exp(1j * k * distances)[:, None] * at_reference,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(h, host * np.cross([0, 0, 1], e), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        np.linalg.norm(e, axis=1) / np.linalg.norm(e[1]),
        np.exp(-k.imag * distances),
        rtol=tolerance,
        atol=tolerance,
    )
    offset = np.array([2**40, -(2**40), 2**40])
    translated = PlaneWave(
        wavelength=source.wavelength,
        medium_n=host,
        amplitude=source.amplitude,
        polarization=source.polarization,
        focal_point=tuple(np.array(source.focal_point) + offset),
    )
    e_shifted, h_shifted = compute_initial_field(points + offset, beam=translated, **options)
    np.testing.assert_array_equal(e_shifted, e)
    np.testing.assert_array_equal(h_shifted, h)
    np.testing.assert_array_equal(
        translated.incident_coeffs(points + offset, 3, dtype=dtype),
        source.incident_coeffs(points, 3, dtype=dtype),
    )


def test_focused_laguerre_focal_plane_mirror_symmetry():
    beam = FocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=0,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1000.0,
        focal_length=900.0,
        numerical_aperture=0.8,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    pts = np.array(
        [
            [130.0, 0.0, 0.0],
            [-130.0, 0.0, 0.0],
            [0.0, 130.0, 0.0],
            [0.0, -130.0, 0.0],
        ],
        dtype=float,
    )
    k = 2.0 * np.pi / beam.wavelength * np.real(beam.medium_n)
    polar = np.linspace(0.0, np.pi, 301)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    E, _ = compute_initial_field(
        pts,
        k=k,
        n_medium=beam.medium_n,
        beam=beam,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
        force_general_initial_field=True,
        show_progress=False,
    )
    i0 = float(np.sum(np.abs(E[0]) ** 2))
    i1 = float(np.sum(np.abs(E[1]) ** 2))
    i2 = float(np.sum(np.abs(E[2]) ** 2))
    i3 = float(np.sum(np.abs(E[3]) ** 2))
    np.testing.assert_allclose(i0, i1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(i2, i3, rtol=1e-12, atol=1e-12)


def test_focused_laguerre_longitudinal_component_is_center_suppressed():
    beam = FocusedLaguerreGaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        radial_order_p=0,
        azimuthal_order_l=0,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=900.0,
        focal_length=850.0,
        numerical_aperture=0.82,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 150.0, 0.0],
            [0.0, -150.0, 0.0],
        ],
        dtype=float,
    )
    k = 2.0 * np.pi / beam.wavelength * np.real(beam.medium_n)
    polar = np.linspace(0.0, np.pi, 321)
    azimuthal = np.linspace(0.0, 2.0 * np.pi, 241, endpoint=False)
    E, _ = compute_initial_field(
        pts,
        k=k,
        n_medium=beam.medium_n,
        beam=beam,
        polar_angles=polar,
        azimuthal_angles=azimuthal,
        force_general_initial_field=True,
        show_progress=False,
    )
    ez_center = float(np.abs(E[0, 2]))
    ex_center = float(np.abs(E[0, 0]))
    ez_off = float(0.5 * (np.abs(E[1, 2]) + np.abs(E[2, 2])))
    assert ez_center <= 0.05 * max(ex_center, 1e-12)
    assert ez_off > ez_center
