import numpy as np

from pyceles.core.angular import periodic_azimuthal_weights, trapezoidal_weights
from pyceles.core.fields import FocusedLaguerreGaussianBeam, GaussianBeam
from pyceles.postprocessing.nearfield import compute_initial_field


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
        ) -> tuple[dict, dict]:
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

            pwp_base = {"alpha": alpha, "beta": beta, "kx": kx, "ky": ky, "kz": kz}
            return {**pwp_base, "coeff": coeff_te}, {**pwp_base, "coeff": coeff_tm}

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
