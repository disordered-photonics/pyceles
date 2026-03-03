import numpy as np

from pyceles.core.fields import PlaneWave, incident_coeffs_planewave, transformation_coefficients
from pyceles.core.indexing import n_modes, n_scalar, scalar_index
from pyceles.core.spherical import spherical_functions_trigon
from pyceles.postprocessing.nearfield import compute_initial_field


def _incident_coeffs_planewave_reference(
    positions: np.ndarray, lmax: int, source: PlaneWave
) -> np.ndarray:
    pos = np.asarray(positions, dtype=float)
    Ns = pos.shape[0]
    Nm = n_modes(lmax)
    Nscl = n_scalar(lmax)
    k = 2 * np.pi / float(source.wavelength) * float(np.real(source.medium_n))
    beta = float(source.polar_angle)
    alpha = float(source.azimuthal_angle)
    sb = np.sin(beta)
    cb = np.cos(beta)
    PI, TAU = spherical_functions_trigon(np.asarray(cb), np.asarray(sb), lmax, xp=np)
    a_te, a_tm = source.jones_coefficients()
    pol = 1 if abs(a_tm) < 1e-15 and abs(a_te) > 0.0 else 2
    fp = np.asarray(source.focal_point, dtype=float).reshape(3)
    rel = pos - fp
    kvec = k * np.array([sb * np.cos(alpha), sb * np.sin(alpha), cb], dtype=float)
    eikr = np.exp(1j * (rel @ kvec))

    aI = np.zeros((Ns, Nm), dtype=np.complex128)
    for m in range(-lmax, lmax + 1):
        for tau in (1, 2):
            for l in range(max(1, abs(m)), lmax + 1):
                idx = (tau - 1) * Nscl + scalar_index(l, m)
                aI[:, idx] = (
                    4.0
                    * float(source.amplitude)
                    * np.exp(-1j * m * alpha)
                    * eikr
                    * transformation_coefficients(PI, TAU, tau, l, m, pol, dagger=True)
                )
    return aI


def test_incident_coeffs_planewave_matches_celes_formula():
    source = PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.31,
        azimuthal_angle=0.7,
        focal_point=(10.0, -20.0, 30.0),
        amplitude=1.2,
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [50.0, -10.0, 20.0],
            [-30.0, 40.0, -80.0],
        ],
        dtype=float,
    )
    lmax = 4

    got = incident_coeffs_planewave(positions, lmax, source)
    ref = _incident_coeffs_planewave_reference(positions, lmax, source)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_compute_initial_field_plane_wave_direct_branch():
    source = PlaneWave(
        wavelength=632.8,
        medium_n=1.33 + 0j,
        polarization="TM",
        polar_angle=0.2,
        azimuthal_angle=-0.3,
        focal_point=(5.0, -3.0, 10.0),
        amplitude=0.7,
    )
    k = 2 * np.pi / source.wavelength * float(np.real(source.medium_n))
    n_medium = source.medium_n
    pts = np.array([[0.0, 0.0, 0.0], [10.0, -5.0, 4.0], [-7.0, 2.0, 12.0]], dtype=float)

    E, H = compute_initial_field(
        pts,
        k=k,
        n_medium=n_medium,
        beam=source,
        polar_angles=np.linspace(0.0, np.pi, 5),
        azimuthal_angles=np.linspace(0.0, 2 * np.pi, 6, endpoint=False),
        show_progress=False,
    )

    alpha = float(source.azimuthal_angle)
    beta = float(source.polar_angle)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    khat = np.array([sb * ca, sb * sa, cb], dtype=float)
    kvec = k * khat
    ehat = np.array([cb * ca, cb * sa, -sb], dtype=float)
    hhat = np.cross(khat, ehat) * complex(n_medium)
    phase = np.exp(1j * ((pts - np.asarray(source.focal_point)[None, :]) @ kvec))
    E_ref = complex(source.amplitude) * phase[:, None] * ehat[None, :]
    H_ref = complex(source.amplitude) * phase[:, None] * hhat[None, :]

    np.testing.assert_allclose(E, E_ref, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(H, H_ref, rtol=1e-13, atol=1e-13)
