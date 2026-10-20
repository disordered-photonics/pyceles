import numpy as np
import pytest

from pyceles.core.fields import PlaneWave, incident_coeffs_planewave, transformation_coefficients
from pyceles.core.indexing import index_vswf, n_modes, n_scalar, scalar_index
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
                    * complex(source.amplitude)
                    * np.exp(-1j * m * alpha)
                    * eikr
                    * transformation_coefficients(PI, TAU, tau, l, m, pol, dagger=True)
                )
    return aI


@pytest.mark.reference
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


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_bloch_plane_wave_projection_matches_real_angle_plane_wave(direction, polarization):
    import pyceles as pcl

    wavelength = 550.0
    beta = 0.43
    alpha = -0.28
    k = 2.0 * np.pi / wavelength
    q = (k * np.sin(beta) * np.cos(alpha), k * np.sin(beta) * np.sin(alpha))
    reference_point = (11.0, -7.0, 25.0)
    angle = beta if direction == "up" else np.pi - beta
    plane_wave = PlaneWave(
        wavelength=wavelength,
        medium_n=1.0,
        polarization=polarization,
        polar_angle=angle,
        azimuthal_angle=alpha,
        focal_point=reference_point,
        amplitude=0.7 + 0.2j,
    )
    bloch_wave = pcl.BlochPlaneWave(
        wavelength=wavelength,
        medium_n=1.0,
        polarization=polarization,
        k_parallel=q,
        direction=direction,
        reference_point=reference_point,
        amplitude=0.7 + 0.2j,
    )
    positions = np.asarray([[0.0, 20.0, 80.0], [-25.0, 15.0, -40.0]])

    np.testing.assert_allclose(
        bloch_wave.incident_coeffs(positions, 3),
        plane_wave.incident_coeffs(positions, 3),
        rtol=2e-13,
        atol=2e-13,
    )


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


def test_mixed_jones_projection_preserves_small_coefficients_in_single_precision():
    beta = 0.7
    delta = 2.0**-30
    source = PlaneWave(
        wavelength=600,
        polar_angle=beta,
        polarization=(1, 1j * (np.cos(beta) + delta)),
    )
    coefficients = source.incident_coeffs(np.zeros((1, 3)), 1, dtype=np.complex64)
    # For l=m=1, tau=1, pi=sqrt(3)/2 and tau=cos(beta)*pi.
    # The two Jones terms cancel except for the prescribed delta.
    np.testing.assert_allclose(
        coefficients[0, index_vswf(1, 1, 1, 1)],
        1j * np.sqrt(3) * delta,
        rtol=3e-7,
        atol=0,
    )


@pytest.mark.reference
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_plane_wave_coefficients_and_fields_against_treams(dtype):
    """TREAMS 0.4.8.dev1 (1f5d0d6), parity basis and exp(-i*omega*t).

    Convert each incident coefficient by c_celes=c_treams/D, with
    D=-i*sqrt(pi)*(-1)^m for positive m and -i*sqrt(pi) otherwise.
    Cartesian magnetic values are Z0*H. Only selected modes and one exterior
    point are retained; the source has finite-reference complex amplitude.
    """
    source = PlaneWave(
        wavelength=600,
        medium_n=1.33 + 0.5j,
        polar_angle=2.3,
        azimuthal_angle=-0.62,
        amplitude=0.3 + 0.9j,
        focal_point=(-25, 11, -18),
    )
    positions = np.array([[-312, -108, -132], [246, -186, -48], [-114, 288, 126], [198, 162, 264]])
    coefficients = source.incident_coeffs(positions, 3, dtype=dtype)
    selected = [(0, 1, -1, 1), (0, 2, 1, 2), (1, 3, -2, 1), (2, 2, 0, 2), (3, 1, 1, 2)]
    expected = np.array(
        [
            0.029264287172171236 + 1.39753479647918j,
            -0.6217323453051632 - 1.6941218179116695j,
            0.06997343679935414 - 0.09690224569258149j,
            0j,
            0.6827233228321632 - 2.970080286389556j,
        ]
    )
    actual = [coefficients[p, index_vswf(l, m, tau, 3)] for p, l, m, tau in selected]
    tolerance = 2e-6 if dtype == np.complex64 else 2e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    e, h = compute_initial_field(
        np.array([[0, 0, 1050]]),
        k=2 * np.pi / source.wavelength * source.medium_n,
        n_medium=source.medium_n,
        beam=source,
        polar_angles=np.empty(0),
        azimuthal_angles=np.empty(0),
        accum_dtype=dtype,
        show_progress=False,
    )
    np.testing.assert_allclose(
        e[0],
        [-10.419436566162155 - 17.783133944154265j, -14.594908411255709 - 24.909524572845775j, 0j],
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        h[0],
        [
            -4.634926422385362 - 26.935621923006483j,
            3.3089157181438025 + 19.22958309080241j,
            -6.373768512856227 - 37.04080782337484j,
        ],
        rtol=tolerance,
        atol=tolerance,
    )
    assert coefficients.dtype == e.dtype == h.dtype == dtype


@pytest.mark.api_contract
@pytest.mark.parametrize(
    "kwargs",
    [
        {"medium_n": 1 - 0.01j},
        {"medium_n": complex(np.inf, 0)},
        {"medium_n": 0j},
        {"wavelength": 0},
        {"wavelength": np.nan},
        {"amplitude": complex(0, np.inf)},
        {"polar_angle": -0.1},
        {"polar_angle": np.inf},
        {"azimuthal_angle": np.nan},
        {"focal_point": (0, np.nan, 0)},
    ],
)
def test_plane_wave_rejects_invalid_finite_reference_parameters(kwargs):
    parameters = {"wavelength": 600, **kwargs}
    with pytest.raises(ValueError):
        PlaneWave(**parameters)
