import numpy as np

from pyceles.core.fields import GaussianBeam
from pyceles.postprocessing.farfield import compute_far_field_patterns


def test_compute_far_field_patterns_for_gaussian_beam_includes_total():
    positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    coeffs = np.zeros((1, 6), dtype=np.complex128)
    beam = GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=2000.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    out = compute_far_field_patterns(
        positions,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=1,
        polar_angles=np.linspace(0.0, np.pi, 33),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False),
        source=beam,
        show_progress=False,
    )
    assert out.initial_te is not None
    assert out.initial_tm is not None
    assert out.total_te is not None
    assert out.total_tm is not None
    np.testing.assert_allclose(out.scattered_te["coeff"], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out.scattered_tm["coeff"], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out.total_te["coeff"], out.initial_te["coeff"])
    np.testing.assert_allclose(out.total_tm["coeff"], out.initial_tm["coeff"])
