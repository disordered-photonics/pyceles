from dataclasses import dataclass

import numpy as np

from pyceles.core.fields import GaussianBeam, LocalExpansionSource
from pyceles.core.indexing import n_modes
from pyceles.postprocessing.farfield import compute_far_field_patterns


@dataclass(frozen=True)
class _ToyLocalExpansionSource:
    wavelength: float = 550.0
    medium_n: complex = 1.0 + 0.0j
    amplitude: float = 1.0

    def has_finite_incident_power(self) -> bool:
        return False

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype=np.complex128,
    ) -> np.ndarray:
        del polar_angles, azimuthal_angles
        ns = np.asarray(positions, dtype=float).reshape(-1, 3).shape[0]
        return np.zeros((ns, n_modes(int(lmax))), dtype=np.dtype(dtype))

    def source_positions(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0]], dtype=float)

    def outgoing_coeffs(self, lmax: int, *, dtype=np.complex128) -> np.ndarray:
        out = np.zeros((1, n_modes(int(lmax))), dtype=np.dtype(dtype))
        out[0, 0] = 1.0 + 0.0j
        return out


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


def test_compute_far_field_patterns_accepts_local_expansion_source_protocol():
    positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    coeffs = np.zeros((1, 6), dtype=np.complex128)
    source = _ToyLocalExpansionSource()

    assert isinstance(source, LocalExpansionSource)

    out = compute_far_field_patterns(
        positions,
        coeffs,
        k=2.0 * np.pi / 550.0,
        lmax=1,
        polar_angles=np.linspace(0.0, np.pi, 33),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False),
        source=source,
        show_progress=False,
    )

    assert out.initial_te is not None
    assert out.initial_tm is not None
    assert out.total_te is not None
    assert out.total_tm is not None
