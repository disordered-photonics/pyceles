from dataclasses import dataclass

import numpy as np
import pytest

from pyceles.core.fields import GaussianBeam, LocalExpansionSource
from pyceles.core.indexing import index_vswf, n_modes
from pyceles.postprocessing.farfield import (
    compute_far_field_patterns,
    scattered_field_plane_wave_pattern,
)
from pyceles.postprocessing.nearfield.initial import compute_initial_field
from pyceles.postprocessing.nearfield.scattered import compute_scattered_field


@dataclass(frozen=True)
class _ToyLocalExpansionSource:
    wavelength: float = 550.0
    medium_n: complex = 1.0 + 0.0j
    amplitude: float = 1.0
    outgoing_lmax: int = 2

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

    def outgoing_coeffs(self, *, dtype=np.complex128) -> np.ndarray:
        out = np.zeros((1, n_modes(self.outgoing_lmax)), dtype=np.dtype(dtype))
        out[0, index_vswf(2, 1, 2, self.outgoing_lmax)] = 1.0 + 0.3j
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
    assert out.initial is not None
    assert out.total is not None
    np.testing.assert_allclose(out.scattered.coeff_te, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out.scattered.coeff_tm, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(out.total.coeff_te, out.initial.coeff_te)
    np.testing.assert_allclose(out.total.coeff_tm, out.initial.coeff_tm)


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_local_expansion_degree_is_independent_of_particle_degree(backend, request):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    coeffs = np.zeros((1, 6), dtype=np.complex128)
    source = _ToyLocalExpansionSource()

    assert isinstance(source, LocalExpansionSource)

    k = 2.0 * np.pi / source.wavelength
    angles = dict(polar_angles=np.linspace(0, np.pi, 9), azimuthal_angles=np.arange(8) * np.pi / 4)

    out = compute_far_field_patterns(
        positions,
        coeffs,
        k=k,
        lmax=1,
        **angles,
        source=source,
        backend=backend,
        show_progress=False,
    )

    assert out.initial is not None
    assert out.total is not None
    expected = scattered_field_plane_wave_pattern(
        source.source_positions(),
        source.outgoing_coeffs(),
        k,
        source.outgoing_lmax,
        polar_angles=angles["polar_angles"],
        azimuthal_angles=angles["azimuthal_angles"],
    )
    np.testing.assert_allclose(out.initial.coeff_te, expected.coeff_te, atol=1e-15)
    np.testing.assert_allclose(out.initial.coeff_tm, expected.coeff_tm, atol=1e-15)
    assert np.linalg.norm(out.initial.coeff_te) > 0
    assert np.linalg.norm(out.initial.coeff_tm) > 0

    points = np.array([[100, -30, 50], [0, 0, 200]])
    initial = compute_initial_field(
        points,
        k=k,
        n_medium=source.medium_n,
        beam=source,
        backend=backend,
        show_progress=False,
        polar_angles=angles["polar_angles"],
        azimuthal_angles=angles["azimuthal_angles"],
    )
    expected_fields = compute_scattered_field(
        points,
        source.source_positions(),
        source.outgoing_coeffs(),
        k=k,
        lmax=source.outgoing_lmax,
        show_progress=False,
    )
    for value, reference in zip(initial, expected_fields, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-13, atol=1e-15)
