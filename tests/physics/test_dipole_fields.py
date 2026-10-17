"""Point-dipole normalization against the homogeneous dyadic Green function."""

from dataclasses import replace

import numpy as np
import pytest

from pyceles.core.indexing import index_vswf
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.postprocessing.nearfield.initial import compute_initial_field


def _green_fields(points, source):
    """Electric Green tensor and its curl, in E and Z0*H units."""
    k0 = 2 * np.pi / source.wavelength
    k = k0 * source.medium_n
    e = np.zeros((len(points), 3), dtype=complex)
    h = np.zeros_like(e)
    for centre, moment in zip(source.positions, source.dipole_moments_array(), strict=True):
        displacement = points - centre
        r = np.linalg.norm(displacement, axis=1)
        direction = displacement / r[:, None]
        longitudinal = direction * (direction @ moment)[:, None]
        phase = np.exp(1j * k * r) / (4 * np.pi * r)
        e += (k0**2 * phase)[:, None] * (
            moment
            - longitudinal
            + (1 / (k * r) ** 2 - 1j / (k * r))[:, None] * (3 * longitudinal - moment)
        )
        h += (k0 * k * phase * (1 + 1j / (k * r)))[:, None] * np.cross(direction, moment)
    return e, h


@pytest.mark.parametrize("host", [1.33, 1.33 + 0.01j, 1.33 + 0.5j])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_dipole_fields_and_regular_translation_match_green_tensor(host, dtype, backend, request):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    source = DipoleCollection(
        600,
        medium_n=host,
        positions=np.array([[20, -30, 10], [20, -30, 16]]),
        dipole_moments=np.array([[1, 0.3j, -0.2], [-0.1j, 0.5, 0.2j]]),
        amplitude=0.7 - 0.2j,
        radial_lut_dr=0.25,
    )
    # Integral separations lie on LUT knots, isolating normalization from interpolation.
    points = source.positions[0] + np.array([[4, 0, 3], [0, 0, -2], [0, 0, 106], [0, 4, 3]])
    expected_e, expected_h = _green_fields(points, source)
    actual = compute_initial_field(
        points,
        k=2 * np.pi / 600 * host,
        n_medium=host,
        beam=source,
        polar_angles=np.empty(0),
        azimuthal_angles=np.empty(0),
        compute_dtype=dtype,
        backend=backend,
        show_progress=False,
    )
    tolerance = 2e-6 if dtype == np.complex64 else 2e-13
    for value, reference in zip(actual, (expected_e, expected_h), strict=True):
        assert (
            np.max(np.linalg.norm(value - reference, axis=1) / np.linalg.norm(reference, axis=1))
            < tolerance
        )

    # The regular l=1 origin limits recover E and Z0*H. Higher receiving orders
    # must not change this sector or the source's own degree-one representation.
    for lmax in (1, 3):
        translated = source.incident_coeffs(points, lmax, dtype=dtype)
        assert translated.dtype == dtype
        for tau, field in ((2, expected_e), (1, 1j / host * expected_h)):
            reference = np.column_stack(
                (
                    np.sqrt(3) * (field[:, 0] + 1j * field[:, 1]),
                    np.sqrt(6) * field[:, 2],
                    np.sqrt(3) * (field[:, 0] - 1j * field[:, 1]),
                )
            )
            columns = [index_vswf(1, m, tau, lmax) for m in (-1, 0, 1)]
            error = np.linalg.norm(translated[:, columns] - reference, axis=1) / np.linalg.norm(
                reference, axis=1
            )
            assert np.max(error) < tolerance

    singles = [
        DipoleSource(
            600,
            medium_n=host,
            position=tuple(pos),
            dipole_moment=tuple(moment),
            amplitude=source.amplitude,
        )
        for pos, moment in zip(source.positions, source.dipole_moments, strict=True)
    ]
    np.testing.assert_array_equal(
        source.outgoing_coeffs(), np.vstack([s.outgoing_coeffs() for s in singles])
    )
    if complex(host).imag:
        emitters: list[DipoleSource | DipoleCollection] = [source, *singles]
        for emitter in emitters:
            with pytest.raises(ValueError, match=r"self-power.*nonabsorbing host"):
                emitter.dissipated_power_homogeneous_background()


def test_zero_emitters_do_not_set_singularities_or_translation_extent():
    source = DipoleCollection(
        600,
        medium_n=1.33 + 0.5j,
        positions=np.array([[0, 0, 0], [1, 0, 0], [1e9, 1e9, 1e9]]),
        dipole_moments=np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
        radial_lut_dr=0.25,
    )
    expected = replace(
        source, positions=source.positions[1:2], dipole_moments=source.dipole_moments[1:2]
    )
    points = np.array([[0, 0, 0], [2, 1, 0]])
    np.testing.assert_array_equal(
        source.incident_coeffs(points, 2), expected.incident_coeffs(points, 2)
    )
    with pytest.raises(ValueError, match="coincide"):
        source.incident_coeffs(source.positions[1:2], 2)
