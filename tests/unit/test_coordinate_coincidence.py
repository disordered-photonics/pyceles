from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyceles.core.geometry_bounds import coincident_point_mask
from pyceles.core.sources import DipoleSource
from pyceles.postprocessing.nearfield import initial
from pyceles.postprocessing.nearfield.slice import interpolate_center_pixels


def test_coincidence_is_absolute_componentwise_and_closed_at_boundary() -> None:
    tolerance = 2.0**-20
    points = np.asarray([[0, 0, 0], [tolerance, tolerance, 0], [2 * tolerance, 0, 0]])
    centers = np.zeros((2, 3))  # Duplicate centers do not require a neighbor list.
    np.testing.assert_array_equal(
        coincident_point_mask(points, centers, atol=tolerance), [True, True, False]
    )
    shift = np.full(3, 2.0**30)
    np.testing.assert_array_equal(
        coincident_point_mask(shift[None, :] + [[0, 0, 0], [128, 128, 128]], shift[None, :]),
        [True, False],
    )
    assert coincident_point_mask(np.empty((0, 3)), centers).shape == (0,)
    np.testing.assert_array_equal(coincident_point_mask(points, np.empty((0, 3))), False)


@pytest.mark.parametrize("tolerance", (-1, np.inf, np.nan))
def test_coincidence_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="atol"):
        coincident_point_mask(np.zeros((1, 3)), np.zeros((1, 3)), atol=tolerance)


def test_dipole_projection_does_not_confuse_large_coordinates_with_zero_separation() -> None:
    source = DipoleSource(wavelength=550.0, radial_lut_dr=0.5)
    receivers = np.asarray([[128, 256, 128], [-256, 128, 128]], dtype=float)
    shift = np.full(3, 2.0**30)
    shifted_source = replace(source, position=tuple(shift))
    expected = source.incident_coeffs(receivers, lmax=1)
    actual = shifted_source.incident_coeffs(receivers + shift, lmax=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-20)
    with pytest.raises(ValueError, match="coincide"):
        shifted_source.incident_coeffs(shift[None, :], lmax=1)


def test_initial_local_field_masks_only_true_centers_and_reuses_owned_results(monkeypatch) -> None:
    shift = np.full(3, 2.0**30)
    source = DipoleSource(wavelength=550.0, position=tuple(shift))
    points = shift[None, :] + np.asarray([[0, 0, 0], [128, 128, 128]])
    electric = np.ones((2, 3), dtype=np.complex128)
    magnetic = np.full((2, 3), 2, dtype=np.complex128)

    def evaluate(_points, _positions, _coeffs, **kwargs):
        # Keep the evaluator on its direct path: active_mask can allocate
        # additional point/output copies on the reference backend.
        assert kwargs.get("active_mask") is None
        return electric, magnetic

    monkeypatch.setattr(initial, "compute_scattered_field", evaluate)
    e, h = initial.compute_initial_field(
        points,
        k=2 * np.pi / 550,
        n_medium=1.0,
        beam=source,
        polar_angles=np.empty(0),
        azimuthal_angles=np.empty(0),
        show_progress=False,
    )
    assert e is electric
    assert h is magnetic
    assert np.all(np.isnan(e[0])) and np.all(np.isnan(h[0]))
    np.testing.assert_array_equal(e[1], 1)
    np.testing.assert_array_equal(h[1], 2)


@pytest.mark.parametrize("offset", ((0, 0, 128), (1, 0, 0), (0, 1, 0), (0, 0, 0)))
def test_slice_interpolates_only_actual_center_pixels(offset: tuple[int, int, int]) -> None:
    origin = float(2**30)
    u, v = np.meshgrid(origin + np.asarray([0, 128, 256]), origin + np.asarray([0, 128, 256]))
    position = np.asarray([[origin + 128, origin + 128, origin]]) + offset
    e = np.ones((3, 3, 3), dtype=np.complex128)
    h = e.copy()
    e[1, 1] = h[1, 1] = 9
    interpolate_center_pixels(
        run=SimpleNamespace(positions=position),
        axis_0_grid=u,
        axis_1_grid=v,
        field_maps={"field": (e, h)},
        plane="z",
        plane_value=origin,
    )
    expected = 1 if offset == (0, 0, 0) else 9
    np.testing.assert_array_equal(e[1, 1], expected)
    np.testing.assert_array_equal(h[1, 1], expected)
