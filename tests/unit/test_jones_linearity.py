"""Jones support is exact; pure-channel shortcuts must retain complex weights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.core.polarization import pure_polarization_label
from pyceles.core.sources import (
    BesselBeam,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
    PlaneWave,
    polarization_to_jones,
)
from pyceles.postprocessing.nearfield.initial import compute_initial_field

WEIGHTS = (
    (2.0 + 3.0j, 0.0j),
    (0.0j, -0.5j),
    (1.0e-12 + 0.0j, 1.0 + 0.0j),
    (1.0 + 0.0j, 1.0e-18j),
    (1.0e-20 + 0.0j, 0.0j),
)


def _source(name: str) -> Any:
    common: dict[str, Any] = dict(wavelength=532.0, azimuthal_angle=0.37, amplitude=0.8)
    if name == "plane":
        return PlaneWave(**common, polar_angle=0.31)
    if name == "gaussian_normal":
        return GaussianBeam(**common, beam_width=700.0)
    if name == "gaussian_tilted":
        return GaussianBeam(**common, beam_width=700.0, polar_angle=0.31)
    if name == "laguerre":
        return LaguerreGaussianBeam(**common, beam_width=700.0, azimuthal_order_l=1)
    if name == "focused":
        return FocusedLaguerreGaussianBeam(**common, azimuthal_order_l=1)
    if name == "bessel":
        return BesselBeam(**common, order_m=1, cone_angle=0.3)
    raise AssertionError(name)


@pytest.mark.parametrize("weights", WEIGHTS)
def test_jones_canonicalization_preserves_all_nonzero_weights(weights) -> None:
    assert polarization_to_jones(weights) == weights
    expected = "TE" if weights[1] == 0 else "TM" if weights[0] == 0 else None
    assert pure_polarization_label(*weights) == expected


def test_jones_zero_vector_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        polarization_to_jones((0.0j, 0.0j))


@pytest.mark.parametrize(
    "name", ("plane", "gaussian_normal", "gaussian_tilted", "laguerre", "focused", "bessel")
)
@pytest.mark.parametrize("weights", WEIGHTS)
def test_incident_coefficients_retain_pure_channel_weight(name: str, weights) -> None:
    base = _source(name)
    positions = np.asarray([[0.0, 0.0, 0.0], [30.0, -20.0, 50.0]])
    kwargs: dict[str, Any] = dict(
        polar_angles=np.linspace(0.0, np.pi, 121),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False),
    )
    te = base.with_polarization("TE").incident_coeffs(positions, 2, **kwargs)
    tm = base.with_polarization("TM").incident_coeffs(positions, 2, **kwargs)
    actual = base.with_polarization(weights).incident_coeffs(positions, 2, **kwargs)
    expected = weights[0] * te + weights[1] * tm
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-14 * scale)


@pytest.mark.parametrize("name", ("gaussian_normal", "laguerre", "focused", "bessel"))
@pytest.mark.parametrize("weights", WEIGHTS)
def test_angular_spectrum_retain_pure_channel_weight(name: str, weights) -> None:
    base = _source(name)
    kwargs: dict[str, Any] = dict(
        k=2.0 * np.pi / base.wavelength,
        polar_angles=np.linspace(0.0, np.pi, 121),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False),
    )
    te = base.with_polarization("TE").angular_spectrum(**kwargs)
    tm = base.with_polarization("TM").angular_spectrum(**kwargs)
    actual = base.with_polarization(weights).angular_spectrum(**kwargs)
    for attr in ("coeff_te", "coeff_tm"):
        expected = weights[0] * getattr(te, attr) + weights[1] * getattr(tm, attr)
        scale = float(np.max(np.abs(expected)))
        np.testing.assert_allclose(
            getattr(actual, attr), expected, rtol=2.0e-13, atol=2.0e-14 * scale
        )


@pytest.mark.parametrize("backend", ("numpy", pytest.param("cupy", marks=pytest.mark.gpu)))
@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("polar_angle", (0.0, 0.31))
def test_gaussian_initial_field_is_jones_linear(backend: str, weights, polar_angle: float) -> None:
    base = GaussianBeam(
        wavelength=532.0, beam_width=700.0, azimuthal_angle=0.37, polar_angle=polar_angle
    )
    points = np.asarray([[0.0, 0.0, 0.0], [70.0, -40.0, 80.0]])
    kwargs: dict[str, Any] = dict(
        k=2.0 * np.pi / base.wavelength,
        n_medium=1.0 + 0.0j,
        polar_angles=np.linspace(0.0, np.pi, 121),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False),
        backend=backend,
        show_progress=False,
    )
    te = compute_initial_field(points, beam=base.with_polarization("TE"), **kwargs)
    tm = compute_initial_field(points, beam=base.with_polarization("TM"), **kwargs)
    actual = compute_initial_field(points, beam=base.with_polarization(weights), **kwargs)
    for result, te_field, tm_field in zip(actual, te, tm, strict=True):
        expected = weights[0] * te_field + weights[1] * tm_field
        scale = float(np.max(np.abs(expected)))
        assert scale > 0.0
        np.testing.assert_allclose(result, expected, rtol=3.0e-13, atol=3.0e-14 * scale)
