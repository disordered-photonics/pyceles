from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from _source_contract_helper import assert_source_compliance

from pyceles.core.indexing import n_modes
from pyceles.core.sources import (
    BesselBeam,
    CartesianPolarizedBesselBeam,
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    DipoleCollection,
    DipoleSource,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
    PlaneWave,
    SLMSource,
    Source,
)
from pyceles.postprocessing.farfield import finite_beam_power_fractions


def _make_dummy_pwp(alpha: np.ndarray, beta: np.ndarray) -> dict:
    coeff = np.zeros((alpha.size, beta.size), dtype=np.complex128)
    agrid = alpha[:, None]
    bgrid = beta[None, :]
    return {
        "alpha": alpha,
        "beta": beta,
        "kx": np.sin(bgrid) * np.cos(agrid),
        "ky": np.sin(bgrid) * np.sin(agrid),
        "kz": np.cos(bgrid) * np.ones_like(agrid),
        "coeff": coeff,
    }


@dataclass(frozen=True)
class _ExplicitInfinitePowerSource:
    wavelength: float = 550.0
    medium_n: complex = 1.0 + 0.0j
    amplitude: float = 1.0
    polarization: str = "TE"
    beam_width: float = 1000.0

    def jones_coefficients(self) -> tuple[complex, complex]:
        return 1.0 + 0.0j, 0.0 + 0.0j

    def with_polarization(self, polarization: str) -> _ExplicitInfinitePowerSource:
        return _ExplicitInfinitePowerSource(
            wavelength=self.wavelength,
            medium_n=self.medium_n,
            amplitude=self.amplitude,
            polarization=polarization,
            beam_width=self.beam_width,
        )

    def has_finite_incident_power(self) -> bool:
        return False

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        del polar_angles, azimuthal_angles
        ns = np.asarray(positions, dtype=float).reshape(-1, 3).shape[0]
        nm = n_modes(int(lmax))
        return np.zeros((ns, nm), dtype=np.dtype(dtype))


def _finite_gaussian() -> GaussianBeam:
    return GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0.0j, 0.3 - 0.1j),
        polar_angle=0.2,
        azimuthal_angle=0.4,
        beam_width=1400.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )


def _infinite_gaussian() -> GaussianBeam:
    return GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.2,
        azimuthal_angle=0.4,
        beam_width=np.inf,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )


@pytest.mark.parametrize(
    ("factory", "expect_angular_spectrum", "expect_finite_power", "expect_jones"),
    [
        (_finite_gaussian, True, True, True),
        (_infinite_gaussian, True, False, True),
        (
            lambda: SLMSource(base_source=_finite_gaussian(), modulation=1.0 + 0.0j),
            True,
            True,
            True,
        ),
        (
            lambda: SLMSource(base_source=_infinite_gaussian(), modulation=1.0 + 0.0j),
            True,
            False,
            True,
        ),
        (
            lambda: BesselBeam(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                order_m=1,
                cone_angle=0.5,
                polar_angle=0.3,
                azimuthal_angle=0.4,
                polarization=(1.0 + 0.0j, 0.2 - 0.3j),
                amplitude=1.0,
                center=(0.0, 0.0, 0.0),
                forward_only=True,
            ),
            True,
            False,
            True,
        ),
        (
            lambda: CartesianPolarizedBesselBeam(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                order_m=1,
                cone_angle=0.5,
                polar_angle=0.3,
                azimuthal_angle=0.4,
                global_polarization=(1.0 + 0.0j, 0.2 - 0.3j, 0.0 + 0.0j),
                amplitude=1.0,
                center=(0.0, 0.0, 0.0),
                forward_only=True,
            ),
            True,
            False,
            False,
        ),
        (
            lambda: LaguerreGaussianBeam(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                radial_order_p=1,
                azimuthal_order_l=2,
                polarization=(1.0 + 0.0j, 0.2 - 0.1j),
                polar_angle=0.3,
                azimuthal_angle=0.4,
                beam_width=1400.0,
                focal_point=(0.0, 0.0, 0.0),
                amplitude=1.0,
                azimuthal_phase=0.1,
            ),
            True,
            True,
            True,
        ),
        (
            lambda: FocusedLaguerreGaussianBeam(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                radial_order_p=0,
                azimuthal_order_l=1,
                polarization="TE",
                polar_angle=0.1,
                azimuthal_angle=0.4,
                beam_width=1200.0,
                focal_length=1000.0,
                numerical_aperture=0.75,
                focal_point=(0.0, 0.0, 0.0),
                amplitude=1.0,
                azimuthal_phase=0.0,
                sine_condition_apodization=True,
            ),
            True,
            True,
            True,
        ),
        (
            lambda: CartesianPolarizedFocusedLaguerreGaussianBeam(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                radial_order_p=0,
                azimuthal_order_l=1,
                polar_angle=0.1,
                azimuthal_angle=0.4,
                global_polarization=(1.0 + 0.0j, 0.0 + 0.2j, 0.0 + 0.0j),
                beam_width=1200.0,
                focal_length=1000.0,
                numerical_aperture=0.75,
                focal_point=(0.0, 0.0, 0.0),
                amplitude=1.0,
                azimuthal_phase=0.0,
                sine_condition_apodization=True,
            ),
            True,
            True,
            False,
        ),
        (
            lambda: PlaneWave(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                polarization="TE",
                polar_angle=0.2,
                azimuthal_angle=0.4,
                focal_point=(0.0, 0.0, 0.0),
                amplitude=1.0,
            ),
            False,
            False,
            True,
        ),
        (
            lambda: DipoleSource(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                dipole_moment=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
                position=(0.0, 0.0, 0.0),
            ),
            False,
            False,
            False,
        ),
        (
            lambda: DipoleCollection(
                wavelength=550.0,
                medium_n=1.0 + 0j,
                positions=np.array([[0.0, 0.0, 0.0], [150.0, 0.0, 0.0]], dtype=float),
                dipole_moments=np.array(
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    ],
                    dtype=np.complex128,
                ),
            ),
            False,
            False,
            False,
        ),
    ],
)
def test_builtin_sources_satisfy_capability_contract(
    factory: Callable[[], object],
    expect_angular_spectrum: bool,
    expect_finite_power: bool,
    expect_jones: bool,
):
    source_obj = factory()
    expect_local = isinstance(source_obj, (DipoleSource, DipoleCollection))
    assert_source_compliance(
        source_obj,
        expect_angular_spectrum=expect_angular_spectrum,
        expect_finite_incident_power=expect_finite_power,
        expect_jones_polarization=expect_jones,
        expect_local_expansion=expect_local,
    )


def test_finite_power_policy_honors_explicit_source_capability_contract():
    alpha = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    beta = np.linspace(0.0, np.pi, 17)
    pwp = _make_dummy_pwp(alpha, beta)
    source = _ExplicitInfinitePowerSource()

    assert source.has_finite_incident_power() is False
    with pytest.raises(ValueError, match="infinite-power sources"):
        finite_beam_power_fractions(
            cast(Source, source),
            pwp,
            pwp,
            pwp,
            pwp,
            k0=2.0 * np.pi / 550.0,
            k_medium=2.0 * np.pi / 550.0,
        )
