from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import numpy.typing as npt
import pytest
from _source_contract_helper import assert_source_compliance

from pyceles.core.indexing import n_modes
from pyceles.core.sources import (
    BesselBeam,
    DipoleCollection,
    DipoleSource,
    GaussianBeam,
    PlaneWave,
    SLMSource,
    Source,
    source_has_finite_incident_power,
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

    def with_polarization(self, polarization: str) -> "_ExplicitInfinitePowerSource":
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
    assert_source_compliance(
        factory(),
        expect_angular_spectrum=expect_angular_spectrum,
        expect_finite_incident_power=expect_finite_power,
        expect_jones_polarization=expect_jones,
    )


def test_finite_power_policy_honors_explicit_source_capability_contract():
    alpha = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False)
    beta = np.linspace(0.0, np.pi, 17)
    pwp = _make_dummy_pwp(alpha, beta)
    source = _ExplicitInfinitePowerSource()

    assert source_has_finite_incident_power(cast(Source, source)) is False
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
