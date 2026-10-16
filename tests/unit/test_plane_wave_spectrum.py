from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.plane_wave_spectrum import PlaneWaveSpectrum

pytestmark = pytest.mark.api_contract


def _spectrum(*, alpha_shift: float = 0.0) -> PlaneWaveSpectrum:
    alpha = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False) + alpha_shift
    beta = np.linspace(0.0, np.pi, 9)
    agrid = alpha[:, None]
    bgrid = beta[None, :]
    shape = (alpha.size, beta.size)
    return PlaneWaveSpectrum(
        alpha,
        beta,
        np.sin(bgrid) * np.cos(agrid),
        np.sin(bgrid) * np.sin(agrid),
        np.broadcast_to(np.cos(bgrid), shape),
        np.ones(shape, dtype=np.complex64),
        np.zeros(shape, dtype=np.complex64),
    )


def test_plane_wave_spectrum_rejects_component_shape_mismatch() -> None:
    spectrum = _spectrum()
    with pytest.raises(ValueError, match=r"coeff_tm.*shape"):
        PlaneWaveSpectrum(
            spectrum.alpha,
            spectrum.beta,
            spectrum.kx,
            spectrum.ky,
            spectrum.kz,
            spectrum.coeff_te,
            spectrum.coeff_tm[:, :-1],
        )


def test_plane_wave_spectrum_combination_requires_one_shared_grid() -> None:
    with pytest.raises(ValueError, match="different `alpha` grids"):
        _spectrum().linear_combination(
            _spectrum(alpha_shift=0.01),
            weight_self=1.0,
            weight_other=1.0,
            dtype=np.complex64,
        )


def test_empty_spectrum_components_are_independent_and_repr_is_compact() -> None:
    spectrum = PlaneWaveSpectrum.empty(np.complex64)
    assert spectrum.shape == (0, 0)
    assert spectrum.coeff_te is not spectrum.coeff_tm
    assert repr(spectrum) == "PlaneWaveSpectrum(shape=(0, 0), coeff_dtype=dtype('complex64'))"


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
@pytest.mark.parametrize("host", [1.33, 1.33 + 0.5j])
def test_homogeneous_wavevector_validation_is_independent_of_length_units(scale, host):
    unit = _spectrum()
    k = host * scale
    spectrum = PlaneWaveSpectrum(
        unit.alpha,
        unit.beta,
        k * unit.kx,
        k * unit.ky,
        k * unit.kz,
        unit.coeff_te,
        unit.coeff_tm,
    )
    spectrum.require_homogeneous_wavevectors(k)
    assert np.iscomplexobj(spectrum.kx) == np.iscomplexobj(k)
    for component in (spectrum.kx, spectrum.ky, spectrum.kz):
        original = component[0, 0]
        component[0, 0] += k * 1e-6
        with pytest.raises(ValueError, match="inconsistent with `k`"):
            spectrum.require_homogeneous_wavevectors(k)
        component[0, 0] = original
