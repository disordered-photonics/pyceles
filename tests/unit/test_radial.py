"""Radial-table precision, physical intervals and translation endpoints."""

import numpy as np
import pytest

from pyceles.core.translation import (
    RadialLUT,
    translation_ab5_table,
    translation_block,
    translation_block_rect,
)
from pyceles.postprocessing.nearfield.scattered import NearFieldRadialLUT


@pytest.mark.parametrize("factory", [RadialLUT, NearFieldRadialLUT])
def test_radial_storage_cast_follows_double_precision_sampling(factory):
    options = dict(lmax=3, k=1.3 + 0.5j, r_max=2.0, dr=0.03)
    single = factory(**options, dtype=np.complex64)
    double = factory(**options, dtype=np.complex128)
    np.testing.assert_array_equal(np.asarray(single.h), np.asarray(double.h).astype(np.complex64))
    assert single.k == options["k"]
    single.validate_interval(0.1, 2.0)
    for bounds in ((0.0, 1.0), (0.1, 3.0), (np.nan, 1.0), (1.0, 0.5)):
        with pytest.raises(ValueError, match="interval"):
            single.validate_interval(*bounds)


def test_finite_singular_prefix_is_outside_physical_query_interval():
    lut = RadialLUT(5, 2 * np.pi * (1.33 + 0.5j), 0.1, 1 / 16384, dtype=np.complex64)
    assert np.isfinite(lut.h).all()
    assert lut.minimum_hankel_radius > lut.dr
    with pytest.raises(ValueError, match="padding"):
        lut.validate_interval(lut.dr, 0.1)
    lut.validate_interval(lut.minimum_hankel_radius, 0.1)
    np.testing.assert_array_equal(lut.bessel_j_all_p(0), [1] + [0] * 10)


@pytest.mark.parametrize("k", [1.3, 1.3 + 0.5j])
def test_last_translation_sample_is_preserved_at_upper_endpoint(k):
    lut = RadialLUT(2, k, 2.0, 0.125)
    np.testing.assert_array_equal(lut.hankel_all_p(lut.r_grid[-1]), lut.h[:, -1])
    np.testing.assert_array_equal(lut.bessel_j_all_p(lut.r_grid[-1]), lut.j[:, -1])


@pytest.mark.parametrize("factory", [RadialLUT, NearFieldRadialLUT])
@pytest.mark.parametrize("k", [0, -1, 1 - 0.1j, complex(1, np.nan), np.inf])
def test_radial_sampling_rejects_invalid_passive_wavenumber(factory, k):
    with pytest.raises(ValueError, match="finite"):
        factory(lmax=1, k=k, r_max=1, dr=0.01)


def test_rectangular_translation_preserves_complex_k_when_expanding_table():
    k = 1.3 + 0.5j
    lut = RadialLUT(1, k, 2.0, 0.001)
    displacement = np.array([0.4, -0.3, 0.7])
    actual = translation_block_rect(2, 2, k, displacement, radial_lut=lut)
    expected = translation_block(2, k, displacement, ab5=translation_ab5_table(2))
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-10)
