"""Radial-table precision, physical intervals and translation endpoints."""

import numpy as np
import pytest

from pyceles.core.translation import (
    RadialLUT,
    translation_ab5_table,
    translation_block,
    translation_block_rect,
)
from pyceles.postprocessing.nearfield import scattered
from pyceles.postprocessing.nearfield.scattered import NearFieldRadialLUT, compute_scattered_field


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


@pytest.mark.parametrize("k", [1.3, 1.3 + 0.5j])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_scattered_field_uses_only_active_physical_lut_queries(k, backend, request):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    options = dict(k=k, lmax=1, backend=backend, show_progress=False, compute_dtype=np.complex64)
    positions = np.zeros((1, 3))
    coeffs = np.ones((1, 6), dtype=np.complex64)
    lut = NearFieldRadialLUT(lmax=1, k=k, r_max=2, dr=0.125)
    points = np.array([[0, 0, 0], [1, 0, 0]])
    for invalid_points in (points, np.array([[0.05, 0, 0]]), np.array([[3, 0, 0]])):
        with pytest.raises(ValueError, match="padding or exceeds"):
            compute_scattered_field(invalid_points, positions, coeffs, lut=lut, **options)
    e, h = compute_scattered_field(
        points, positions, coeffs, lut=lut, active_mask=np.array([False, True]), **options
    )
    assert np.isfinite(e).all() and np.isfinite(h).all()
    np.testing.assert_array_equal(e[0], 0)
    np.testing.assert_array_equal(h[0], 0)
    for wrong in (
        NearFieldRadialLUT(lmax=1, k=k * 2, r_max=2, dr=0.125),
        NearFieldRadialLUT(lmax=2, k=k, r_max=2, dr=0.125),
    ):
        with pytest.raises(ValueError, match="match k and lmax"):
            compute_scattered_field(points[1:], positions, coeffs, lut=wrong, **options)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_zero_expansions_do_not_size_tables_or_create_singularities(
    dtype, backend, request, monkeypatch
):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    positions = np.array([[0, 0, 0], [-1, 0, 0], [1000, 1000, 1000]])
    coeffs = np.zeros((3, 6), dtype=dtype)
    coeffs[1] = 1
    points = np.array([[0, 0, 0], [1.2, 0.4, 0]])
    options = dict(
        k=1.3 + 0.5j,
        lmax=1,
        backend=backend,
        show_progress=False,
        compute_dtype=dtype,
        particle_distance_resolution=0.125,
    )
    expected = compute_scattered_field(points, positions[1:2], coeffs[1:2], **options)
    table_factory = scattered.NearFieldRadialLUT
    table_sizes = []

    def make_table(**kwargs):
        assert kwargs["r_max"] < 3
        table_sizes.append(kwargs["r_max"])
        return table_factory(**kwargs)

    monkeypatch.setattr(scattered, "NearFieldRadialLUT", make_table)
    for got, ref in zip(
        compute_scattered_field(points, positions, coeffs, **options), expected, strict=True
    ):
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got, ref)
    assert len(table_sizes) == 1
    e, h = compute_scattered_field(points, positions, np.zeros_like(coeffs), **options)
    np.testing.assert_array_equal(e, 0)
    np.testing.assert_array_equal(h, 0)
    assert not np.shares_memory(e, h)
    assert len(table_sizes) == 1


def test_supplied_field_table_can_cover_actual_distances_below_the_aabb_bound():
    positions = np.array([[0, 0, 0], [1, 1, 0]])
    points = np.array([[0, 1, 0], [1, 0, 0]])
    coeffs = np.ones((2, 6))
    lut = NearFieldRadialLUT(lmax=1, k=1.3, r_max=1, dr=0.125)
    e, h = compute_scattered_field(
        points, positions, coeffs, k=1.3, lmax=1, lut=lut, show_progress=False
    )
    assert np.isfinite(e).all() and np.isfinite(h).all()
