from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.translation import translation_ab5_table
from pyceles.postprocessing.nearfield.periodic_interior_cupy import (
    _periodic_near_pair_batch_size_for_workspace,
    _periodic_near_pair_workspace_bytes_per_pair,
)
from pyceles.postprocessing.nearfield.periodic_projection import (
    l1_compact_projection_data,
    l1_projection_data,
    reduce_compact_structural_sums_to_l1,
    reduce_structural_sums_to_l1,
)


@pytest.mark.parametrize("lmax", [1, 2, 3, 4, 5, 8])
def test_compact_l1_projection_matches_dense_reference(lmax: int) -> None:
    rng = np.random.default_rng(1800 + lmax)
    lmax_struct, _offset, dense_kernel, _rows = l1_projection_data(lmax)
    (
        compact_lmax_struct,
        structural_order,
        degrees,
        orders,
        compact_kernel,
    ) = l1_compact_projection_data(lmax)
    assert compact_lmax_struct == lmax_struct
    assert structural_order == lmax + 1

    maximum_order = 2 * lmax_struct
    structural = rng.normal(size=(7, maximum_order + 1, 2 * maximum_order + 1)) + 1j * rng.normal(
        size=(7, maximum_order + 1, 2 * maximum_order + 1)
    )
    # Invalid (p,m) slots do not occur in an Ewald table. Zero them so the
    # reference and compact representations compare the same mathematical data.
    for degree in range(maximum_order + 1):
        for column in range(2 * maximum_order + 1):
            order_m = column - maximum_order
            if abs(order_m) > degree or degree > structural_order:
                structural[:, degree, column] = 0.0
    coeffs = rng.normal(size=n_modes(lmax)) + 1j * rng.normal(size=n_modes(lmax))

    want = reduce_structural_sums_to_l1(structural, coeffs, kernel=dense_kernel)
    compact_structural = structural[
        :,
        : structural_order + 1,
        maximum_order - structural_order : maximum_order + structural_order + 1,
    ]
    got = reduce_compact_structural_sums_to_l1(
        compact_structural,
        coeffs,
        degree_indices=degrees,
        order_indices=orders,
        kernel=compact_kernel,
    )
    np.testing.assert_allclose(got, want, rtol=5e-13, atol=5e-13)


@pytest.mark.parametrize("lmax", [2, 3, 4, 5, 8])
def test_l1_translation_has_no_support_above_triangle_limit(lmax: int) -> None:
    """The compact p<=lmax+1 cutoff follows the exact Wigner triangle rule."""
    _lmax_struct, _offset, _kernel, l1_rows = l1_projection_data(lmax)
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    unsupported = np.asarray(ab5[l1_rows, :, lmax + 2 :])
    np.testing.assert_array_equal(unsupported, np.zeros_like(unsupported))


@pytest.mark.parametrize("compute_dtype", [np.complex64, np.complex128])
def test_periodic_near_pair_batch_planner_respects_budget(compute_dtype: Any) -> None:
    per_pair = _periodic_near_pair_workspace_bytes_per_pair(
        lmax=4,
        compute_dtype=np.dtype(compute_dtype),
    )
    batch = _periodic_near_pair_batch_size_for_workspace(
        total_pairs=1_000_000,
        lmax=4,
        compute_dtype=np.dtype(compute_dtype),
        workspace_bytes=100 * per_pair,
    )
    assert batch == 100


def test_periodic_near_pair_batch_planner_caps_large_workspaces() -> None:
    batch = _periodic_near_pair_batch_size_for_workspace(
        total_pairs=1_000_000,
        lmax=4,
        compute_dtype=np.dtype(np.complex64),
        workspace_bytes=10**12,
    )
    assert batch == 65_536


def test_ewald_structural_order_truncation_matches_full_table() -> None:
    import pyceles as pcl
    from pyceles.core.periodic.ewald import ewald_structural_sums_2d_batch

    kwargs = dict(
        lmax_struct=3,
        k=2.0 * np.pi / 550.0,
        destinations=np.asarray([[70.0, 25.0, 31.0], [-42.0, 88.0, 0.0]]),
        source=np.asarray([0.0, 0.0, 0.0]),
        lattice=pcl.RectangularLattice2D(430.0, 510.0),
        k_parallel=np.asarray([8.0e-4, -5.0e-4]),
        eta=0.02,
        real_shells=4,
        reciprocal_shells=4,
    )
    full = ewald_structural_sums_2d_batch(**cast(Any, kwargs))
    truncated = ewald_structural_sums_2d_batch(structural_order=5, **cast(Any, kwargs))

    # The compact table has a different m-offset, so compare physical (p,m)
    # entries rather than rectangular storage columns.
    for degree in range(6):
        for order_m in range(-degree, degree + 1):
            np.testing.assert_allclose(
                truncated[:, degree, order_m + 5],
                full[:, degree, order_m + 6],
                rtol=2e-11,
                atol=2e-11,
            )


def test_cartesian_xy_z_layout_detects_complete_grid() -> None:
    from pyceles.core.periodic.rayleigh_cupy import _cartesian_xy_z_layout

    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    z = np.asarray([-1.0, 3.0, 7.0, 9.0])
    points = np.asarray([[x, y, zz] for zz in z for x, y in xy], dtype=float)
    order = np.asarray([7, 0, 10, 3, 5, 1, 11, 4, 8, 2, 9, 6])
    points = points[order]

    layout = _cartesian_xy_z_layout(points)
    assert layout is not None
    unique_xy, unique_z, xy_inverse, z_inverse = layout
    reconstructed = np.column_stack([unique_xy[xy_inverse], unique_z[z_inverse]])
    np.testing.assert_allclose(reconstructed, points)


def test_cartesian_xy_z_layout_rejects_incomplete_product() -> None:
    from pyceles.core.periodic.rayleigh_cupy import _cartesian_xy_z_layout

    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _cartesian_xy_z_layout(points) is None
