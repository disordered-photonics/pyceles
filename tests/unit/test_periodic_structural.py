from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.periodic.directsum import periodic_direct_sum_block
from pyceles.core.periodic.structural import (
    block_from_structural_sums,
    periodic_direct_structural_block,
)
from pyceles.core.translation import translation_ab5_table


@pytest.mark.parametrize(
    ("destination", "source"),
    [
        (
            np.array([38.0, 19.0, 24.0], dtype=float),
            np.array([-27.0, -13.0, -35.0], dtype=float),
        ),
        (
            np.array([42.0, 11.0, 8.0], dtype=float),
            np.array([-31.0, -17.0, 8.0], dtype=float),
        ),
    ],
)
def test_direct_structural_block_matches_finite_image_sum(
    destination: np.ndarray,
    source: np.ndarray,
) -> None:
    lattice = pcl.RectangularLattice2D(ax=430.0, ay=470.0)
    k_parallel = np.array([0.0012, -0.0007], dtype=float)
    lmax = 2
    k = 2.0 * np.pi / 550.0
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)

    expected = periodic_direct_sum_block(
        lmax=lmax,
        k=k,
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=k_parallel,
        window=1,
        ab5=ab5,
        dtype=np.complex128,
    )
    actual = periodic_direct_structural_block(
        lmax=lmax,
        k=k,
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=k_parallel,
        window=1,
        ab5=ab5,
        dtype=np.complex128,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_direct_structural_block_matches_self_image_sum() -> None:
    lattice = pcl.RectangularLattice2D(ax=360.0, ay=390.0)
    position = np.array([17.0, -29.0, 41.0], dtype=float)
    k_parallel = np.array([-0.0009, 0.0016], dtype=float)
    lmax = 2
    k = 2.0 * np.pi / 620.0
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)

    expected = periodic_direct_sum_block(
        lmax=lmax,
        k=k,
        destination=position,
        source=position,
        lattice=lattice,
        k_parallel=k_parallel,
        window=1,
        ab5=ab5,
        dtype=np.complex128,
        exclude_zero_shift=True,
    )
    actual = periodic_direct_structural_block(
        lmax=lmax,
        k=k,
        destination=position,
        source=position,
        lattice=lattice,
        k_parallel=k_parallel,
        window=1,
        ab5=ab5,
        dtype=np.complex128,
        exclude_zero_shift=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_structural_block_rejects_incompatible_table_shape() -> None:
    ab5 = translation_ab5_table(1, dtype=np.complex128)

    with pytest.raises(ValueError, match="structural_sums"):
        block_from_structural_sums(
            lmax=1,
            structural_sums=np.zeros((2, 5), dtype=np.complex128),
            ab5=ab5,
        )
