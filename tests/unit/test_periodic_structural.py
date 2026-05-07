from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.indexing import n_modes
from pyceles.core.periodic.directsum import periodic_direct_sum_block
from pyceles.core.periodic.structural import (
    apply_structural_sums_to_vector,
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


def test_apply_structural_sums_to_vector_matches_dense_blocks() -> None:
    lmax = 2
    nm = n_modes(lmax)
    order = 2 * lmax
    rng = np.random.default_rng(1234)
    sums = (
        rng.normal(size=(3, order + 1, 2 * order + 1))
        + 1j * rng.normal(size=(3, order + 1, 2 * order + 1))
    ).astype(np.complex128)
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    vector = (rng.normal(size=nm) + 1j * rng.normal(size=nm)).astype(np.complex128)

    expected = np.vstack(
        [
            block_from_structural_sums(
                lmax=lmax,
                structural_sums=sums[i],
                ab5=ab5,
                dtype=np.complex128,
            )
            @ vector
            for i in range(sums.shape[0])
        ]
    )

    actual = apply_structural_sums_to_vector(
        lmax=lmax,
        structural_sums=sums,
        ab5=ab5,
        vector=vector,
        dtype=np.complex128,
    )

    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)
