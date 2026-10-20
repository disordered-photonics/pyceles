from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.indexing import iter_modes, n_modes
from pyceles.core.periodic.directsum import periodic_direct_sum_block
from pyceles.core.periodic.structural import (
    apply_structural_sums_to_vector,
    block_from_structural_sums,
    blocks_from_structural_sums,
    direct_structural_sums_2d,
    periodic_direct_structural_block,
    sparse_translation_contraction,
    translation_contraction_tensor,
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


def test_complex_k_structural_block_matches_finite_image_sum() -> None:
    lattice = pcl.RectangularLattice2D(ax=430.0, ay=470.0)
    destination = np.array([38.0, 19.0, 24.0], dtype=float)
    source = np.array([-27.0, -13.0, -35.0], dtype=float)
    k_parallel = np.array([0.0012, -0.0007], dtype=float)
    lmax = 2
    k = 2.0 * np.pi / 550.0 * (1.0 + 0.1j)
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


@pytest.mark.reference
def test_complex_k_structural_channels_match_treams_references() -> None:
    """Low-order structural channels agree with TREAMS for a passive complex k."""
    lattice = pcl.RectangularLattice2D(ax=1.1, ay=1.1)
    k = 1.1 + 1.3j
    k_parallel = np.array([0.13, -0.17], dtype=float)
    references = {
        (0, 0): -0.13162341341421477 - 0.5398330001188543j,
        (1, -1): 1.3245766329564208 + 0.6402455593103076j,
        (1, 1): -1.1970171347512775 + 0.8803182169972379j,
        (2, 0): 5.884788872029801 - 8.570711128291595j,
        (2, 2): 5.832078669350794 + 10.336326471419452j,
        (4, -4): -1918.5385476714225 + 1235.6102993165173j,
        (4, 4): -519.9602098672777 - 2221.507886170242j,
    }

    actual = direct_structural_sums_2d(
        lmax=2,
        k=k,
        destination=np.array([-0.23, 0.14, 0.0]),
        source=np.zeros(3),
        lattice=lattice,
        k_parallel=k_parallel,
        window=12,
    )

    # TREAMS lattice.lsumsw2d uses h_l Y_lm at r + R. The same Bloch phase
    # with pyceles's displacement d - R uses d=-r. CELES P_l^|m| differs
    # from Y_lm by sqrt(2*pi), plus TREAMS's Condon sign for positive m.
    for (degree, order), treams_value in references.items():
        normalization = np.sqrt(2.0 * np.pi)
        if order > 0 and order % 2:
            normalization *= -1.0
        np.testing.assert_allclose(
            actual[degree, order + 4],
            normalization * treams_value,
            rtol=1e-8,
            atol=1e-8,
        )


@pytest.mark.reference
def test_complex_k_vertical_structural_channels_match_treams() -> None:
    """Noncoplanar channels match TREAMS sums in the same SVWF normalization."""
    modes = ((0, 0), (1, -1), (1, 1), (2, 0), (2, 2), (4, -4), (4, 4))
    # Infinite lattice sums, converted from TREAMS's spherical harmonics.
    expected = np.asarray(
        [
            0.1140686385453382 - 0.7923951778392783j,
            0.3506809780661142 + 0.21718978043930304j,
            0.40117163386175303 - 0.18939109915826058j,
            -1.7682307709286018 + 3.2492187050691816j,
            0.5180381396866833 + 1.3991014628741036j,
            -18.69200533427095 + 18.085309890809814j,
            0.08065627987504802 - 25.139478290848793j,
        ]
    )
    actual = direct_structural_sums_2d(
        lmax=2,
        k=1.1 + 1.3j,
        destination=np.asarray([-0.23, 0.14, 0.4]),
        source=np.zeros(3),
        lattice=pcl.RectangularLattice2D(1.1, 1.1),
        k_parallel=np.asarray([0.13, -0.17]),
        window=16,
    )

    np.testing.assert_allclose(
        [actual[degree, order + 4] for degree, order in modes],
        expected,
        rtol=5e-10,
        atol=2e-10,
    )


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


def test_batched_structural_blocks_match_single_block_assembly() -> None:
    lmax = 2
    order = 2 * lmax
    rng = np.random.default_rng(4321)
    sums = (
        rng.normal(size=(4, order + 1, 2 * order + 1))
        + 1j * rng.normal(size=(4, order + 1, 2 * order + 1))
    ).astype(np.complex128)
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    tensor = translation_contraction_tensor(lmax=lmax, ab5=ab5)

    expected = np.zeros((sums.shape[0], n_modes(lmax), n_modes(lmax)), dtype=np.complex128)
    for d in range(sums.shape[0]):
        for _tau_dst, _l_dst, m_dst, dst_idx in iter_modes(lmax):
            for _tau_src, _l_src, m_src, src_idx in iter_modes(lmax):
                expected[d, dst_idx, src_idx] = np.sum(
                    ab5[dst_idx, src_idx, :] * sums[d, :, m_src - m_dst + order]
                )
    actual = blocks_from_structural_sums(
        lmax=lmax,
        structural_sums=sums,
        ab5=ab5,
        dtype=np.complex128,
        contraction_tensor=tensor,
    )

    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize("lmax", [1, 2, 3, 4, 6])
def test_sparse_translation_contraction_matches_dense_valid_channels(lmax: int) -> None:
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    dense = translation_contraction_tensor(lmax=lmax, ab5=ab5)
    order = 2 * lmax
    compact = np.concatenate(
        [dense[:, :, degree, order - degree : order + degree + 1] for degree in range(order + 1)],
        axis=2,
    )
    row_ptr, input_modes, channels, values = sparse_translation_contraction(
        lmax=lmax,
        ab5=ab5,
        dtype=np.complex128,
    )
    reconstructed = np.zeros_like(compact)
    for output_mode in range(n_modes(lmax)):
        start = int(row_ptr[output_mode])
        stop = int(row_ptr[output_mode + 1])
        reconstructed[output_mode, input_modes[start:stop], channels[start:stop]] = values[
            start:stop
        ]
    np.testing.assert_array_equal(reconstructed, compact)
