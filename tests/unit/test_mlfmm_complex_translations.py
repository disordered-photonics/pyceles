"""Physical regular reception and algebraic adjoints in CELES ordering."""

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.operators.mlfmm import (
    _aggregate_leaf_box_states,
    _build_numpy_leaf_apply_groups_dense,
    _build_numpy_leaf_apply_groups_otf,
    _receive_leaf_boxes_to_particles,
    _regular_leaf_receive_map,
)
from pyceles.core.operators.mlfmm_partition import build_uniform_mlfmm_partition
from pyceles.core.translation import translation_block_rect


@pytest.mark.parametrize("k", [1.1, 1.1 + 0.3j])
@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("lmax, box_order", [(1, 3), (2, 3), (3, 2)])
def test_regular_reception_and_both_leaf_adjoints_match_opposite_shifts(k, dense, lmax, box_order):
    positions = np.array([[0.2, -0.3, 0.4], [-0.1, 0.2, 0.3]])
    partition = build_uniform_mlfmm_partition(
        positions, particle_circumscribing_radii=np.zeros(2), depth=0
    )
    deltas = partition.leaves[0].center - positions
    aggregation = np.hstack(
        [translation_block_rect(box_order, lmax, k, delta, family="interior") for delta in deltas]
    )
    reception = np.vstack(
        [translation_block_rect(lmax, box_order, k, -delta, family="interior") for delta in deltas]
    )
    actual = _regular_leaf_receive_map(aggregation, lmax=lmax, box_order=box_order, k=k)
    np.testing.assert_allclose(actual, reception, rtol=2e-13, atol=2e-15)
    if complex(k).imag:
        assert np.linalg.norm(reception - aggregation.conj().T) / np.linalg.norm(reception) > 0.01

    options = dict(
        lmax=lmax, box_order=box_order, k=k, radial_lut=None, dtype=np.dtype(np.complex128)
    )
    groups = (
        _build_numpy_leaf_apply_groups_dense(positions=positions, partition=partition, **options)
        if dense
        else _build_numpy_leaf_apply_groups_otf(lmax=lmax, positions=positions, partition=partition)
    )
    rng = np.random.default_rng(67)
    particle = rng.normal(size=2 * n_modes(lmax)) + 1j * rng.normal(size=2 * n_modes(lmax))
    box = rng.normal(size=(1, n_modes(box_order))) + 1j * rng.normal(size=(1, n_modes(box_order)))
    for adjoint in (False, True):
        outgoing = _aggregate_leaf_box_states(
            x=particle, leaf_groups=groups, n_leaves=1, adjoint_receive=adjoint, **options
        )
        incoming = _receive_leaf_boxes_to_particles(
            incoming_box=box,
            n_particles=2,
            leaf_groups=groups,
            adjoint_aggregation=adjoint,
            **options,
        )
        expected_out = (reception.conj().T if adjoint else aggregation) @ particle
        expected_in = (aggregation.conj().T if adjoint else reception) @ box[0]
        np.testing.assert_allclose(outgoing[0], expected_out, rtol=2e-13, atol=2e-15)
        np.testing.assert_allclose(incoming.reshape(-1), expected_in, rtol=2e-13, atol=2e-15)
