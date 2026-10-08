"""GPU contracts for the sampled-far transforms and transfer workspace."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.operators.mlfmm_cupy import (
    CuPyDirectionalInterpolationData,
    _apply_directional_map,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nrhs", (1, 3))
@pytest.mark.parametrize("storage", ("dense", "sparse", "packed_stencil"))
def test_directional_transfer_layouts(nrhs: int, storage: str) -> None:
    cp, _ = import_cupy()
    rng = np.random.default_rng(101)
    host = rng.standard_normal((2, 4, 7, 2 * nrhs)) + 1j * rng.standard_normal((2, 4, 7, 2 * nrhs))
    source = cp.asarray(host)[..., ::2]  # Device-strided, not just a sliced host upload.
    before = source.copy()
    cols = np.array([[0, 3, -1], [1, 5, 6], [2, 4, -1]], dtype=np.int32)
    vals = np.array([[1, 2j, 0], [2, -1j, 3], [1j, -2, 0]], dtype=np.complex128)
    dense = np.zeros((3, 7), dtype=np.complex128)
    for row in range(3):
        for slot in range(3):
            if cols[row, slot] >= 0:
                dense[row, cols[row, slot]] = vals[row, slot]
    matrix: Any
    if storage == "dense":
        matrix = cp.asarray(dense)
    elif storage == "sparse":
        from cupyx.scipy.sparse import csr_matrix

        matrix = csr_matrix(cp.asarray(dense))
    else:
        matrix = (cp.asarray(cols.ravel()), cp.asarray(vals.ravel()), np.int32(3))
    mapping = CuPyDirectionalInterpolationData(
        source_order=7,
        target_order=3,
        matrix=matrix,
        nnz=int(np.count_nonzero(dense)),
        storage=storage,
    )
    result = _apply_directional_map(source, mapping, cupy=cp)
    expected = np.einsum("ij,bcjr->bcir", dense, host[..., ::2])
    np.testing.assert_allclose(asnumpy(result), expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(asnumpy(source), asnumpy(before))


@pytest.mark.parametrize("order", (3, 8))
@pytest.mark.parametrize("nrhs", (1, 3))
def test_directional_maps_and_adjoint_match_independent_dense(
    order: int,
    nrhs: int,
) -> None:
    from pyceles.core.operators.mlfmm_cupy import (
        _box_outgoing_to_directional_adjoint_cupy,
        _box_outgoing_to_directional_cupy,
        _directional_to_box_regular_adjoint_cupy,
        _directional_to_box_regular_cupy,
        _upload_directional_transforms,
    )
    from pyceles.core.operators.mlfmm_directional import directional_transforms

    cp, _ = import_cupy()
    dense = directional_transforms(order, grid_order=order + 2)
    device = _upload_directional_transforms(dense, cupy=cp)
    theta = dense.Fth[dense.grid.reflection_permutation]
    phi = dense.Fph[dense.grid.reflection_permutation]
    zeros = np.zeros_like(theta)
    outgoing = np.block([[theta, zeros], [phi, zeros], [zeros, theta], [zeros, phi]])
    theta_h = theta.conj().T
    phi_h = phi.conj().T
    receive = np.block(
        [
            [theta_h, phi_h, -1j * phi_h, 1j * theta_h],
            [-1j * phi_h, 1j * theta_h, theta_h, phi_h],
        ]
    )
    rng = np.random.default_rng(210 + order)
    n_modes = 2 * theta.shape[1]
    n_dir = theta.shape[0]
    states_storage = rng.standard_normal((2, n_modes, 2 * nrhs)) + 1j * rng.standard_normal(
        (2, n_modes, 2 * nrhs)
    )
    channel_storage = rng.standard_normal((2, 4, n_dir, 2 * nrhs)) + 1j * rng.standard_normal(
        (2, 4, n_dir, 2 * nrhs)
    )
    states = cp.asarray(states_storage)[..., ::2]
    channels = cp.asarray(channel_storage)[..., ::2]
    host_states = states_storage[..., ::2]
    host_channels = channel_storage[..., ::2]
    flat_channels = host_channels.reshape(2, 4 * n_dir, nrhs)
    forward_out = cp.full((2, 4, n_dir, nrhs), complex(np.nan, np.nan))
    receive_out = cp.full((2, n_modes, nrhs), complex(np.nan, np.nan))
    cases = (
        (
            _box_outgoing_to_directional_cupy(device, states, out=forward_out, cupy=cp),
            np.einsum("ij,bjr->bir", outgoing, host_states).reshape(2, 4, n_dir, nrhs),
        ),
        (
            _directional_to_box_regular_cupy(device, channels, out=receive_out, cupy=cp),
            np.einsum("ij,bjr->bir", receive, flat_channels),
        ),
        (
            _box_outgoing_to_directional_adjoint_cupy(device, channels, cupy=cp),
            np.einsum("ij,bjr->bir", outgoing.conj().T, flat_channels),
        ),
        (
            _directional_to_box_regular_adjoint_cupy(device, states, cupy=cp),
            np.einsum("ij,bjr->bir", receive.conj().T, host_states).reshape(2, 4, n_dir, nrhs),
        ),
    )
    for actual, expected in cases:
        np.testing.assert_allclose(asnumpy(actual), expected, rtol=5e-11, atol=5e-11)
    assert cases[0][0].data.ptr == forward_out.data.ptr
    assert cases[1][0].data.ptr == receive_out.data.ptr
    np.testing.assert_array_equal(asnumpy(states), host_states)
    np.testing.assert_array_equal(asnumpy(channels), host_channels)


def test_phase_add_updates_only_the_supplied_strided_channel() -> None:
    from pyceles.core.operators.mlfmm_cupy import _directional_phase_add_kernel

    cp, _ = import_cupy()
    rng = np.random.default_rng(44)
    storage_host = rng.standard_normal((2, 4, 5, 7, 6)) + 1j * rng.standard_normal((2, 4, 5, 7, 6))
    phase_host = np.exp(1j * np.arange(5)).reshape(1, 5, 1, 1)
    beta_host = rng.standard_normal((2, 1, 7, 3)) + 1j * rng.standard_normal((2, 1, 7, 3))
    storage = cp.asarray(storage_host)
    target = storage[:, 1, :, :, ::2]
    phase = cp.asarray(phase_host)
    beta = cp.asarray(beta_host)
    returned = _directional_phase_add_kernel()(phase, beta, target)
    expected = storage_host.copy()
    expected[:, 1, :, :, ::2] += phase_host * beta_host
    assert returned is target
    np.testing.assert_allclose(asnumpy(storage), expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(asnumpy(phase), phase_host)
    np.testing.assert_array_equal(asnumpy(beta), beta_host)
