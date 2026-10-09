"""GPU contracts for the sampled-far transforms and transfer workspace."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles._optional import asnumpy, import_cupy
from pyceles.core.operators.mlfmm_directional_cupy import (
    CuPyDirectionalInterpolationData,
    _apply_directional_map,
    _box_outgoing_to_directional_adjoint_cupy,
    _box_outgoing_to_directional_cupy,
    _directional_phase_add_kernel,
    _directional_sum_difference_kernel,
    _directional_to_box_regular_adjoint_cupy,
    _directional_to_box_regular_cupy,
)

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nrhs", (1, 3))
@pytest.mark.parametrize("storage", ("dense", "sparse", "packed_stencil"))
def test_directional_transfer_layouts(nrhs: int, storage: str) -> None:
    cp, _ = import_cupy()
    rng = np.random.default_rng(101)
    host = rng.standard_normal((2, 2, 7, 2 * nrhs)) + 1j * rng.standard_normal((2, 2, 7, 2 * nrhs))
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
@pytest.mark.parametrize("sampling", ((1.0, 1.0), (1.5, 1.25)))
@pytest.mark.parametrize("nrhs", (1, 3))
def test_directional_maps_and_adjoint_match_independent_dense(
    order: int,
    nrhs: int,
    sampling: tuple[float, float],
) -> None:
    from pyceles.core.operators.mlfmm_directional import directional_transforms
    from pyceles.core.operators.mlfmm_directional_cupy import _upload_directional_transforms

    cp, _ = import_cupy()
    dense = directional_transforms(
        order, grid_order=order + 2, alpha_factor=sampling[0], beta_factor=sampling[1]
    )
    device = _upload_directional_transforms(dense, cupy=cp)
    theta = dense.Fth[dense.grid.reflection_permutation]
    phi = dense.Fph[dense.grid.reflection_permutation]
    zeros = np.zeros_like(theta)
    n_dir = theta.shape[0]
    outgoing_four = np.block([[theta, zeros], [phi, zeros], [zeros, theta], [zeros, phi]])
    identity = np.eye(n_dir, dtype=np.complex128)
    mixing = np.block(
        [
            [identity, np.zeros_like(identity), np.zeros_like(identity), 1j * identity],
            [np.zeros_like(identity), identity, -1j * identity, np.zeros_like(identity)],
        ]
    )
    linear_outgoing = mixing @ outgoing_four
    outgoing = np.concatenate(
        (
            (linear_outgoing[:n_dir] + 1j * linear_outgoing[n_dir:]) / np.sqrt(2.0),
            (linear_outgoing[:n_dir] - 1j * linear_outgoing[n_dir:]) / np.sqrt(2.0),
        ),
        axis=0,
    )
    theta_h = theta.conj().T
    phi_h = phi.conj().T
    receive = np.block(
        [
            [theta_h, phi_h],
            [-1j * phi_h, 1j * theta_h],
        ]
    )
    receive = np.concatenate(
        (
            (receive[:, :n_dir] - 1j * receive[:, n_dir:]) / np.sqrt(2.0),
            (receive[:, :n_dir] + 1j * receive[:, n_dir:]) / np.sqrt(2.0),
        ),
        axis=1,
    )
    rng = np.random.default_rng(210 + order)
    n_modes = 2 * theta.shape[1]
    states_storage = rng.standard_normal((2, n_modes, 2 * nrhs)) + 1j * rng.standard_normal(
        (2, n_modes, 2 * nrhs)
    )
    channel_storage = rng.standard_normal((2, 2, n_dir, 2 * nrhs)) + 1j * rng.standard_normal(
        (2, 2, n_dir, 2 * nrhs)
    )
    states = cp.asarray(states_storage)[..., ::2]
    channels = cp.asarray(channel_storage)[..., ::2]
    host_states = states_storage[..., ::2]
    host_channels = channel_storage[..., ::2]
    flat_channels = host_channels.reshape(2, 2 * n_dir, nrhs)
    forward_out = cp.full((2, 2, n_dir, nrhs), complex(np.nan, np.nan))
    receive_out = cp.full((2, n_modes, nrhs), complex(np.nan, np.nan))
    cases = (
        (
            _box_outgoing_to_directional_cupy(device, states, out=forward_out, cupy=cp),
            np.einsum("ij,bjr->bir", outgoing, host_states).reshape(2, 2, n_dir, nrhs),
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
            np.einsum("ij,bjr->bir", receive.conj().T, host_states).reshape(2, 2, n_dir, nrhs),
        ),
    )
    for actual, expected in cases:
        np.testing.assert_allclose(asnumpy(actual), expected, rtol=5e-11, atol=5e-11)
    assert cases[0][0].data.ptr == forward_out.data.ptr
    assert cases[1][0].data.ptr == receive_out.data.ptr
    np.testing.assert_array_equal(asnumpy(states), host_states)
    np.testing.assert_array_equal(asnumpy(channels), host_channels)


def test_phase_add_updates_only_the_supplied_strided_channel() -> None:
    cp, _ = import_cupy()
    rng = np.random.default_rng(44)
    storage_host = rng.standard_normal((2, 2, 5, 7, 6)) + 1j * rng.standard_normal((2, 2, 5, 7, 6))
    phase_host = np.exp(1j * np.arange(5)).reshape(1, 5, 1, 1)
    first_host = rng.standard_normal((2, 1, 7, 3)) + 1j * rng.standard_normal((2, 1, 7, 3))
    storage = cp.asarray(storage_host)
    target = storage[:, 1, :, :, ::2]
    phase = cp.asarray(phase_host)
    first = cp.asarray(first_host)
    returned = _directional_phase_add_kernel()(phase, first, target)
    expected = storage_host.copy()
    expected[:, 1, :, :, ::2] += phase_host * first_host
    assert returned is target
    np.testing.assert_allclose(asnumpy(storage), expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(asnumpy(phase), phase_host)
    np.testing.assert_array_equal(asnumpy(first), first_host)


@pytest.mark.parametrize("reuse", (False, True))
def test_directional_sum_difference_owned_and_strided(reuse: bool) -> None:
    cp, _ = import_cupy()
    rng = np.random.default_rng(91)
    first_host = rng.standard_normal((3, 5, 6)) + 1j * rng.standard_normal((3, 5, 6))
    second_host = rng.standard_normal((3, 5, 6)) + 1j * rng.standard_normal((3, 5, 6))
    first_storage = cp.asarray(first_host)
    second_storage = cp.asarray(second_host)
    first = first_storage[..., ::2]
    second = second_storage[..., ::2]
    kernel = _directional_sum_difference_kernel()
    if reuse:
        plus, minus = kernel(first, second, first, second)
        assert plus.data.ptr == first.data.ptr
        assert minus.data.ptr == second.data.ptr
    else:
        plus, minus = kernel(first, second)
        np.testing.assert_array_equal(asnumpy(first_storage), first_host)
        np.testing.assert_array_equal(asnumpy(second_storage), second_host)
    np.testing.assert_allclose(asnumpy(plus), first_host[..., ::2] + second_host[..., ::2])
    np.testing.assert_allclose(asnumpy(minus), first_host[..., ::2] - second_host[..., ::2])
    np.testing.assert_array_equal(asnumpy(first_storage[..., 1::2]), first_host[..., 1::2])
    np.testing.assert_array_equal(asnumpy(second_storage[..., 1::2]), second_host[..., 1::2])


@pytest.mark.parametrize("action", ("emit", "receive"))
def test_directional_output_contract_and_strided_storage(action: str) -> None:
    from pyceles.core.operators.mlfmm_directional import directional_transforms
    from pyceles.core.operators.mlfmm_directional_cupy import _upload_directional_transforms

    cp, _ = import_cupy()
    prepared = _upload_directional_transforms(directional_transforms(2, grid_order=3), cupy=cp)
    nrhs = 2
    states_shape = (2, 2 * prepared.nscl, nrhs)
    channels_shape = (2, 2, prepared.grid.n_directions, nrhs)
    apply: Any
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    if action == "emit":
        apply = _box_outgoing_to_directional_cupy
        input_shape, output_shape = states_shape, channels_shape
    else:
        apply = _directional_to_box_regular_cupy
        input_shape, output_shape = channels_shape, states_shape
    rng = np.random.default_rng(602)
    host = rng.standard_normal(input_shape) + 1j * rng.standard_normal(input_shape)
    values = cp.asarray(host)
    expected = apply(prepared, values, cupy=cp)
    invalid_outputs = (
        (np.full(output_shape, 13 + 2j, dtype=np.complex128), TypeError, "CuPy"),
        (cp.full(output_shape, 13 + 2j, dtype=cp.complex64), ValueError, "complex128"),
        (cp.full((int(np.prod(output_shape)),), 13 + 2j, dtype=cp.complex128), ValueError, "shape"),
    )
    for invalid, error, message in invalid_outputs:
        before = asnumpy(invalid).copy()
        with pytest.raises(error, match=message):
            apply(prepared, values, out=invalid, cupy=cp)
        np.testing.assert_array_equal(asnumpy(invalid), before)

    # Changing only the last-axis stride keeps a reshape that splits ndir into
    # alpha/beta a view. The unused lanes must never be cleared or overwritten.
    storage = cp.full((*output_shape[:-1], 2 * nrhs), 13 + 2j, dtype=cp.complex128)
    target = storage[..., ::2]
    actual = apply(prepared, values, out=target, cupy=cp)
    assert actual is target
    np.testing.assert_allclose(asnumpy(target), asnumpy(expected), rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(asnumpy(storage[..., 1::2]), 13 + 2j)
    np.testing.assert_array_equal(asnumpy(values), host)

    with pytest.raises(ValueError, match="shape"):
        apply(prepared, values.reshape(-1), cupy=cp)
