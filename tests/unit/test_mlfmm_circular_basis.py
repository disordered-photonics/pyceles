"""Independent normalization/transport checks for circular sampled channels."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from pyceles.core.operators.mlfmm_directional import (
    MLFMMDirectionalTransforms,
    box_outgoing_to_directional,
    box_outgoing_to_directional_adjoint,
    directional_to_box_regular,
    directional_to_box_regular_adjoint,
    directional_transforms,
    structured_directional_transforms,
)


def _linear_emission(transforms: MLFMMDirectionalTransforms) -> np.ndarray:
    # Independent legacy two-channel matrix; no circular-factor helper is used.
    theta = transforms.Fth[transforms.grid.reflection_permutation]
    phi = transforms.Fph[transforms.grid.reflection_permutation]
    return cast(
        np.ndarray,
        np.asarray(np.block([[theta, 1j * phi], [phi, -1j * theta]]), dtype=np.complex128),
    )


def _rotate(channels: np.ndarray) -> np.ndarray:
    return cast(
        np.ndarray,
        np.stack((channels[0] + 1j * channels[1], channels[0] - 1j * channels[1])) / np.sqrt(2.0),
    )


@pytest.mark.parametrize("order", (1, 3, 6))
@pytest.mark.parametrize("sampling", ((1.0, 1.0), (1.5, 1.25)))
def test_circular_boundary_is_unitary_and_receive_is_its_adjoint(
    order: int, sampling: tuple[float, float]
) -> None:
    dense = directional_transforms(
        order, grid_order=order + 2, alpha_factor=sampling[0], beta_factor=sampling[1]
    )
    structured = structured_directional_transforms(
        order, grid_order=order + 2, alpha_factor=sampling[0], beta_factor=sampling[1]
    )
    old = _linear_emission(dense)
    n_dir = dense.Fth.shape[0]
    rng = np.random.default_rng(47 + order)
    state = np.asarray(
        rng.standard_normal(old.shape[1]) + 1j * rng.standard_normal(old.shape[1]),
        dtype=np.complex128,
    )
    channels = np.asarray(
        rng.standard_normal((2, n_dir)) + 1j * rng.standard_normal((2, n_dir)),
        dtype=np.complex128,
    )
    original_state, original_channels = state.copy(), channels.copy()
    state.setflags(write=False)
    channels.setflags(write=False)
    old_channels = (old @ state).reshape(2, n_dir)
    expected = _rotate(old_channels)
    circular_matrix = _rotate(old.reshape(2, n_dir, -1)).reshape(old.shape)
    expected_receive = circular_matrix.conj().T @ channels.reshape(-1)
    np.testing.assert_allclose(np.linalg.norm(expected), np.linalg.norm(old_channels), rtol=2e-14)
    for transforms in (dense, structured):
        actual = np.stack(box_outgoing_to_directional(transforms, state))
        receive = directional_to_box_regular(transforms, *channels)
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(receive, expected_receive, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(
            np.vdot(actual, channels), np.vdot(state, receive), rtol=3e-12, atol=3e-12
        )
        np.testing.assert_allclose(
            box_outgoing_to_directional_adjoint(transforms, *channels),
            expected_receive,
            rtol=3e-12,
            atol=3e-12,
        )
        np.testing.assert_allclose(
            np.stack(directional_to_box_regular_adjoint(transforms, state)),
            expected,
            rtol=3e-12,
            atol=3e-12,
        )
    np.testing.assert_array_equal(state, original_state)
    np.testing.assert_array_equal(channels, original_channels)


def test_circular_basis_preserves_complex_rectangular_channel_independent_transport() -> None:
    source = directional_transforms(2, grid_order=3)
    target = directional_transforms(3, grid_order=4)
    old_source = _linear_emission(source)
    old_target = _linear_emission(target)
    source_dirs, target_dirs = source.Fth.shape[0], target.Fth.shape[0]
    rng = np.random.default_rng(52)
    transport = rng.standard_normal((target_dirs, source_dirs)) + 1j * rng.standard_normal(
        (target_dirs, source_dirs)
    )
    state = np.asarray(
        rng.standard_normal(old_source.shape[1]) + 1j * rng.standard_normal(old_source.shape[1]),
        dtype=np.complex128,
    )
    old_emitted = (old_source @ state).reshape(2, source_dirs)
    expected = old_target.conj().T @ (old_emitted @ transport.T).reshape(-1)
    circular_emitted = np.stack(box_outgoing_to_directional(source, state))
    transported = circular_emitted @ transport.T
    result = directional_to_box_regular(target, *transported)
    np.testing.assert_allclose(result, expected, rtol=5e-12, atol=5e-11)
