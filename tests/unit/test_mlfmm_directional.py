from __future__ import annotations

import numpy as np

from pyceles.core.indexing import n_modes
from pyceles.core.operators.mlfmm_directional import (
    apply_directional_reflection,
    box_outgoing_to_directional,
    directional_anterpolation,
    directional_grid,
    directional_interpolation,
    directional_to_box_regular,
    directional_transforms,
)
from pyceles.core.translation import translation_ab5_table, translation_block_regular


def test_directional_transform_maps_are_finite_and_shape_consistent() -> None:
    transforms = directional_transforms(3, grid_order=5)
    rng = np.random.default_rng(21)
    box_state = rng.standard_normal(n_modes(3)) + 1j * rng.standard_normal(n_modes(3))

    channels = box_outgoing_to_directional(transforms, box_state)
    recovered = directional_to_box_regular(transforms, *channels)

    assert len(channels) == 4
    for channel in channels:
        assert channel.shape == (transforms.grid.directions.shape[0],)
        assert np.all(np.isfinite(channel))
    assert recovered.shape == box_state.shape
    assert np.all(np.isfinite(recovered))


def test_directional_interpolation_anterpolation_transpose_relation() -> None:
    interpolation = directional_interpolation(3, 5)
    anterpolation = directional_anterpolation(3, 5)
    rng = np.random.default_rng(7)
    u = rng.standard_normal(interpolation.matrix.shape[1]) + 1j * rng.standard_normal(
        interpolation.matrix.shape[1]
    )
    v = rng.standard_normal(interpolation.matrix.shape[0]) + 1j * rng.standard_normal(
        interpolation.matrix.shape[0]
    )

    lhs = np.vdot(interpolation.matrix @ u, v)
    rhs = np.vdot(u, anterpolation.matrix @ v)
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)


def test_physical_directional_reflection_is_involutive() -> None:
    grid = directional_grid(4)
    rng = np.random.default_rng(9)
    channel_size = int(grid.directions.shape[0])
    channels = (
        np.asarray(
            rng.standard_normal(channel_size) + 1j * rng.standard_normal(channel_size),
            dtype=np.complex128,
        ),
        np.asarray(
            rng.standard_normal(channel_size) + 1j * rng.standard_normal(channel_size),
            dtype=np.complex128,
        ),
        np.asarray(
            rng.standard_normal(channel_size) + 1j * rng.standard_normal(channel_size),
            dtype=np.complex128,
        ),
        np.asarray(
            rng.standard_normal(channel_size) + 1j * rng.standard_normal(channel_size),
            dtype=np.complex128,
        ),
    )

    reflected = apply_directional_reflection(grid.reflection_permutation, *channels)
    recovered = apply_directional_reflection(grid.reflection_permutation, *reflected)
    for got, expected in zip(recovered, channels, strict=True):
        np.testing.assert_allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)

    perm = np.asarray(grid.reflection_permutation, dtype=np.int64)
    np.testing.assert_array_equal(perm[perm], np.arange(perm.size, dtype=np.int64))


def test_directional_grid_cache_reuse() -> None:
    grid_a = directional_grid(4, alpha_factor=1.0, beta_factor=1.0)
    grid_b = directional_grid(4, alpha_factor=1.0, beta_factor=1.0)
    assert grid_a is grid_b


def test_directional_grid_beta_order_matches_validated_multilevel_convention() -> None:
    grid = directional_grid(5)
    beta = np.asarray(grid.beta, dtype=float)
    assert beta[0] > beta[-1]


def test_directional_phase_mediation_tracks_exact_axial_recenter_channels() -> None:
    """Directional phase mediation oracle against exact regular translation.

    Reference context:
    Dufva et al., PIER B 4 (2008) 79-99, translational addition-theorem
    framework used for the exact regular-translation baseline.
    """
    box_order = 4
    transforms = directional_transforms(box_order, grid_order=6)
    rng = np.random.default_rng(31)
    box_state = rng.standard_normal(n_modes(box_order)) + 1j * rng.standard_normal(
        n_modes(box_order)
    )

    k = 2.0 * np.pi / 550.0
    delta = np.array([30.0, 0.0, 130.0], dtype=float)
    phase = np.exp(1j * k * (transforms.grid.directions @ delta))

    sampled_channels = tuple(
        channel * phase for channel in box_outgoing_to_directional(transforms, box_state)
    )

    exact_state = translation_block_regular(
        box_order,
        k,
        delta,
        ab5=translation_ab5_table(box_order, dtype=np.complex128),
    ) @ np.asarray(box_state, dtype=np.complex128)
    exact_channels = box_outgoing_to_directional(transforms, exact_state)

    rel = np.linalg.norm(
        np.concatenate(
            [lhs - rhs for lhs, rhs in zip(sampled_channels, exact_channels, strict=True)]
        )
    ) / np.linalg.norm(np.concatenate(exact_channels))
    assert rel < 0.8
