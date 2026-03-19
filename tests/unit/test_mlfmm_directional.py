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


def test_directional_transform_roundtrip_sanity() -> None:
    transforms = directional_transforms(3, grid_order=5)
    rng = np.random.default_rng(21)
    box_state = rng.standard_normal(n_modes(3)) + 1j * rng.standard_normal(n_modes(3))

    channels = box_outgoing_to_directional(transforms, box_state)
    recovered = directional_to_box_regular(transforms, *channels)

    rel = np.linalg.norm(recovered - box_state) / np.linalg.norm(box_state)
    assert np.isfinite(rel)
    assert rel < 5.0e-2


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
