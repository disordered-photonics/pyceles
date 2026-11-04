"""Resampling accuracy must be independently refinable from Hankel order."""

from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.operators.mlfmm_directional import (
    apply_directional_reflection,
    box_outgoing_to_directional,
    directional_anterpolation,
    directional_interpolation,
    structured_directional_transforms,
)


def test_interpolation_width_is_part_of_cache_identity() -> None:
    old = directional_interpolation(22, 39)
    assert old is directional_interpolation(22, 39, stencil_half_width=2)
    refined = directional_interpolation(22, 39, stencil_half_width=6)
    assert refined is directional_interpolation(22, 39, stencil_half_width=6)
    assert refined is not old
    assert refined.matrix.nnz > old.matrix.nnz
    down = directional_anterpolation(22, 39, stencil_half_width=6)
    assert (down.matrix - refined.matrix.T).nnz == 0


@pytest.mark.parametrize("width", [0, -1, 24])
def test_interpolation_rejects_invalid_width(width: int) -> None:
    with pytest.raises(ValueError, match="stencil_half_width"):
        directional_interpolation(22, 39, stencil_half_width=width)


def test_interpolation_rejects_fractional_width() -> None:
    with pytest.raises(TypeError):
        directional_interpolation(22, 39, stencil_half_width=2.5)  # type: ignore[arg-type]


def test_identical_grids_do_not_require_a_resampling_stencil() -> None:
    interpolation = directional_interpolation(1, 1, stencil_half_width=6)
    np.testing.assert_array_equal(
        interpolation.matrix.toarray(), np.eye(interpolation.matrix.shape[0])
    )


@pytest.mark.parametrize("offset", [0.0, 3.0])
def test_width_refines_sampled_physical_pattern_without_increasing_order(offset: float) -> None:
    rng = np.random.default_rng(4731)
    b = rng.normal(size=n_modes(3)) + 1j * rng.normal(size=n_modes(3))
    source = structured_directional_transforms(3, grid_order=22)
    target = structured_directional_transforms(3, grid_order=39)
    direction = np.array([0.3, 0.5, 0.8])
    direction /= np.linalg.norm(direction)

    def values(t):
        unreflected = np.asarray(
            apply_directional_reflection(
                t.grid.reflection_permutation, *box_outgoing_to_directional(t, b)
            )
        )
        return unreflected * np.exp(1j * offset * (t.grid.directions @ direction))[None, :]

    x, expected = values(source), values(target)
    errors = []
    for width in [2, 4, 6]:
        interp = directional_interpolation(22, 39, stencil_half_width=width)
        actual = (interp.matrix @ x.T).T
        errors.append(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    assert errors[1] < errors[0] / 10
    assert errors[2] < errors[1] / 10
