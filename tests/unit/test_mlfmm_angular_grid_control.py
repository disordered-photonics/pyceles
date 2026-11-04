"""Angular integration must be refinable independently of translator order."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.operators.mlfmm import (
    MLFMMCouplingOperator,
    MLFMMOptions,
    _angular_grid_order,
    _sampled_rokhlin_translator,
    build_multilevel_mlfmm_operators,
    build_single_level_mlfmm_operators,
    prepare_mlfmm_coupling,
)
from pyceles.core.operators.mlfmm_cupy import prepare_mlfmm_cupy_coupling
from pyceles.core.operators.mlfmm_directional import (
    box_outgoing_to_directional,
    directional_to_box_regular,
    structured_directional_transforms,
)
from pyceles.core.operators.mlfmm_partition import build_uniform_mlfmm_partition
from pyceles.core.translation import RadialLUT


@pytest.mark.parametrize("factor", [0.0, 0.9, np.nan, np.inf])
def test_grid_factor_rejects_invalid_values(factor: float) -> None:
    with pytest.raises(ValueError, match="angular_grid_factor"):
        _angular_grid_order(6, factor)


@pytest.mark.parametrize("depth", [2, 3])
def test_builders_preserve_translator_order_when_refining_grid(depth: int) -> None:
    points = np.array([[x, y, z] for x in [-5.0, 5.0] for y in [-5.0, 5.0] for z in [-5.0, 5.0]])
    partition = build_uniform_mlfmm_partition(
        points, particle_circumscribing_radii=np.full(8, 0.05), depth=depth
    )
    if depth == 2:
        single = build_single_level_mlfmm_operators(
            lmax=1,
            k=1.0,
            positions=points,
            partition=partition,
            box_order=3,
            angular_grid_factor=2.0,
            build_leaf_maps=False,
        )
        assert single.box_order == single.translator_order == 3
        assert single.grid_order == 6
    else:
        multi = build_multilevel_mlfmm_operators(
            lmax=1,
            k=1.0,
            positions=points,
            partition=partition,
            box_order=3,
            angular_grid_factor=2.0,
            build_leaf_maps=False,
        )
        for level in multi.levels[2:]:
            assert level.box_order == level.translator_order == 3
            assert level.grid_order == 6
        transfer = multi.transfers[0]
        assert (transfer.anterpolation.matrix - transfer.interpolation.matrix.T).nnz == 0


def test_refined_grid_resolves_fixed_truncated_modal_action() -> None:
    order = 6
    rng = np.random.default_rng(20261104)
    b = rng.normal(size=n_modes(order)) + 1j * rng.normal(size=n_modes(order))

    def action(grid_order: int) -> np.ndarray:
        transforms = structured_directional_transforms(order, grid_order=grid_order)
        channels = box_outgoing_to_directional(transforms, b)
        diagonal = _sampled_rokhlin_translator(
            np.array([20.0, 0.0, 10.0]),
            k=1.0,
            truncation_order=order,
            directions=transforms.grid.directions,
            weights=transforms.grid.weights,
            dtype=np.dtype(np.complex128),
        )
        return directional_to_box_regular(
            transforms, diagonal * channels[0], diagonal * channels[1]
        )

    reference = action(4 * order)
    old_error = np.linalg.norm(action(order) - reference) / np.linalg.norm(reference)
    new_error = np.linalg.norm(action(9) - reference) / np.linalg.norm(reference)
    assert old_error > 0.1
    assert new_error < 1e-12


@pytest.mark.gpu
def test_refined_grid_and_stencil_preserve_complex_host_gpu_actions_and_adjoint(
    cupy_runtime: tuple[Any, Any],
) -> None:
    cupy, _ = cupy_runtime
    corners = np.array([[x, y, z] for x in [-5.0, 5.0] for y in [-5.0, 5.0] for z in [-5.0, 5.0]])
    points = np.vstack((corners, corners + np.array([0.2, 0.3, 0.1])))
    k = 1.0 + 0.1j
    reference = prepare_mlfmm_coupling(
        lmax=1,
        k=k,
        positions=points,
        particle_circumscribing_radii=np.full(len(points), 0.05),
        radial_lut=RadialLUT(lmax=1, k=k, r_max=20.0, dr=0.05),
        options=MLFMMOptions(
            max_leaf_particles=1,
            max_depth=3,
            accuracy_level=1,
            order_additive=0,
            interpolation_half_width=4,
            angular_grid_factor=2.0,
        ),
    )
    assert isinstance(reference, MLFMMCouplingOperator)
    assert reference.resolved_plan.selected_depth == 3
    device = prepare_mlfmm_cupy_coupling(reference)
    rng = np.random.default_rng(4721)
    shape = (2, len(points) * n_modes(1))
    x, y = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    forward = cupy.asnumpy(device.apply(cupy.asarray(x))).copy()
    adjoint = cupy.asnumpy(device.apply_adjoint(cupy.asarray(y))).copy()
    np.testing.assert_allclose(forward, reference.apply(x), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(adjoint, reference.apply_adjoint(y), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.vdot(forward, y), np.vdot(x, adjoint), rtol=1e-11, atol=1e-12)
