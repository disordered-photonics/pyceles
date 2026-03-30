from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.operators.mlfmm import (
    MLFMMCouplingOperator,
    MLFMMLevelOperators,
    MLFMMOptions,
    MLFMMTransferOperators,
    apply_multilevel_mlfmm,
    apply_single_level_mlfmm,
    box_order_rokhlin_like,
    build_multilevel_mlfmm_operators,
    build_single_level_mlfmm_operators,
    resolve_mlfmm_plan,
)
from pyceles.core.operators.mlfmm_directional import (
    box_outgoing_to_directional,
    directional_anterpolation,
    directional_interpolation,
    directional_to_box_regular,
    directional_transforms,
)
from pyceles.core.operators.mlfmm_partition import MLFMMBox, build_uniform_mlfmm_partition
from pyceles.core.translation import (
    RadialLUT,
    translation_ab5_table,
    translation_block,
    translation_block_rect,
)


@dataclass(frozen=True)
class _TransferScaffold:
    levels: tuple[MLFMMLevelOperators, ...]
    transfer: MLFMMTransferOperators
    child_level: int
    parent_level: int
    leaf_level: int


def _single_level_fixture() -> tuple[np.ndarray, np.ndarray, int, float]:
    positions = np.array(
        [
            [-900.0, -900.0, -900.0],
            [-870.0, -860.0, -930.0],
            [-900.0, -900.0, 900.0],
            [-860.0, -930.0, 870.0],
            [900.0, 900.0, -900.0],
            [870.0, 930.0, -860.0],
            [900.0, 900.0, 900.0],
            [930.0, 870.0, 860.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 80.0, dtype=float)
    return positions, radii, 3, 2.0 * np.pi / 550.0


def _multilevel_fixture() -> tuple[np.ndarray, np.ndarray, int, float]:
    positions = np.array(
        [
            [-300.0, -300.0, -300.0],
            [-300.0, -300.0, 300.0],
            [-300.0, 300.0, -300.0],
            [-300.0, 300.0, 300.0],
            [300.0, -300.0, -300.0],
            [300.0, -300.0, 300.0],
            [300.0, 300.0, -300.0],
            [300.0, 300.0, 300.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 15.0, dtype=float)
    return positions, radii, 1, 2.0 * np.pi / 550.0


def _transfer_fixture() -> tuple[np.ndarray, np.ndarray, int, float]:
    positions = np.array(
        [
            [-980.0, -860.0, -920.0],
            [-940.0, -820.0, 900.0],
            [-900.0, 840.0, -880.0],
            [-960.0, 900.0, 940.0],
            [920.0, -840.0, -900.0],
            [980.0, -900.0, 920.0],
            [880.0, 820.0, -940.0],
            [940.0, 900.0, 960.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 100.0, dtype=float)
    return positions, radii, 3, 2.0 * np.pi / 550.0


@pytest.fixture(scope="module")
def transfer_scaffold():
    positions, radii, lmax, k = _transfer_fixture()
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=2,
    )
    operators = build_multilevel_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=partition,
        box_order=11,
    )
    child_level = int(operators.leaf_level)
    parent_level = int(child_level - 1)
    levels = list(operators.levels)
    for level_idx in (parent_level, child_level):
        level = levels[level_idx]
        levels[level_idx] = MLFMMLevelOperators(
            level=int(level.level),
            coords=level.coords,
            centers=level.centers,
            parent_indices=level.parent_indices,
            children=level.children,
            box_order=int(level.box_order),
            translator_order=int(level.translator_order),
            grid_order=int(level.grid_order),
            directional=directional_transforms(
                int(level.box_order), grid_order=int(level.grid_order)
            ),
            far_offset_batches=level.far_offset_batches,
            offset_diagonals=level.offset_diagonals,
        )
    child = levels[child_level]
    parent = levels[parent_level]
    interpolation = directional_interpolation(child.grid_order, parent.grid_order)
    anterpolation = directional_anterpolation(child.grid_order, parent.grid_order)
    grouped: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for child_idx, parent_idx_raw in enumerate(child.parent_indices):
        parent_idx = int(parent_idx_raw)
        shift = (
            int(child.coords[child_idx, 0] - 2 * parent.coords[parent_idx, 0]),
            int(child.coords[child_idx, 1] - 2 * parent.coords[parent_idx, 1]),
            int(child.coords[child_idx, 2] - 2 * parent.coords[parent_idx, 2]),
        )
        grouped.setdefault(shift, []).append((int(child_idx), parent_idx))
    batches = {
        shift: (
            np.asarray([pair[0] for pair in pairs], dtype=np.int64),
            np.asarray([pair[1] for pair in pairs], dtype=np.int64),
        )
        for shift, pairs in grouped.items()
    }
    phase_up: dict[tuple[int, int, int], np.ndarray] = {}
    phase_down: dict[tuple[int, int, int], np.ndarray] = {}
    for shift, (child_indices, parent_indices) in batches.items():
        delta = np.asarray(
            parent.centers[int(parent_indices[0])] - child.centers[int(child_indices[0])],
            dtype=float,
        )
        phase_up[shift] = np.asarray(
            np.exp(1j * complex(k) * (parent.directional.grid.directions @ delta)),
            dtype=np.complex128,
        )
        phase_down[shift] = np.asarray(
            np.exp(1j * complex(k) * (parent.directional.grid.directions @ (-delta))),
            dtype=np.complex128,
        )
    scaffold = _TransferScaffold(
        levels=tuple(levels),
        transfer=MLFMMTransferOperators(
            child_level=child_level,
            parent_level=parent_level,
            interpolation=interpolation,
            anterpolation=anterpolation,
            batches_by_shift=batches,
            phase_up_by_shift=phase_up,
            phase_down_by_shift=phase_down,
        ),
        child_level=child_level,
        parent_level=parent_level,
        leaf_level=int(operators.leaf_level),
    )
    return scaffold, k


def _full_exact_apply(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    radial_lut: RadialLUT | None,
) -> np.ndarray:
    nm = n_modes(lmax)
    arr = np.asarray(x, dtype=np.complex128).reshape(positions.shape[0], nm)
    y = np.zeros_like(arr, dtype=np.complex128)
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if i == j:
                continue
            y[i] += (
                translation_block(
                    lmax,
                    k,
                    np.asarray(positions[i] - positions[j], dtype=float),
                    ab5=ab5,
                    radial_lut=radial_lut,
                )
                @ arr[j]
            )
    return y.reshape(-1)


def _near_exact_apply(
    *,
    lmax: int,
    k: float,
    positions: np.ndarray,
    x: np.ndarray,
    radial_lut: RadialLUT | None,
    near_pairs: tuple[tuple[int, int], ...],
    leaves: tuple[MLFMMBox, ...],
) -> np.ndarray:
    nm = n_modes(lmax)
    arr = np.asarray(x, dtype=np.complex128).reshape(positions.shape[0], nm)
    y = np.zeros_like(arr, dtype=np.complex128)
    ab5 = translation_ab5_table(lmax, dtype=np.complex128)
    for a, b in near_pairs:
        leaf_a = leaves[a]
        leaf_b = leaves[b]
        if a == b:
            for i in leaf_a.particle_indices:
                for j in leaf_a.particle_indices:
                    if int(i) == int(j):
                        continue
                    y[int(i)] += (
                        translation_block(
                            lmax,
                            k,
                            np.asarray(positions[int(i)] - positions[int(j)], dtype=float),
                            ab5=ab5,
                            radial_lut=radial_lut,
                        )
                        @ arr[int(j)]
                    )
            continue
        for i in leaf_a.particle_indices:
            for j in leaf_b.particle_indices:
                y[int(i)] += (
                    translation_block(
                        lmax,
                        k,
                        np.asarray(positions[int(i)] - positions[int(j)], dtype=float),
                        ab5=ab5,
                        radial_lut=radial_lut,
                    )
                    @ arr[int(j)]
                )
                y[int(j)] += (
                    translation_block(
                        lmax,
                        k,
                        np.asarray(positions[int(j)] - positions[int(i)], dtype=float),
                        ab5=ab5,
                        radial_lut=radial_lut,
                    )
                    @ arr[int(i)]
                )
    return y.reshape(-1)


def test_single_level_exact_near_matches_pairwise_near_subset() -> None:
    positions, radii, lmax, k = _single_level_fixture()
    nm = n_modes(lmax)
    rng = np.random.default_rng(4)
    x = rng.standard_normal(positions.shape[0] * nm) + 1j * rng.standard_normal(
        positions.shape[0] * nm
    )
    plan = resolve_mlfmm_plan(
        positions,
        particle_circumscribing_radii=radii,
        options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
    )
    lut = RadialLUT(
        lmax=12,
        k=k,
        r_max=float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2))),
        dr=5.0,
    )
    operators = build_single_level_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=plan.partition,
        radial_lut=lut,
        box_order=6,
        translator_order=6,
    )

    y_near, _y_far = apply_single_level_mlfmm(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        operators=operators,
        radial_lut=lut,
    )
    y_near_exact = _near_exact_apply(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        radial_lut=lut,
        near_pairs=plan.partition.leaf_near_pairs,
        leaves=plan.partition.leaves,
    )
    np.testing.assert_allclose(y_near, y_near_exact, rtol=1.0e-12, atol=1.0e-12)


def test_single_level_grouped_far_apply_is_finite() -> None:
    positions, radii, lmax, k = _single_level_fixture()
    nm = n_modes(lmax)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(positions.shape[0] * nm) + 1j * rng.standard_normal(
        positions.shape[0] * nm
    )
    plan = resolve_mlfmm_plan(
        positions,
        particle_circumscribing_radii=radii,
        options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
    )
    operators = build_single_level_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=plan.partition,
        radial_lut=None,
        box_order=6,
        translator_order=6,
    )

    _y_near, y_far = apply_single_level_mlfmm(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        operators=operators,
        radial_lut=None,
    )
    assert y_far.shape == x.shape
    assert np.all(np.isfinite(y_far))


def test_multilevel_build_produces_sensible_levels_and_offset_batches() -> None:
    positions, radii, lmax, k = _multilevel_fixture()
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=3,
    )
    operators = build_multilevel_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=partition,
        radial_lut=None,
        box_order=4,
    )

    assert operators.leaf_level == 3
    assert operators.hf_start_level == 3
    assert operators.hf_end_level == 3
    assert [level.coords.shape[0] for level in operators.levels] == [1, 8, 8, 8]
    assert len(operators.transfers) == 0
    assert all(
        len(level.far_offset_batches) == 0 for level in operators.levels[: operators.hf_start_level]
    )


def test_multilevel_auto_hf_start_uses_wavelength_threshold() -> None:
    positions, radii, lmax, k = _multilevel_fixture()
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=3,
    )
    operators = build_multilevel_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=partition,
        radial_lut=None,
        accuracy_level=3,
        order_additive=2,
    )

    assert operators.hf_start_level == 3
    assert operators.hf_end_level == 3


def test_multilevel_auto_hf_start_falls_back_to_two_finest_levels() -> None:
    positions, radii, lmax, _k = _multilevel_fixture()
    k_small_wavelength = 2.0 * np.pi / 150.0
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=3,
    )
    operators = build_multilevel_mlfmm_operators(
        lmax=lmax,
        k=k_small_wavelength,
        positions=positions,
        partition=partition,
        radial_lut=None,
        accuracy_level=3,
        order_additive=2,
    )
    assert operators.hf_start_level == 2
    assert operators.hf_end_level == 3


def test_multilevel_explicit_hf_start_overrides_auto_policy() -> None:
    positions, radii, lmax, k = _multilevel_fixture()
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=3,
    )
    operators = build_multilevel_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=partition,
        radial_lut=None,
        hf_start_level=2,
    )
    assert operators.hf_start_level == 2


def test_single_level_full_apply_stays_in_expected_small_fixture_ballpark() -> None:
    positions, radii, lmax, k = _single_level_fixture()
    nm = n_modes(lmax)
    rng = np.random.default_rng(6)
    x = rng.standard_normal(positions.shape[0] * nm) + 1j * rng.standard_normal(
        positions.shape[0] * nm
    )
    plan = resolve_mlfmm_plan(
        positions,
        particle_circumscribing_radii=radii,
        options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
    )
    lut = RadialLUT(
        lmax=12,
        k=k,
        r_max=float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2))),
        dr=5.0,
    )
    operators = build_single_level_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=plan.partition,
        radial_lut=lut,
        box_order=6,
        translator_order=6,
    )
    y_near, y_far = apply_single_level_mlfmm(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        operators=operators,
        radial_lut=lut,
    )
    y_exact = _full_exact_apply(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        radial_lut=lut,
    )
    rel = np.linalg.norm((y_near + y_far) - y_exact) / np.linalg.norm(y_exact)
    assert rel < 1.5


def test_multilevel_apply_runs_and_produces_finite_output() -> None:
    positions, radii, lmax, k = _multilevel_fixture()
    nm = n_modes(lmax)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(positions.shape[0] * nm) + 1j * rng.standard_normal(
        positions.shape[0] * nm
    )
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=3,
    )
    operators = build_multilevel_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=partition,
        radial_lut=None,
        box_order=4,
    )
    y_near, y_far = apply_multilevel_mlfmm(
        lmax=lmax,
        k=k,
        positions=positions,
        x=x,
        operators=operators,
        radial_lut=None,
    )
    assert y_near.shape == x.shape
    assert y_far.shape == x.shape
    assert np.all(np.isfinite(y_near))
    assert np.all(np.isfinite(y_far))


def test_mlfmm_populate_warms_exact_leaf_near_cache_directly() -> None:
    positions, radii, lmax, k = _single_level_fixture()
    plan = resolve_mlfmm_plan(
        positions,
        particle_circumscribing_radii=radii,
        options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
    )
    radial_lut = RadialLUT(
        lmax=12,
        k=k,
        r_max=float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2))),
        dr=5.0,
    )
    single_level = build_single_level_mlfmm_operators(
        lmax=lmax,
        k=k,
        positions=positions,
        partition=plan.partition,
        radial_lut=radial_lut,
        box_order=6,
        translator_order=6,
        accuracy_level=3,
        order_additive=2,
        dtype=np.complex128,
    )
    coupling = MLFMMCouplingOperator(
        lmax=lmax,
        k=k,
        positions=np.asarray(positions, dtype=float),
        radial_lut=radial_lut,
        resolved_plan=plan,
        dtype=np.dtype(np.complex128),
        cache_translation_blocks=True,
        single_level=single_level,
    )

    assert isinstance(coupling, MLFMMCouplingOperator)
    assert coupling._exact_block_cache == {}
    coupling.populate()

    assert coupling._exact_block_cache is not None
    assert len(coupling._exact_block_cache) > 0


def test_mlfmm_options_expose_box_order_policy_controls() -> None:
    positions, radii, lmax, k = _single_level_fixture()
    partition = build_uniform_mlfmm_partition(
        positions,
        particle_circumscribing_radii=radii,
        depth=2,
    )
    leaf_half_size = float(partition.leaves[0].half_size)

    baseline = box_order_rokhlin_like(
        particle_lmax=lmax,
        k=k,
        box_half_size=leaf_half_size,
    )
    stricter = box_order_rokhlin_like(
        particle_lmax=lmax,
        k=k,
        box_half_size=leaf_half_size,
        accuracy=4,
        additive=4,
    )

    assert stricter > baseline


def test_multilevel_transfer_anterpolation_is_interpolation_transpose() -> None:
    matrix = directional_interpolation(14, 14).matrix
    anterpolation = directional_anterpolation(14, 14)
    rng = np.random.default_rng(15)
    u = rng.standard_normal(matrix.shape[1]) + 1j * rng.standard_normal(matrix.shape[1])
    v = rng.standard_normal(matrix.shape[0]) + 1j * rng.standard_normal(matrix.shape[0])
    lhs = np.vdot(matrix @ u, v)
    rhs = np.vdot(u, anterpolation.matrix @ v)
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)


def test_directional_g_operators_are_derived_from_f_basis() -> None:
    transforms = directional_transforms(8, grid_order=8)
    np.testing.assert_allclose(transforms.Gth, 1j * transforms.Fph, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(transforms.Gph, -1j * transforms.Fth, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(transforms.Gth_adj, -1j * transforms.Fph_adj, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(transforms.Gph_adj, 1j * transforms.Fth_adj, rtol=0.0, atol=0.0)


def test_multilevel_upward_transfer_matches_exact_recenter_oracle(transfer_scaffold) -> None:
    scaffold, k = transfer_scaffold
    transfer = scaffold.transfer
    shift = sorted(transfer.batches_by_shift)[0]
    child_idx_batch, parent_idx_batch = transfer.batches_by_shift[shift]
    child_idx = int(child_idx_batch[0])
    parent_idx = int(parent_idx_batch[0])
    child_level = scaffold.levels[scaffold.child_level]
    parent_level = scaffold.levels[scaffold.parent_level]

    rng = np.random.default_rng(123)
    child_box_state = rng.standard_normal(
        n_modes(child_level.box_order)
    ) + 1j * rng.standard_normal(n_modes(child_level.box_order))
    child_channels = np.vstack(
        box_outgoing_to_directional(child_level.directional, child_box_state)
    )
    child_reindexed = child_channels[:, child_level.directional.grid.reflection_permutation]
    parent_reindexed = child_reindexed @ transfer.interpolation.matrix.T
    parent_channels = parent_reindexed[:, parent_level.directional.grid.reflection_permutation]
    sampled_parent_channels = (
        parent_channels
        * np.asarray(transfer.phase_up_by_shift[shift], dtype=np.complex128)[None, :]
    )

    delta = np.asarray(
        parent_level.centers[parent_idx] - child_level.centers[child_idx], dtype=float
    )
    exact_parent_state = translation_block_rect(
        parent_level.box_order,
        child_level.box_order,
        k,
        delta,
        ab5=translation_ab5_table(
            max(parent_level.box_order, child_level.box_order), dtype=np.complex128
        ),
        family="interior",
    ) @ np.asarray(child_box_state, dtype=np.complex128)
    exact_parent_channels = np.vstack(
        box_outgoing_to_directional(parent_level.directional, exact_parent_state)
    )
    rel = np.linalg.norm(sampled_parent_channels - exact_parent_channels) / np.linalg.norm(
        exact_parent_channels
    )
    assert rel < 0.80


def test_multilevel_downward_transfer_matches_exact_recenter_oracle(transfer_scaffold) -> None:
    scaffold, k = transfer_scaffold
    transfer = scaffold.transfer
    shift = sorted(transfer.batches_by_shift)[0]
    child_idx_batch, parent_idx_batch = transfer.batches_by_shift[shift]
    child_idx = int(child_idx_batch[0])
    parent_idx = int(parent_idx_batch[0])
    child_level = scaffold.levels[scaffold.child_level]
    parent_level = scaffold.levels[scaffold.parent_level]

    rng = np.random.default_rng(321)
    parent_channels = rng.standard_normal(
        (4, parent_level.directional.grid.directions.shape[0])
    ) + 1j * rng.standard_normal((4, parent_level.directional.grid.directions.shape[0]))
    shifted = (
        parent_channels
        * np.asarray(transfer.phase_down_by_shift[shift], dtype=np.complex128)[None, :]
    )
    shifted_reindexed = shifted[:, parent_level.directional.grid.reflection_permutation]
    child_reindexed = shifted_reindexed @ transfer.anterpolation.matrix.T
    child_channels = child_reindexed[:, child_level.directional.grid.reflection_permutation]
    sampled_child_state = directional_to_box_regular(
        child_level.directional,
        child_channels[0],
        child_channels[1],
        child_channels[2],
        child_channels[3],
    )

    delta = np.asarray(
        child_level.centers[child_idx] - parent_level.centers[parent_idx], dtype=float
    )
    parent_state = directional_to_box_regular(
        parent_level.directional,
        parent_channels[0],
        parent_channels[1],
        parent_channels[2],
        parent_channels[3],
    )
    exact_child_state = translation_block_rect(
        child_level.box_order,
        parent_level.box_order,
        k,
        delta,
        ab5=translation_ab5_table(
            max(child_level.box_order, parent_level.box_order), dtype=np.complex128
        ),
        family="interior",
    ) @ np.asarray(parent_state, dtype=np.complex128)
    rel = np.linalg.norm(sampled_child_state - exact_child_state) / np.linalg.norm(
        exact_child_state
    )
    assert rel < 0.50
