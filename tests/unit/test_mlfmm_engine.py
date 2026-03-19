from __future__ import annotations

import numpy as np

from pyceles.core.indexing import n_modes
from pyceles.core.operators.mlfmm import (
    MLFMMOptions,
    apply_multilevel_mlfmm,
    apply_single_level_mlfmm,
    build_multilevel_mlfmm_operators,
    build_single_level_mlfmm_operators,
    resolve_mlfmm_plan,
)
from pyceles.core.operators.mlfmm_partition import MLFMMBox, build_uniform_mlfmm_partition
from pyceles.core.translation import RadialLUT, translation_ab5_table, translation_block


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
            [-1200.0, -1200.0, -1200.0],
            [-1200.0, -1200.0, 1200.0],
            [-1200.0, 1200.0, -1200.0],
            [-1200.0, 1200.0, 1200.0],
            [1200.0, -1200.0, -1200.0],
            [1200.0, -1200.0, 1200.0],
            [1200.0, 1200.0, -1200.0],
            [1200.0, 1200.0, 1200.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 60.0, dtype=float)
    return positions, radii, 2, 2.0 * np.pi / 550.0


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
    assert [level.coords.shape[0] for level in operators.levels] == [1, 8, 8, 8]
    assert len(operators.transfers) == 3
    assert any(len(level.far_offset_batches) > 0 for level in operators.levels[1:])


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
