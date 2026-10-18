"""Absorbing-host hierarchy actions, solves, and exact algebraic adjoints."""

from dataclasses import replace

import numpy as np
import pytest

from pyceles._optional import asnumpy
from pyceles.core.operators import PairwiseCouplingOperator, prepare_matvec
from pyceles.core.operators.mlfmm import (
    MLFMMCouplingOperator,
    MLFMMOptions,
    prepare_mlfmm_coupling,
    resolve_mlfmm_plan,
)
from pyceles.core.operators.mlfmm_cupy import (
    CuPyMLFMMHostCachePolicy,
    prepare_mlfmm_cupy_coupling,
)
from pyceles.core.particles import TMatrixParticle, spheres_from_arrays
from pyceles.core.sources import PlaneWave
from pyceles.linear import solve_linear_system


@pytest.fixture(
    scope="module",
    params=[
        (1, True, "complex128"),
        (2, True, "complex128"),
        (2, False, "complex128"),
        (3, True, "complex128"),
        (3, False, "complex128"),
        (4, True, "complex128"),
        (4, False, "complex128"),
        (3, False, "complex64"),
    ],
    ids=lambda case: f"depth{case[0]}-{'dense' if case[1] else 'otf'}-{case[2]}",
)
def absorbing_hierarchy(request):
    depth, dense, dtype = request.param
    corners = np.array([[x, y, z] for x in (-2.5, 2.5) for y in (-2.5, 2.5) for z in (-2.5, 2.5)])
    positions = np.vstack(
        [
            corners + offset
            for offset in ([0, 0, 0], [0.14, 0.11, 0.09], [0.3, 0.2, 0.17], [0.4, 0.3, 0.28])
        ]
        + [corners[0] + [[0.7, 0.15, 0.09], [1.4, 0.3, 0.18], [0.35, 0.8, 0.2]]]
    )
    host = 1.33 + 0.1j
    particles = spheres_from_arrays(
        positions=positions, radii=np.full(len(positions), 0.04), refractive_indices=1.7 + 0.03j
    )
    options = MLFMMOptions(
        max_leaf_particles=2, max_depth=depth, accuracy_level=2, order_additive=1
    )
    reference = prepare_matvec(
        lmax=1,
        k=host,
        n_medium=host,
        particles=particles,
        operator_dtype=dtype,
        radial_lut_dr=2 * np.pi / 8192,
    )
    assert isinstance(reference.coupling, PairwiseCouplingOperator)
    assert reference.coupling.radial_lut is not None
    plan = resolve_mlfmm_plan(
        positions, particle_circumscribing_radii=particles.circumscribing_radii, options=options
    )
    assert plan.selected_depth == depth
    coupling = prepare_mlfmm_coupling(
        lmax=1,
        k=host,
        positions=positions,
        particle_circumscribing_radii=particles.circumscribing_radii,
        radial_lut=reference.coupling.radial_lut,
        options=options,
        dtype=np.dtype(dtype),
        build_leaf_maps=dense,
    )
    near: set[tuple[int, int]] = set()
    for a, b in plan.partition.leaf_near_pairs:
        for i in plan.partition.leaves[a].particle_indices:
            for j in plan.partition.leaves[b].particle_indices:
                near.update(((int(i), int(j)), (int(j), int(i))))
    far_targets = {
        source: [i for i in range(len(positions)) if i != source and (i, source) not in near]
        for source in (0, 17, 34)
    }
    if depth > 1:
        assert isinstance(coupling, MLFMMCouplingOperator)
        if coupling.multilevel is not None:
            assert all(level.far_offset_batches for level in coupling.multilevel.levels[2:])
        else:
            assert coupling.single_level is not None
            assert coupling.single_level.far_offset_batches
        assert all(far_targets.values())
        if coupling.multilevel:
            for transfer in coupling.multilevel.transfers:
                parent = coupling.multilevel.levels[transfer.parent_level]
                child = coupling.multilevel.levels[transfer.child_level]
                for shift, (child_ids, parent_ids) in transfer.batches_by_shift.items():
                    delta = parent.centers[parent_ids[0]] - child.centers[child_ids[0]]
                    phase = host * (parent.directional.grid.directions @ delta)
                    np.testing.assert_allclose(
                        transfer.phase_up_by_shift[shift], np.exp(1j * phase), rtol=1e-14
                    )
                    np.testing.assert_allclose(
                        transfer.phase_down_by_shift[shift], np.exp(-1j * phase), rtol=1e-14
                    )
    return reference, coupling, particles, options, far_targets, dense


def _check_hierarchy(reference, candidate, far_targets, backend):
    dtype = np.dtype(reference.dtype)
    # Deep sampled transfers can cancel large intermediate terms. Compare
    # whole coefficient blocks at the same c128 tolerance as the hierarchy
    # adjoint checks, rather than requiring relative accuracy in weak entries.
    tolerance = 2e-6 if dtype == np.dtype(np.complex64) else 1e-11
    rng = np.random.default_rng(67)
    n = len(reference.positions) * 6
    x = np.asarray(rng.normal(size=n) + 1j * rng.normal(size=n), dtype=dtype)
    y = np.asarray(rng.normal(size=n) + 1j * rng.normal(size=n), dtype=dtype)
    wx = candidate.apply_W(x)
    why = candidate.coupling.apply_adjoint(y)
    error = abs(np.vdot(asnumpy(wx), y) - np.vdot(x, asnumpy(why)))
    assert error / (np.linalg.norm(asnumpy(wx)) * np.linalg.norm(y)) < tolerance

    # Isolate far destinations so strong exact-near interactions cannot conceal
    # approximation errors. Accuracy level 2 is an ordinal setting, not rtol.
    for source, targets in far_targets.items():
        probe = np.zeros((len(reference.positions), 6), dtype=dtype)
        probe[source] = rng.normal(size=6) + 1j * rng.normal(size=6)
        expected = reference.apply_W(probe.reshape(-1)).reshape(-1, 6)
        actual = asnumpy(candidate.apply_W(probe.reshape(-1))).reshape(-1, 6)
        if targets:
            assert (
                np.linalg.norm(actual[targets] - expected[targets])
                / np.linalg.norm(expected[targets])
                < 0.01
            )
        else:
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-12)

    source = PlaneWave(2 * np.pi, medium_n=1.33 + 0.1j, polar_angle=0.4, amplitude=0.7 - 0.2j)
    rhs = reference.rhs_Tb(source.incident_coeffs(reference.positions, 1, dtype=dtype).reshape(-1))
    rtol = 2e-6 if dtype == np.dtype(np.complex64) else 1e-10
    expected_solve = solve_linear_system(
        reference.apply_A, rhs, rtol=rtol, dtype=dtype, show_progress=False
    )
    actual_solve = solve_linear_system(
        candidate.apply_A, rhs, backend=backend, rtol=rtol, dtype=dtype, show_progress=False
    )
    assert actual_solve.info == expected_solve.info == 0
    assert (
        np.linalg.norm(actual_solve.x - expected_solve.x) / np.linalg.norm(expected_solve.x) < 3e-6
    )

    if backend == "cupy":
        block = np.column_stack((x, y))
        actual = candidate.apply_W(block)
        dual = np.asarray(
            rng.normal(size=block.shape) + 1j * rng.normal(size=block.shape), dtype=dtype
        )
        reverse = candidate.coupling.apply_adjoint(dual)
        expected = np.column_stack([asnumpy(candidate.apply_W(column)) for column in block.T])
        assert np.linalg.norm(asnumpy(actual) - expected) / np.linalg.norm(expected) < tolerance
        error = abs(np.vdot(asnumpy(actual), dual) - np.vdot(block, asnumpy(reverse)))
        assert error / (np.linalg.norm(asnumpy(actual)) * np.linalg.norm(dual)) < tolerance


def test_absorbing_hierarchy_numpy_actions_and_solve(absorbing_hierarchy):
    reference, coupling, particles, options, far_targets, dense = absorbing_hierarchy
    if dense:
        _check_hierarchy(reference, replace(reference, coupling=coupling), far_targets, "numpy")
        return
    # The diagnostic NumPy on-the-fly route uses the same maps. Compare its two
    # actions directly rather than repeating the dense path's full solves.
    dense_coupling = prepare_mlfmm_coupling(
        lmax=1,
        k=1.33 + 0.1j,
        positions=reference.positions,
        particle_circumscribing_radii=particles.circumscribing_radii,
        radial_lut=reference.coupling.radial_lut,
        options=options,
        dtype=reference.dtype,
        build_leaf_maps=True,
    )
    assert isinstance(dense_coupling, MLFMMCouplingOperator)
    rng = np.random.default_rng(68)
    x = np.asarray(
        rng.normal(size=len(reference.positions) * 6)
        + 1j * rng.normal(size=len(reference.positions) * 6),
        dtype=reference.dtype,
    )
    for action, expected_action in (
        (coupling.apply, dense_coupling.apply),
        (coupling.apply_adjoint, dense_coupling.apply_adjoint),
    ):
        actual, expected = action(x), expected_action(x)
        tolerance = 2e-6 if reference.dtype == np.dtype(np.complex64) else 1e-11
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance


@pytest.mark.gpu
def test_absorbing_hierarchy_cupy_actions_and_solve(cupy_runtime, absorbing_hierarchy):
    reference, coupling, particles, options, far_targets, dense = absorbing_hierarchy
    # Sampled cases upload either dense or on-the-fly maps explicitly below;
    # the depth-1 case exercises public MLFMM preparation's exact fallback.
    device = prepare_matvec(
        lmax=1,
        k=1.33 + 0.1j,
        n_medium=1.33 + 0.1j,
        particles=particles,
        operator_dtype=reference.dtype,
        radial_lut_dr=2 * np.pi / 8192,
        backend="cupy",
        coupling_backend="pairwise" if isinstance(coupling, MLFMMCouplingOperator) else "mlfmm",
        mlfmm_options=options,
    )
    if isinstance(coupling, MLFMMCouplingOperator):
        runtime = prepare_mlfmm_cupy_coupling(
            coupling,
            host_cache_policy=CuPyMLFMMHostCachePolicy(
                leaf_apply_mode="dense" if dense else "on_the_fly"
            ),
        )
        device = replace(device, coupling=runtime)
    _check_hierarchy(reference, device, far_targets, "cupy")


@pytest.mark.parametrize("absorbing_hierarchy", [(3, True, "complex128")], indirect=True)
def test_absorbing_hierarchy_lsqr_with_general_dense_particle_responses(absorbing_hierarchy):
    reference, coupling, _particles, _options, _targets, _dense = absorbing_hierarchy
    rng = np.random.default_rng(69)
    # A general T does not commute with W: A^H must apply W^H T^H in that order.
    matrix = 1e-4 * (rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    prepared = prepare_matvec(
        lmax=1,
        k=1.33 + 0.1j,
        n_medium=1.33 + 0.1j,
        particles=[
            TMatrixParticle(position=tuple(position), radius=0.04, lmax=1, t_matrix=matrix)
            for position in reference.positions
        ],
        radial_lut_dr=2 * np.pi / 8192,
    )
    prepared = replace(prepared, coupling=coupling)
    expected = rng.normal(size=len(reference.positions) * 6) + 1j * rng.normal(
        size=len(reference.positions) * 6
    )
    rhs = prepared.apply_A(expected)
    result = solve_linear_system(
        prepared.apply_A,
        rhs,
        A_h_mv=prepared.make_adjoint(),
        method="lsqr",
        rtol=1e-10,
        show_progress=False,
    )
    assert result.info == 0
    assert np.linalg.norm(result.x - expected) / np.linalg.norm(expected) < 1e-8
