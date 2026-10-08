from __future__ import annotations

import pickle
from typing import Any, Literal, cast

import numpy as np
import pytest

import pyceles as pcl
from pyceles._optional import asnumpy, import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    AdjointCouplingOperator,
    CuPyMLFMMCouplingOperator,
    CuPyMLFMMHostCachePolicy,
    MLFMMCouplingOperator,
    MLFMMOptions,
    build_mlfmm_cupy_host_cache,
    prepare_matvec,
    prepare_mlfmm_coupling,
    prepare_mlfmm_cupy_coupling,
    prepare_mlfmm_cupy_data,
)
from pyceles.core.operators.mlfmm_cupy import (
    _box_outgoing_to_directional_cupy,
    _combine_near_far_outputs,
    _directional_to_box_regular_cupy,
    _upload_directional_transforms,
    _upload_offset_batches,
)
from pyceles.core.operators.mlfmm_directional import (
    box_outgoing_to_directional,
    directional_to_box_regular,
    directional_transforms,
)
from pyceles.core.particles import Particle, ParticleCollection, spheres_from_arrays
from pyceles.core.translation import RadialLUT
from pyceles.io import far_field_intensity
from pyceles.postprocessing.farfield import (
    compute_far_field_patterns,
    local_absorbed_power_from_exciting,
    local_absorption_cross_section_from_exciting,
    local_power_balance_from_exciting,
)

pytestmark = pytest.mark.gpu


def _required_float(value: float | None) -> float:
    assert value is not None
    return value


def _small_cluster_particles() -> ParticleCollection:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [220.0, 25.0, -60.0],
            [-180.0, 90.0, 70.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 82.0, 79.0], dtype=float)
    n_particle = np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128)
    return spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )


def _random_uniform_sphere_particles(*, n_particles: int, seed: int = 4) -> ParticleCollection:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1000.0, 1000.0, size=(int(n_particles), 3))
    radii = np.full((positions.shape[0],), 20.0, dtype=float)
    n_particle = np.full((positions.shape[0],), 1.59 + 0.0j, dtype=np.complex128)
    return spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )


@pytest.mark.parametrize(
    ("lmax", "compute_dtype"),
    (
        (2, np.complex64),  # 16 modes in one 32-thread block
        (3, np.complex64),  # common 30-mode case in one 32-thread block
        (2, np.complex128),
        (4, np.complex128),  # 48 modes in two 32-thread blocks
        (5, np.complex64),  # 70 modes in two 64-thread blocks
    ),
)
def test_cupy_pairwise_scalar_mode_blocks_match_numpy(
    lmax: int, compute_dtype: type[np.complexfloating[Any, Any]]
) -> None:
    """The scalar-mode production blocks agree with the NumPy reference."""

    particles = _small_cluster_particles()
    kwargs: dict[str, Any] = dict(
        lmax=lmax,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=compute_dtype,
        coupling_backend="pairwise",
    )
    prepared_numpy = prepare_matvec(**kwargs, backend="numpy")
    prepared_cupy = prepare_matvec(**kwargs, backend="cupy")
    rng = np.random.default_rng(20260827)
    x = np.asarray(
        rng.standard_normal(len(particles) * n_modes(lmax))
        + 1j * rng.standard_normal(len(particles) * n_modes(lmax)),
        dtype=compute_dtype,
    )
    reference = np.asarray(prepared_numpy.apply_W(x), dtype=np.complex128)
    actual = np.asarray(asnumpy(prepared_cupy.apply_W(x)), dtype=np.complex128)
    tolerance = 2.0e-5 if compute_dtype == np.complex64 else 1.0e-12
    np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)

    x_block = np.column_stack((x, (0.5 + 0.25j) * x))
    reference_block = np.column_stack(
        [
            np.asarray(prepared_numpy.apply_W(x_block[:, column]), dtype=np.complex128)
            for column in range(x_block.shape[1])
        ]
    )
    actual_block = np.asarray(asnumpy(prepared_cupy.apply_W(x_block)), dtype=np.complex128)
    np.testing.assert_allclose(actual_block, reference_block, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("compute_dtype", (np.complex64, np.complex128))
def test_cupy_pairwise_single_particle_has_zero_coupling(compute_dtype) -> None:
    particles = spheres_from_arrays(
        positions=np.zeros((1, 3), dtype=float),
        radii=np.asarray([20.0]),
        refractive_indices=np.asarray([1.59 + 0.0j]),
    )
    prepared = prepare_matvec(
        lmax=2,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=compute_dtype,
        coupling_backend="pairwise",
        backend="cupy",
    )
    rng = np.random.default_rng(20261007)
    x = np.asarray(
        rng.standard_normal(n_modes(2)) + 1j * rng.standard_normal(n_modes(2)),
        dtype=compute_dtype,
    )
    np.testing.assert_array_equal(asnumpy(prepared.apply_W(x)), np.zeros_like(x))
    coupling = cast(AdjointCouplingOperator, prepared.coupling)
    np.testing.assert_array_equal(asnumpy(coupling.apply_adjoint(x)), np.zeros_like(x))


@pytest.mark.parametrize(
    ("compute_dtype", "tolerance"),
    ((np.complex64, 5.0e-6), (np.complex128, 1.0e-12)),
)
@pytest.mark.parametrize("lmax", (1, 5))
@pytest.mark.parametrize("nrhs", (1, 3))
def test_cupy_pairwise_adjoint_matches_inner_product(
    compute_dtype: type[np.complexfloating[Any, Any]],
    tolerance: float,
    lmax: int,
    nrhs: int,
) -> None:
    particles = _small_cluster_particles()
    kwargs: dict[str, Any] = dict(
        lmax=lmax,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=compute_dtype,
        coupling_backend="pairwise",
    )
    prepared = prepare_matvec(**kwargs, backend="cupy")
    rng = np.random.default_rng(20260912)
    n = len(particles) * n_modes(lmax)
    shape = (n,) if nrhs == 1 else (n, nrhs)
    x = np.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=compute_dtype
    )
    y = np.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=compute_dtype
    )
    coupling = cast(AdjointCouplingOperator, prepared.coupling)
    lhs = np.vdot(asnumpy(coupling.apply(x)), y)
    rhs = np.vdot(x, asnumpy(coupling.apply_adjoint(y)))
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0) < tolerance


@pytest.mark.parametrize("compute_dtype", (np.complex64, np.complex128))
def test_cupy_parity_translation_preserves_general_dense_t_blocks(compute_dtype) -> None:
    """Translation parity must not impose particle-T polarization or m symmetry."""
    lmax = 3
    nm = n_modes(lmax)
    rng = np.random.default_rng(20261003)
    particles = tuple(
        pcl.TMatrixParticle(
            position=position,
            radius=30.0,
            lmax=lmax,
            t_matrix=(
                0.01 * (rng.standard_normal((nm, nm)) + 1j * rng.standard_normal((nm, nm)))
            ).astype(compute_dtype),
        )
        for position in ((0.0, 0.0, 0.0), (220.0, 25.0, -60.0), (-180.0, 90.0, 70.0))
    )
    kwargs = dict(
        lmax=lmax,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=compute_dtype,
    )
    reference = prepare_matvec(**kwargs, backend="numpy")
    device = prepare_matvec(**kwargs, backend="cupy")
    x = (rng.standard_normal((3 * nm, 3)) + 1j * rng.standard_normal((3 * nm, 3))).astype(
        compute_dtype
    )
    tolerance = 2e-5 if compute_dtype == np.complex64 else 2e-12
    for actual, expected in (
        (device.apply_A(x), np.column_stack([reference.apply_A(column) for column in x.T])),
        (
            device.apply_adjoint(x),
            np.column_stack([reference.apply_adjoint(column) for column in x.T]),
        ),
    ):
        np.testing.assert_allclose(asnumpy(actual), expected, rtol=tolerance, atol=tolerance)


def _plane_wave_source(wavelength: float, n_medium: complex) -> pcl.PlaneWave:
    return pcl.PlaneWave(
        wavelength=wavelength,
        medium_n=n_medium,
        polarization="TE",
        polar_angle=0.3,
        azimuthal_angle=0.2,
        amplitude=1.0,
    )


def _mixed_cluster_particles() -> tuple[Particle, ...]:
    return (
        pcl.Sphere(
            position=(-220.0, 0.0, -40.0),
            radius=70.0,
            refractive_index=1.52 + 0.01j,
        ),
        pcl.LayeredSphere(
            position=(40.0, 0.0, 30.0),
            layer_radii=(45.0, 85.0),
            layer_refractive_indices=(1.35 + 0.0j, 1.68 + 0.02j),
        ),
        pcl.Spheroid(
            position=(250.0, 0.0, -20.0),
            equatorial_radius=60.0,
            polar_radius=95.0,
            refractive_index=1.47 + 0.03j,
            euler_angles=(0.1, 0.35, -0.2),
        ),
    )


def test_cupy_mlfmm_directional_upload_uses_structured_factors() -> None:
    cupy, _ = import_cupy()
    transforms = directional_transforms(5, grid_order=6)
    uploaded = _upload_directional_transforms(transforms, cupy=cupy)

    assert not hasattr(uploaded, "forward_F")
    assert not hasattr(uploaded, "inverse_A_adj")
    assert tuple(uploaded.fth_reflected_beta.shape) == (
        transforms.grid.beta.size,
        transforms.Fth.shape[1],
    )
    assert tuple(uploaded.fph_reflected_beta.shape) == (
        transforms.grid.beta.size,
        transforms.Fph.shape[1],
    )
    dense_bytes = int(transforms.Fth.nbytes + transforms.Fph.nbytes)
    structured_bytes = int(
        uploaded.fth_reflected_beta.nbytes
        + uploaded.fph_reflected_beta.nbytes
        + uploaded.phase_by_m.nbytes
        + uploaded.m_of_scalar.nbytes
    )
    assert structured_bytes < dense_bytes // 4


def test_cupy_mlfmm_structured_directional_maps_match_dense_reference() -> None:
    cupy, _ = import_cupy()
    transforms = directional_transforms(4, grid_order=5)
    uploaded = _upload_directional_transforms(transforms, cupy=cupy)
    rng = np.random.default_rng(20260603)
    state = np.asarray(
        rng.standard_normal((2, n_modes(4), 3)) + 1j * rng.standard_normal((2, n_modes(4), 3)),
        dtype=np.complex128,
    )

    got = asnumpy(_box_outgoing_to_directional_cupy(uploaded, cupy.asarray(state), cupy=cupy))
    for batch in range(state.shape[0]):
        for rhs in range(state.shape[2]):
            expected = box_outgoing_to_directional(transforms, state[batch, :, rhs])
            for channel in range(4):
                np.testing.assert_allclose(
                    got[batch, channel, :, rhs],
                    expected[channel],
                    rtol=2.0e-12,
                    atol=2.0e-12,
                )

    channels = np.asarray(
        rng.standard_normal(got.shape) + 1j * rng.standard_normal(got.shape),
        dtype=np.complex128,
    )
    got_box = asnumpy(_directional_to_box_regular_cupy(uploaded, cupy.asarray(channels), cupy=cupy))
    for batch in range(channels.shape[0]):
        for rhs in range(channels.shape[3]):
            expected_box = directional_to_box_regular(
                transforms,
                channels[batch, 0, :, rhs],
                channels[batch, 1, :, rhs],
                channels[batch, 2, :, rhs],
                channels[batch, 3, :, rhs],
            )
            np.testing.assert_allclose(
                got_box[batch, :, rhs],
                expected_box,
                rtol=2.0e-12,
                atol=2.0e-12,
            )


def test_cupy_mlfmm_upload_offset_batches_rejects_nonunique() -> None:
    cupy, _ = import_cupy()
    with pytest.raises(ValueError, match="violates grouped uniqueness contract"):
        _upload_offset_batches(
            {(0, 0, 0): (np.array([1, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64))},
            cupy=cupy,
            name="test_batches",
        )
    with pytest.raises(ValueError, match="violates grouped uniqueness contract"):
        _upload_offset_batches(
            {(0, 0, 0): (np.array([1, 2], dtype=np.int64), np.array([3, 3], dtype=np.int64))},
            cupy=cupy,
            name="test_batches",
        )


def test_local_absorption_cross_section_from_exciting_accepts_cupy_arrays() -> None:
    cupy, _ = import_cupy()
    source = _plane_wave_source(550.0, 1.0 + 0j)
    rng = np.random.default_rng(20260411)
    e_np = np.asarray(
        rng.standard_normal(48) + 1j * rng.standard_normal(48),
        dtype=np.complex128,
    )
    x_np = np.asarray(
        rng.standard_normal(48) + 1j * rng.standard_normal(48),
        dtype=np.complex128,
    )
    ref = local_absorption_cross_section_from_exciting(
        source,
        e_np,
        x_np,
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
    )
    got = local_absorption_cross_section_from_exciting(
        source,
        cupy.asarray(e_np),
        cupy.asarray(x_np),
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_local_absorbed_power_from_exciting_accepts_cupy_arrays() -> None:
    cupy, _ = import_cupy()
    rng = np.random.default_rng(20260412)
    e_np = np.asarray(
        rng.standard_normal(48) + 1j * rng.standard_normal(48),
        dtype=np.complex128,
    )
    x_np = np.asarray(
        rng.standard_normal(48) + 1j * rng.standard_normal(48),
        dtype=np.complex128,
    )
    ref = local_absorbed_power_from_exciting(
        e_np,
        x_np,
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
    )
    got = local_absorbed_power_from_exciting(
        cupy.asarray(e_np),
        cupy.asarray(x_np),
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
    )
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)


def test_local_power_balance_from_exciting_accepts_cupy_arrays() -> None:
    cupy, _ = import_cupy()
    rng = np.random.default_rng(20260413)
    n_particles = 4
    nmodes = 6
    e_np = np.asarray(
        rng.standard_normal((n_particles, nmodes))
        + 1j * rng.standard_normal((n_particles, nmodes)),
        dtype=np.complex128,
    )
    x_np = np.asarray(
        rng.standard_normal((n_particles, nmodes))
        + 1j * rng.standard_normal((n_particles, nmodes)),
        dtype=np.complex128,
    )
    ref = local_power_balance_from_exciting(
        e_np,
        x_np,
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
        n_particles=n_particles,
        nmodes_per_particle=nmodes,
    )
    got = local_power_balance_from_exciting(
        cupy.asarray(e_np),
        cupy.asarray(x_np),
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
        n_particles=n_particles,
        nmodes_per_particle=nmodes,
    )
    np.testing.assert_allclose(
        _required_float(got.local_absorbed_power),
        _required_float(ref.local_absorbed_power),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(got.local_absorbed_power_per_particle, dtype=float),
        np.asarray(ref.local_absorbed_power_per_particle, dtype=float),
        rtol=1e-13,
        atol=1e-13,
    )


def _policy_numpy_mlfmm_coupling() -> MLFMMCouplingOperator:
    prepared = prepare_matvec(
        lmax=3,
        k=2.0 * np.pi / 550.0,
        particles=_mlfmm_policy_particles(),
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(
            max_leaf_particles=4,
            max_depth=4,
            accuracy_level=2,
            order_additive=1,
        ),
        backend="numpy",
    )
    coupling = prepared.coupling
    if not isinstance(coupling, MLFMMCouplingOperator):
        raise AssertionError("Transition fixture unexpectedly resolved to direct stage.")
    return coupling


@pytest.fixture(scope="module")
def _policy_coupling_fixture() -> MLFMMCouplingOperator:
    return _policy_numpy_mlfmm_coupling()


def test_cupy_mlfmm_host_cache_recomputes_static_tables_during_upload(
    _policy_coupling_fixture: MLFMMCouplingOperator,
) -> None:
    coupling = _policy_coupling_fixture
    policy = CuPyMLFMMHostCachePolicy()
    host_cache = build_mlfmm_cupy_host_cache(coupling, host_cache_policy=policy)
    assert host_cache.near_plm_coeffs is None
    assert host_cache.near_compact_re_ab is None
    assert host_cache.near_compact_im_ab is None
    assert host_cache.near_mode_m is None
    assert host_cache.near_pair_offset is None
    assert host_cache.near_pair_pmin is None
    assert host_cache.near_pair_pcount is None

    prepared = prepare_mlfmm_cupy_data(host_cache)
    assert prepared.stage in {"single_level", "multilevel"}
    assert int(prepared.near_pairs.mode_m.size) > 0
    assert int(prepared.near_pairs.pair_offset.size) > 0


def test_cupy_host_cache_build_supports_sparse_cpu_staging_without_dense_leaf_maps() -> None:
    particles = _mlfmm_policy_particles()
    positions = particles.positions
    radii = particles.circumscribing_radii
    k = 2.0 * np.pi / 550.0
    radial_lut = RadialLUT(
        lmax=3,
        k=k,
        r_max=float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2))),
        dr=0.5,
        dtype=np.complex128,
    )
    coupling = prepare_mlfmm_coupling(
        lmax=3,
        k=k,
        positions=positions,
        particle_circumscribing_radii=radii,
        radial_lut=radial_lut,
        options=MLFMMOptions(max_leaf_particles=4, max_depth=4),
        dtype=np.complex128,
        cache_translation_blocks=False,
        leaf_map_backend="cupy",
        build_leaf_maps=False,
    )
    if not isinstance(coupling, MLFMMCouplingOperator):
        raise AssertionError("Transition fixture unexpectedly resolved to direct stage.")
    if coupling.single_level is not None:
        assert len(coupling.single_level.aggregation) == 0
        assert len(coupling.single_level.receive) == 0
    if coupling.multilevel is not None:
        assert len(coupling.multilevel.aggregation) == 0
        assert len(coupling.multilevel.receive) == 0

    policy = CuPyMLFMMHostCachePolicy(leaf_apply_mode="on_the_fly")
    host_cache = build_mlfmm_cupy_host_cache(coupling, host_cache_policy=policy)
    if host_cache.single_level is not None:
        assert host_cache.single_level.aggregation is None
    if host_cache.multilevel is not None:
        assert host_cache.multilevel.aggregation is None
    prepared = prepare_mlfmm_cupy_data(host_cache)
    assert prepared.stage in {"single_level", "multilevel"}


def test_cupy_mlfmm_runtime_operator_is_non_picklable() -> None:
    prepared = prepare_matvec(
        lmax=3,
        k=2.0 * np.pi / 550.0,
        particles=_mlfmm_policy_particles(),
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=4, max_depth=4),
        backend="cupy",
    )
    coupling = prepared.coupling
    if not isinstance(coupling, CuPyMLFMMCouplingOperator):
        raise AssertionError("Transition fixture unexpectedly resolved to direct CuPy fallback.")
    with pytest.raises(TypeError, match="non-picklable"):
        _ = pickle.dumps(coupling)


def test_cupy_prepare_coupling_detaches_host_cache_by_default(
    _policy_coupling_fixture: MLFMMCouplingOperator,
) -> None:
    coupling = _policy_coupling_fixture
    runtime = prepare_mlfmm_cupy_coupling(coupling)
    assert runtime.host_cache is None
    assert runtime.host_cache_summary is not None
    assert runtime.host_cache_summary.get("retained") is False
    payload_obj = runtime.host_cache_summary.get("numpy_payload_bytes_estimate", 0)
    assert isinstance(payload_obj, (int, np.integer))
    payload = int(payload_obj)
    assert payload > 0
    hierarchy = runtime.hierarchy_diagnostics()
    assert hierarchy.get("stage") == str(coupling.resolved_plan.stage)
    levels = hierarchy.get("levels")
    assert isinstance(levels, dict)
    level_rows = levels.get("levels")
    assert isinstance(level_rows, list)
    assert level_rows
    assert all("box_order" in row and "n_directions" in row for row in level_rows)


def test_cupy_prepare_coupling_can_retain_host_cache_when_requested(
    _policy_coupling_fixture: MLFMMCouplingOperator,
) -> None:
    coupling = _policy_coupling_fixture
    runtime = prepare_mlfmm_cupy_coupling(
        coupling,
        host_cache_policy=CuPyMLFMMHostCachePolicy(host_cache_retention="full"),
    )
    assert runtime.host_cache is not None
    assert runtime.host_cache_summary is not None
    assert runtime.host_cache_summary.get("retained") is True


def test_cupy_prepare_coupling_can_drop_host_cache_summary_when_requested(
    _policy_coupling_fixture: MLFMMCouplingOperator,
) -> None:
    coupling = _policy_coupling_fixture
    runtime = prepare_mlfmm_cupy_coupling(
        coupling,
        host_cache_policy=CuPyMLFMMHostCachePolicy(host_cache_retention="none"),
    )
    assert runtime.host_cache is None
    assert runtime.host_cache_summary is None


def test_cupy_prepare_coupling_rejects_nonpositive_leaf_otf_chunk_leaves(
    _policy_coupling_fixture: MLFMMCouplingOperator,
) -> None:
    coupling = _policy_coupling_fixture
    with pytest.raises(ValueError, match="leaf_otf_chunk_leaves must be positive"):
        _ = prepare_mlfmm_cupy_coupling(
            coupling,
            host_cache_policy=CuPyMLFMMHostCachePolicy(leaf_otf_chunk_leaves=0),
        )


def test_cupy_prepare_coupling_rejects_nonpositive_leaf_otf_bytes_budget(
    _policy_coupling_fixture: MLFMMCouplingOperator,
) -> None:
    coupling = _policy_coupling_fixture
    with pytest.raises(ValueError, match="leaf_otf_bytes_budget must be positive"):
        _ = prepare_mlfmm_cupy_coupling(
            coupling,
            host_cache_policy=CuPyMLFMMHostCachePolicy(leaf_otf_bytes_budget=0),
        )


def test_cupy_prepare_coupling_rejects_nonpositive_streamed_far_chunk_bytes_budget(
    _policy_coupling_fixture: MLFMMCouplingOperator,
) -> None:
    coupling = _policy_coupling_fixture
    with pytest.raises(ValueError, match="streamed_far_chunk_bytes_budget must be positive"):
        _ = prepare_mlfmm_cupy_coupling(
            coupling,
            host_cache_policy=CuPyMLFMMHostCachePolicy(streamed_far_chunk_bytes_budget=0),
        )


def _mlfmm_transition_particles() -> ParticleCollection:
    # 27 particles (seed=50) is the smallest deterministic fixture we found
    # that reliably resolves to the multilevel stage with max_leaf_particles=4.
    return _random_uniform_sphere_particles(n_particles=27, seed=50)


def _mlfmm_single_level_particles() -> ParticleCollection:
    # A smaller fixture still resolves to single_level with the same leaf
    # threshold used by the multilevel transition test.
    return _random_uniform_sphere_particles(n_particles=8, seed=45)


def _mlfmm_adjoint_depth_particles() -> ParticleCollection:
    """Sixteen tiny particles populate eight separated leaves for depth probes."""

    corners = np.asarray(
        [
            [x, y, z]
            for x in (-1000.0, 1000.0)
            for y in (-1000.0, 1000.0)
            for z in (-1000.0, 1000.0)
        ],
        dtype=float,
    )
    positions = np.vstack((corners, corners + np.array([1.0, 1.0, 1.0])))
    return spheres_from_arrays(
        positions=positions,
        radii=np.full((positions.shape[0],), 1.0),
        refractive_indices=np.full((positions.shape[0],), 1.59 + 0.0j, dtype=np.complex128),
    )


def _transition_numpy_mlfmm_coupling(*, max_leaf_particles: int) -> MLFMMCouplingOperator:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=_mlfmm_transition_particles(),
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=max_leaf_particles, max_depth=4),
        backend="numpy",
    )
    coupling = prepared.coupling
    if not isinstance(coupling, MLFMMCouplingOperator):
        raise AssertionError("Transition fixture unexpectedly resolved to direct stage.")
    return coupling


@pytest.fixture(scope="module")
def _transition_multilevel_coupling() -> MLFMMCouplingOperator:
    return _transition_numpy_mlfmm_coupling(max_leaf_particles=4)


def _mlfmm_policy_particles() -> ParticleCollection:
    # Policy/retention tests do not require a stage split; keep this fixture
    # small so host-cache/policy tests stay lightweight.
    return _random_uniform_sphere_particles(n_particles=24, seed=4)


def _sim_cfg(
    *,
    operator_backend: Literal["numpy", "cupy"],
    compute_dtype: Literal["complex64", "complex128"],
    wavelength: float,
    n_medium: complex,
) -> pcl.SimulationConfig:
    return pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=3,
        # Backend parity needs representative forward/backward directions,
        # not production-resolution far-field output.
        polar_angles=pcl.core.uniform_polar_grid(61),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(24),
        radial_lut_dr=0.5,
        solver_method="gmres",
        solver_rtol=1e-10 if compute_dtype == "complex128" else 1e-8,
        solver_restart=10,
        solver_maxiter=120,
        operator_backend=operator_backend,
        compute_dtype=compute_dtype,
        accum_dtype="complex128",
        verbose=False,
    )


def test_cupy_local_absorption_postprocess_uses_backend_coefficients(monkeypatch) -> None:
    import pyceles.simulation.postprocess as simulation_postprocess

    cupy, _ = import_cupy()
    seen_backend_payloads: list[tuple[bool, bool]] = []
    build_payload_real = simulation_postprocess._build_exciting_scattered_flat_generic_route

    def _build_payload_wrapped(
        sim: Any,
        *,
        initial_coeffs: Any,
        coeffs: Any,
        accum_dtype: np.dtype,
    ) -> Any:
        seen_backend_payloads.append(
            (
                isinstance(initial_coeffs, cupy.ndarray),
                isinstance(coeffs, cupy.ndarray),
            )
        )
        return build_payload_real(
            sim,
            initial_coeffs=initial_coeffs,
            coeffs=coeffs,
            accum_dtype=accum_dtype,
        )

    monkeypatch.setattr(
        simulation_postprocess,
        "_build_exciting_scattered_flat_generic_route",
        _build_payload_wrapped,
    )

    wavelength = 550.0
    source = _plane_wave_source(wavelength, 1.0 + 0j)
    cfg = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=1.0 + 0j,
        lmax=2,
        polar_angles=pcl.core.uniform_polar_grid(91),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(24),
        radial_lut_dr=0.5,
        solver_method="direct",
        operator_backend="cupy",
        postprocessing_backend="cupy",
        compute_dtype="complex128",
        accum_dtype="complex128",
        verbose=False,
    )
    particles = (
        pcl.Sphere(
            position=(0.0, 0.0, 0.0),
            radius=60.0,
            refractive_index=1.5 + 0.01j,
        ),
    )

    sim = pcl.Simulation(cfg, particles=particles)
    run = sim.run(source, include_farfield=True)

    assert run.cross_sections is not None
    assert seen_backend_payloads
    assert seen_backend_payloads[-1] == (False, True)
    assert not hasattr(run.solver_result, "backend_x")
    assert not hasattr(sim, "_solve_backend_handoffs")


def test_cupy_public_solve_sources_does_not_retain_backend_handoff() -> None:
    wavelength = 550.0
    source = _plane_wave_source(wavelength, 1.0 + 0j)
    cfg = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=1.0 + 0j,
        lmax=2,
        radial_lut_dr=0.5,
        solver_method="direct",
        operator_backend="cupy",
        compute_dtype="complex128",
        accum_dtype="complex128",
        verbose=False,
    )
    particles = (
        pcl.Sphere(
            position=(0.0, 0.0, 0.0),
            radius=60.0,
            refractive_index=1.5 + 0.01j,
        ),
    )
    sim = pcl.Simulation(cfg, particles=particles)

    solved = sim.solve_sources({"mixed": source})

    assert not hasattr(solved.solver_result, "backend_x")
    assert not hasattr(sim, "_solve_backend_handoffs")


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
def test_cupy_angular_spectrum_farfield_uses_canonical_wavevector_grid(dtype) -> None:
    """CuPy far-field grids remain combinable with angular-spectrum sources."""
    wavelength = 550.0
    k = 2.0 * np.pi / wavelength
    alpha = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    beta = np.linspace(0.0, np.pi, 9)
    source = pcl.GaussianBeam(
        wavelength=wavelength,
        medium_n=1.0 + 0j,
        polarization="TE",
        beam_width=2000.0,
    )

    patterns = compute_far_field_patterns(
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        np.zeros((1, n_modes(1)), dtype=dtype),
        k=k,
        lmax=1,
        polar_angles=beta,
        azimuthal_angles=alpha,
        source=source,
        backend="cupy",
        dtype=dtype,
    )

    expected_kx = k * np.sin(beta[None, :]) * np.cos(alpha[:, None])
    expected_ky = k * np.sin(beta[None, :]) * np.sin(alpha[:, None])
    expected_kz = np.broadcast_to(k * np.cos(beta), expected_kx.shape)
    assert patterns.initial is not None
    assert patterns.total is not None
    np.testing.assert_array_equal(patterns.scattered.kx, expected_kx)
    np.testing.assert_array_equal(patterns.scattered.ky, expected_ky)
    np.testing.assert_array_equal(patterns.scattered.kz, expected_kz)
    np.testing.assert_array_equal(patterns.initial.kx, patterns.scattered.kx)
    np.testing.assert_array_equal(patterns.initial.ky, patterns.scattered.ky)
    np.testing.assert_array_equal(patterns.initial.kz, patterns.scattered.kz)


@pytest.mark.parametrize("collect_stream_stats", [False, True])
def test_cupy_multilevel_stream_stats_collection_is_opt_in(
    _transition_multilevel_coupling: MLFMMCouplingOperator,
    collect_stream_stats: bool,
) -> None:
    coupling = _transition_multilevel_coupling
    assert str(coupling.resolved_plan.stage) == "multilevel"
    runtime = prepare_mlfmm_cupy_coupling(
        coupling,
        host_cache_policy=CuPyMLFMMHostCachePolicy(collect_stream_stats=collect_stream_stats),
    )

    diag = runtime.memory_diagnostics()
    streaming = diag.get("multilevel_streaming")
    rolling = diag.get("multilevel_rolling")
    assert isinstance(streaming, dict)
    assert isinstance(rolling, dict)
    if not collect_stream_stats:
        assert streaming.get("collect_stream_stats") is False
        assert streaming.get("last_apply_stats") is None
        assert streaming.get("last_apply_timing_seconds") is None
        return

    nm = n_modes(1)
    n_particles = int(coupling.positions.shape[0])
    rng = np.random.default_rng(20260409)
    x = np.asarray(
        rng.standard_normal(n_particles * nm) + 1j * rng.standard_normal(n_particles * nm),
        dtype=np.complex128,
    )
    _ = asnumpy(runtime.apply(x))
    diag = runtime.memory_diagnostics()
    streaming = diag.get("multilevel_streaming")
    rolling = diag.get("multilevel_rolling")
    assert isinstance(streaming, dict)
    assert isinstance(rolling, dict)
    device_pool = diag.get("device_pool")
    device_mem_info = diag.get("device_mem_info")
    assert isinstance(streaming, dict)
    assert isinstance(rolling, dict)
    assert isinstance(device_pool, dict)
    assert isinstance(device_mem_info, dict)
    assert device_mem_info["effective_free_bytes"] >= device_mem_info["free_bytes"]
    assert device_pool["cached_bytes"] == device_pool["total_bytes"] - device_pool["used_bytes"]
    assert streaming.get("collect_stream_stats") is collect_stream_stats
    chunk_box_cap = streaming.get("resolved_streamed_far_chunk_box_cap")
    frontier_box_cap = streaming.get("resolved_streamed_far_frontier_box_cap")
    leaf_chunk_min = streaming.get("resolved_leaf_otf_chunk_leaves_min")
    leaf_chunk_max = streaming.get("resolved_leaf_otf_chunk_leaves_max")
    leaf_chunk_by_occupancy = streaming.get("resolved_leaf_otf_chunk_leaves_by_occupancy")
    device_limit = streaming.get("device_limit_bytes")
    pool_used_before_plan = streaming.get("pool_used_bytes_before_stream_plan")
    pool_trimmed_before_plan = streaming.get("pool_trimmed_before_stream_plan")
    pool_trimmed_for_fragmentation = streaming.get("pool_trimmed_for_fragmentation")
    transient_budget = streaming.get("stream_transient_budget_bytes")
    unmodeled_temp_reserve = streaming.get("stream_unmodeled_temp_reserve_bytes")
    chunk_bytes_budget = streaming.get("resolved_streamed_far_chunk_bytes_budget")
    frontier_bytes_budget = streaming.get("resolved_streamed_far_frontier_bytes_budget")
    largest_stream_allocation = streaming.get("largest_single_stream_allocation_bytes")
    guaranteed_fresh_allocation = streaming.get("guaranteed_fresh_allocation_bytes")
    fragmentation_guard = streaming.get("stream_fragmentation_guard_bytes")
    assert isinstance(chunk_box_cap, int)
    assert isinstance(frontier_box_cap, int)
    assert isinstance(leaf_chunk_min, int)
    assert isinstance(leaf_chunk_max, int)
    assert isinstance(leaf_chunk_by_occupancy, dict)
    assert isinstance(device_limit, int)
    assert isinstance(pool_used_before_plan, int)
    assert isinstance(pool_trimmed_before_plan, bool)
    assert isinstance(pool_trimmed_for_fragmentation, bool)
    assert isinstance(transient_budget, int)
    assert isinstance(unmodeled_temp_reserve, int)
    assert isinstance(chunk_bytes_budget, int)
    assert isinstance(frontier_bytes_budget, int)
    assert isinstance(largest_stream_allocation, int)
    assert isinstance(guaranteed_fresh_allocation, int)
    assert isinstance(fragmentation_guard, int)
    assert chunk_box_cap > 0
    assert frontier_box_cap > 0
    assert leaf_chunk_min > 0
    assert leaf_chunk_max >= leaf_chunk_min
    assert leaf_chunk_by_occupancy
    assert device_limit > 0
    assert 0 <= pool_used_before_plan <= device_limit
    assert transient_budget > 0
    assert unmodeled_temp_reserve >= 0
    assert chunk_bytes_budget > 0
    assert frontier_bytes_budget > 0
    assert largest_stream_allocation > 0
    assert guaranteed_fresh_allocation >= largest_stream_allocation
    assert fragmentation_guard >= 0
    assert chunk_bytes_budget + frontier_bytes_budget <= transient_budget
    assert pool_used_before_plan + transient_budget + unmodeled_temp_reserve <= device_limit
    assert "resolved_leaf_otf_bytes_budget" not in streaming
    assert "resolved_streamed_far_safety_reserve_bytes" not in streaming
    assert rolling.get("execution_mode") == "streamed_chunk_local"
    assert rolling.get("rolling_incoming_bytes_actual_peak") is None
    assert rolling.get("rolling_outgoing_bytes_actual_peak") is None
    assert rolling.get("incoming_reduction_ratio") is None
    assert rolling.get("outgoing_reduction_ratio") is None
    assert rolling.get("rolling_far_hierarchy_bytes") is None
    if collect_stream_stats:
        stats = streaming.get("last_apply_stats")
        assert isinstance(stats, dict)
        assert "processed_chunk_count" in stats
        assert "timings_seconds_by_level" in stats
        timings = streaming.get("last_apply_timing_seconds")
        assert isinstance(timings, dict)
        assert timings["total"] >= timings["sampled_far"] >= 0.0
        assert timings["total"] >= timings["exact_near"] >= 0.0
        union_stats = stats.get("same_level_source_union_stats")
        assert union_stats is None or isinstance(union_stats, dict)
        outgoing_stack_peak = stats.get("outgoing_build_stack_peak_bytes")
        assert isinstance(outgoing_stack_peak, dict)
        assert all(int(value) >= 0 for value in outgoing_stack_peak.values())
        assert int(stats.get("frontier_in_flight_peak_bytes", 0)) >= 0
        pool_peak_total = stats.get("pool_peak_total_bytes")
        assert isinstance(pool_peak_total, int)
        assert pool_peak_total <= device_limit
        assert "_internal_same_level_source_union_history" not in stats
    else:
        assert streaming.get("last_apply_stats") is None
        assert streaming.get("last_apply_timing_seconds") is None


def test_cupy_mlfmm_workspace_cache_keeps_only_active_rhs_shape(
    _transition_multilevel_coupling: MLFMMCouplingOperator,
) -> None:
    """Changing RHS width replaces, rather than accumulates, device workspaces."""

    cupy, _ = import_cupy()
    runtime = prepare_mlfmm_cupy_coupling(_transition_multilevel_coupling)
    nm = n_modes(1)
    n_particles = int(_transition_multilevel_coupling.positions.shape[0])
    rng = np.random.default_rng(20260912)
    for nrhs in (1, 4, 2):
        x = np.asarray(
            rng.standard_normal((n_particles * nm, nrhs))
            + 1j * rng.standard_normal((n_particles * nm, nrhs)),
            dtype=np.complex128,
        )
        _ = runtime.apply(x)
        cupy.cuda.Stream.null.synchronize()
        entries = runtime.memory_diagnostics()["workspace_cache_entries"]
        assert isinstance(entries, dict)
        assert entries["near"] == 1
        assert entries["single_level"] == 0
        assert entries["multilevel"] == 1


def test_cupy_mlfmm_dense_receive_cache_uses_stable_group_identity() -> None:
    """Dense/debug receive caching is stable across repeated adjoint actions."""

    particles = _mlfmm_policy_particles()
    positions = particles.positions
    k = 2.0 * np.pi / 550.0
    radial_lut = RadialLUT(
        lmax=3,
        k=k,
        r_max=float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2))),
        dr=0.5,
        dtype=np.complex128,
    )
    coupling = prepare_mlfmm_coupling(
        lmax=3,
        k=k,
        positions=positions,
        particle_circumscribing_radii=particles.circumscribing_radii,
        radial_lut=radial_lut,
        options=MLFMMOptions(max_leaf_particles=4, max_depth=4, accuracy_level=2, order_additive=1),
        dtype=np.complex128,
        cache_translation_blocks=False,
        leaf_map_backend="cupy",
        build_leaf_maps=True,
    )
    if not isinstance(coupling, MLFMMCouplingOperator):
        raise AssertionError("Dense receive fixture unexpectedly resolved to direct stage.")
    runtime = prepare_mlfmm_cupy_coupling(
        coupling,
        host_cache_policy=CuPyMLFMMHostCachePolicy(leaf_apply_mode="dense"),
    )
    n = int(coupling.positions.shape[0]) * n_modes(3)
    rng = np.random.default_rng(20260913)
    x = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=np.complex128)
    first = asnumpy(runtime.apply_adjoint(x))
    first_diagnostics = runtime.memory_diagnostics()
    second = asnumpy(runtime.apply_adjoint(x))
    second_diagnostics = runtime.memory_diagnostics()
    np.testing.assert_allclose(first, second, rtol=1e-14, atol=1e-14)
    first_entries = first_diagnostics["workspace_cache_entries"]
    second_entries = second_diagnostics["workspace_cache_entries"]
    first_workspace = first_diagnostics["workspace_bytes"]
    second_workspace = second_diagnostics["workspace_bytes"]
    assert isinstance(first_entries, dict)
    assert isinstance(second_entries, dict)
    assert isinstance(first_workspace, dict)
    assert isinstance(second_workspace, dict)
    assert first_entries["leaf_receive"] > 0
    assert second_entries["leaf_receive"] == first_entries["leaf_receive"]
    assert (
        second_workspace["leaf_receive_cache_bytes"] == first_workspace["leaf_receive_cache_bytes"]
    )
    # Resident reverse outputs are fresh allocations; forwarding their values
    # must preserve them just as it preserves streamed action results.
    retained_reverse = runtime.apply_adjoint(x)
    saved_reverse = asnumpy(retained_reverse)
    retained_forward = runtime.apply(retained_reverse)
    saved_forward = asnumpy(retained_forward)
    runtime.apply_adjoint(-x)
    runtime.apply(x)
    np.testing.assert_array_equal(asnumpy(retained_reverse), saved_reverse)
    np.testing.assert_array_equal(asnumpy(retained_forward), saved_forward)


@pytest.mark.parametrize(
    ("operator_dtype", "rtol", "atol"),
    [
        (np.complex64, 2e-5, 2e-6),
        (np.complex128, 1e-12, 1e-12),
    ],
)
def test_cupy_prepared_operator_matches_numpy_for_diagonal_spheres(
    operator_dtype: np.dtype, rtol: float, atol: float
) -> None:
    lmax = 3
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [220.0, 25.0, -60.0],
            [-180.0, 90.0, 70.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 82.0, 79.0], dtype=float)
    n_particle = np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128)
    particles = spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )
    nm = n_modes(lmax)
    rng = np.random.default_rng(7)
    x = np.asarray(
        rng.standard_normal(positions.shape[0] * nm)
        + 1j * rng.standard_normal(positions.shape[0] * nm),
        dtype=operator_dtype,
    )
    b = np.asarray(
        rng.standard_normal(positions.shape[0] * nm)
        + 1j * rng.standard_normal(positions.shape[0] * nm),
        dtype=operator_dtype,
    )

    prepared_numpy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=operator_dtype,
        backend="numpy",
    )
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=operator_dtype,
        backend="cupy",
    )

    np.testing.assert_allclose(
        prepared_cupy.apply_A(x), prepared_numpy.apply_A(x), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        prepared_cupy.rhs_Tb(b), prepared_numpy.rhs_Tb(b), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    ("operator_dtype", "rtol", "atol"),
    [
        (np.complex64, 2e-5, 2e-6),
        (np.complex128, 1e-12, 1e-12),
    ],
)
def test_cupy_prepared_operator_block_rhs_matches_columnwise(
    operator_dtype: np.dtype, rtol: float, atol: float
) -> None:
    lmax = 3
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [220.0, 25.0, -60.0],
            [-180.0, 90.0, 70.0],
        ],
        dtype=float,
    )
    radii = np.array([80.0, 82.0, 79.0], dtype=float)
    n_particle = np.array([1.59 + 0.0j, 1.61 + 0.0j, 1.58 + 0.0j], dtype=np.complex128)
    particles = spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )
    nm = n_modes(lmax)
    rng = np.random.default_rng(17)
    n_unknowns = positions.shape[0] * nm
    x_block = np.asarray(
        rng.standard_normal((n_unknowns, 3)) + 1j * rng.standard_normal((n_unknowns, 3)),
        dtype=operator_dtype,
    )
    b_block = np.asarray(
        rng.standard_normal((n_unknowns, 3)) + 1j * rng.standard_normal((n_unknowns, 3)),
        dtype=operator_dtype,
    )

    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=operator_dtype,
        backend="cupy",
    )

    y_block = np.asarray(prepared_cupy.apply_A(x_block))
    y_cols = np.column_stack([np.asarray(prepared_cupy.apply_A(x_block[:, j])) for j in range(3)])
    np.testing.assert_allclose(y_block, y_cols, rtol=rtol, atol=atol)

    rhs_block = np.asarray(prepared_cupy.rhs_Tb(b_block))
    rhs_cols = np.column_stack([np.asarray(prepared_cupy.rhs_Tb(b_block[:, j])) for j in range(3)])
    np.testing.assert_allclose(rhs_block, rhs_cols, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("max_leaf_particles", "expected_stage"),
    [
        (4, "single_level"),
        (4, "multilevel"),
    ],
)
def test_cupy_mlfmm_prepared_operator_matches_numpy_reference(
    max_leaf_particles: int, expected_stage: str
) -> None:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    particles = (
        _mlfmm_single_level_particles()
        if expected_stage == "single_level"
        else _mlfmm_transition_particles()
    )
    options = MLFMMOptions(max_leaf_particles=max_leaf_particles, max_depth=4)

    prepared_numpy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="numpy",
    )
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="cupy",
    )

    assert isinstance(prepared_numpy.coupling, MLFMMCouplingOperator)
    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    assert str(prepared_numpy.coupling.resolved_plan.stage) == expected_stage
    assert str(prepared_cupy.coupling.prepared_data.stage) == expected_stage

    nm = n_modes(lmax)
    n_particles = len(particles)
    rng = np.random.default_rng(20260323 + int(max_leaf_particles))
    x = np.asarray(
        rng.standard_normal(n_particles * nm) + 1j * rng.standard_normal(n_particles * nm),
        dtype=np.complex128,
    )
    y_numpy = np.asarray(prepared_numpy.apply_W(x), dtype=np.complex128)
    y_cupy = np.asarray(asnumpy(prepared_cupy.apply_W(x)), dtype=np.complex128)
    np.testing.assert_allclose(y_cupy, y_numpy, rtol=1e-10, atol=1e-10)
    adjoint_numpy = np.asarray(prepared_numpy.coupling.apply_adjoint(x), dtype=np.complex128)
    adjoint_cupy = np.asarray(asnumpy(prepared_cupy.coupling.apply_adjoint(x)), dtype=np.complex128)
    np.testing.assert_allclose(adjoint_cupy, adjoint_numpy, rtol=1e-10, atol=1e-10)
    _assert_mlfmm_live_results_survive_composition(prepared_cupy, x)


@pytest.mark.parametrize("expected_stage", ("single_level", "multilevel"))
@pytest.mark.parametrize(
    ("operator_dtype", "tolerance"),
    ((np.complex64, 5.0e-6), (np.complex128, 1.0e-9)),
)
def test_cupy_mlfmm_adjoint_matches_inner_product(
    expected_stage: str,
    operator_dtype: type[np.complexfloating[Any, Any]],
    tolerance: float,
) -> None:
    """CuPy's exact-near and sampled-far MLFMM reverse map is Hermitian-consistent."""

    lmax = 1
    wavelength = 550.0
    particles = (
        _mlfmm_single_level_particles()
        if expected_stage == "single_level"
        else _mlfmm_transition_particles()
    )
    prepared = prepare_matvec(
        lmax=lmax,
        k=2.0 * np.pi / wavelength,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=operator_dtype,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=4, max_depth=4),
        backend="cupy",
    )
    coupling = prepared.coupling
    assert isinstance(coupling, CuPyMLFMMCouplingOperator)
    assert str(coupling.prepared_data.stage) == expected_stage

    n = len(particles) * n_modes(lmax)
    rng = np.random.default_rng(20260912 + len(particles))
    x = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=operator_dtype)
    y = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=operator_dtype)
    lhs = np.vdot(asnumpy(coupling.apply(x)), y)
    rhs = np.vdot(x, asnumpy(coupling.apply_adjoint(y)))
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0) < tolerance


def test_cupy_mlfmm_adjoint_exact_near_uses_device_kernel() -> None:
    """The reverse exact-near action needs no resident dense block cache."""

    lmax = 1
    particles = _mlfmm_single_level_particles()
    prepared_reference = prepare_matvec(
        lmax=lmax,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=4, max_depth=4),
        backend="numpy",
    )
    coupling = prepared_reference.coupling
    assert isinstance(coupling, MLFMMCouplingOperator)
    runtime = prepare_mlfmm_cupy_coupling(coupling)
    n = len(particles) * n_modes(lmax)
    rng = np.random.default_rng(20260912)
    x = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=np.complex128)
    y = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=np.complex128)
    lhs = np.vdot(asnumpy(runtime.apply(x)), y)
    rhs = np.vdot(x, asnumpy(runtime.apply_adjoint(y)))
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0) < 1.0e-9
    workspace = runtime.memory_diagnostics()["workspace_bytes"]
    assert isinstance(workspace, dict)
    assert "near_adjoint_blocks_bytes" not in workspace


@pytest.mark.parametrize("max_depth", (2, 3, 4))
def test_cupy_mlfmm_adjoint_depth_probe(max_depth: int) -> None:
    """Depth-2/3/4 occupied hierarchies retain the same adjoint identity."""

    particles = _mlfmm_adjoint_depth_particles()
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=5.0,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(
            max_leaf_particles=1,
            max_depth=max_depth,
            accuracy_level=1,
            order_additive=0,
        ),
        backend="cupy",
    )
    coupling = prepared.coupling
    assert isinstance(coupling, CuPyMLFMMCouplingOperator)
    assert coupling.prepared_data.stage in {"single_level", "multilevel"}
    n = len(particles) * n_modes(1)
    rng = np.random.default_rng(20260912 + max_depth)
    x = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    y = np.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    lhs = np.vdot(asnumpy(coupling.apply(x)), y)
    rhs = np.vdot(x, asnumpy(coupling.apply_adjoint(y)))
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0) < 1e-9


def test_cupy_mlfmm_options_collect_stream_stats_reaches_runtime_policy() -> None:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    particles = _mlfmm_transition_particles()
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(
            max_leaf_particles=4,
            max_depth=4,
            collect_stream_stats=True,
        ),
        backend="cupy",
    )

    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    assert prepared_cupy.coupling.host_cache_policy.collect_stream_stats is True


def test_cupy_mlfmm_prepared_operator_block_rhs_matches_columnwise() -> None:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    particles = _mlfmm_transition_particles()
    options = MLFMMOptions(max_leaf_particles=8, max_depth=4)
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex128,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="cupy",
    )
    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    nm = n_modes(lmax)
    n_particles = len(particles)
    rng = np.random.default_rng(202603231)
    x_block = np.asarray(
        rng.standard_normal((n_particles * nm, 2))
        + 1j * rng.standard_normal((n_particles * nm, 2)),
        dtype=np.complex128,
    )
    y_block = np.asarray(asnumpy(prepared_cupy.apply_W(x_block)), dtype=np.complex128)
    y_cols = np.column_stack(
        [
            np.asarray(asnumpy(prepared_cupy.apply_W(x_block[:, j])), dtype=np.complex128)
            for j in range(2)
        ]
    )
    np.testing.assert_allclose(y_block, y_cols, rtol=1e-10, atol=1e-10)
    _assert_mlfmm_live_results_survive_composition(prepared_cupy, x_block)


def test_cupy_mlfmm_complex64_request_matches_numpy_with_far_complex128() -> None:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    particles = _mlfmm_transition_particles()
    options = MLFMMOptions(max_leaf_particles=8, max_depth=4)

    prepared_numpy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex64,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="numpy",
    )
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        operator_dtype=np.complex64,
        coupling_backend="mlfmm",
        mlfmm_options=options,
        backend="cupy",
    )

    assert isinstance(prepared_numpy.coupling, MLFMMCouplingOperator)
    assert isinstance(prepared_cupy.coupling, CuPyMLFMMCouplingOperator)
    assert prepared_numpy.coupling.near_dtype == np.dtype(np.complex64)
    assert prepared_numpy.coupling.far_dtype == np.dtype(np.complex128)
    assert prepared_cupy.coupling.near_dtype == np.dtype(np.complex64)
    assert prepared_cupy.coupling.far_dtype == np.dtype(np.complex128)

    nm = n_modes(lmax)
    n_particles = len(particles)
    rng = np.random.default_rng(20260324)
    x = np.asarray(
        rng.standard_normal(n_particles * nm) + 1j * rng.standard_normal(n_particles * nm),
        dtype=np.complex64,
    )
    y_numpy = np.asarray(prepared_numpy.apply_W(x), dtype=np.complex64)
    y_cupy = np.asarray(asnumpy(prepared_cupy.apply_W(x)), dtype=np.complex64)
    assert y_cupy.dtype == np.dtype(np.complex64)
    np.testing.assert_allclose(y_cupy, y_numpy, rtol=1e-4, atol=1e-5)
    _assert_mlfmm_live_results_survive_composition(prepared_cupy, x)


@pytest.mark.parametrize(
    (
        "operator_dtype",
        "coeff_rtol",
        "coeff_atol",
        "ff_back_rtol",
        "ff_back_atol",
        "nf_rtol",
        "nf_atol",
        "cs_main_rtol",
        "cs_main_atol",
        "cs_local_atol",
        "cs_delta_rtol",
        "cs_delta_atol",
    ),
    [
        (np.complex64, 3e-5, 3e-6, 4e-5, 2e-6, 3e-5, 3e-6, 3e-5, 1e-4, 1e-8, 1e-4, 1e-4),
        (np.complex128, 5e-9, 5e-10, 5e-9, 5e-10, 1e-8, 1e-9, 5e-9, 1e-9, 1e-8, 5e-9, 1e-9),
    ],
)
def test_cupy_simulation_run_matches_numpy_for_coeffs_farfield_and_nearfield(
    operator_dtype: np.dtype,
    coeff_rtol: float,
    coeff_atol: float,
    ff_back_rtol: float,
    ff_back_atol: float,
    nf_rtol: float,
    nf_atol: float,
    cs_main_rtol: float,
    cs_main_atol: float,
    cs_local_atol: float,
    cs_delta_rtol: float,
    cs_delta_atol: float,
) -> None:
    wavelength = 550.0
    n_medium = 1.0 + 0j
    particles = _small_cluster_particles()
    source = _plane_wave_source(wavelength, n_medium)

    compute_dtype: Literal["complex64", "complex128"] = (
        "complex64" if operator_dtype == np.complex64 else "complex128"
    )
    cfg_numpy = _sim_cfg(
        operator_backend="numpy",
        compute_dtype=compute_dtype,
        wavelength=wavelength,
        n_medium=n_medium,
    )
    cfg_cupy = _sim_cfg(
        operator_backend="cupy",
        compute_dtype=compute_dtype,
        wavelength=wavelength,
        n_medium=n_medium,
    )

    run_numpy = pcl.Simulation(cfg_numpy, particles=particles).run(source, include_farfield=True)
    run_cupy = pcl.Simulation(cfg_cupy, particles=particles).run(source, include_farfield=True)

    np.testing.assert_allclose(
        run_cupy.coeffs,
        run_numpy.coeffs,
        rtol=coeff_rtol,
        atol=coeff_atol,
    )
    np.testing.assert_allclose(
        run_cupy.farfield.scattered.coeff_te,
        run_numpy.farfield.scattered.coeff_te,
        rtol=coeff_rtol,
        atol=coeff_atol,
    )
    np.testing.assert_allclose(
        run_cupy.farfield.scattered.coeff_tm,
        run_numpy.farfield.scattered.coeff_tm,
        rtol=coeff_rtol,
        atol=coeff_atol,
    )

    intensity_numpy = far_field_intensity(run_numpy.farfield.scattered)
    intensity_cupy = far_field_intensity(run_cupy.farfield.scattered)
    kz = np.asarray(run_numpy.farfield.scattered.kz, dtype=float)
    backward_mask = kz <= 0.0
    np.testing.assert_allclose(
        intensity_cupy[backward_mask],
        intensity_numpy[backward_mask],
        rtol=ff_back_rtol,
        atol=ff_back_atol,
    )
    if run_numpy.cross_sections is None or run_cupy.cross_sections is None:
        raise AssertionError("Plane-wave CuPy/NumPy parity run must expose cross sections.")
    for attribute in ("extinction", "scattering", "local_absorption"):
        np.testing.assert_allclose(
            getattr(run_cupy.cross_sections, attribute),
            getattr(run_numpy.cross_sections, attribute),
            rtol=cs_main_rtol,
            atol=cs_main_atol,
        )
    # `absorption_by_difference` subtracts two large integrated quantities.  In
    # complex64, the backend solution difference is amplified by this
    # cancellation even though the fields and primary cross sections agree.
    # Keep cross-backend parity for the precision-critical complex128 path;
    # for complex64, validate the diagnostic's own assembly without treating
    # its backend-sensitive residual as a physical parity target.
    if operator_dtype == np.complex128:
        np.testing.assert_allclose(
            run_cupy.cross_sections.absorption_by_difference,
            run_numpy.cross_sections.absorption_by_difference,
            rtol=cs_delta_rtol,
            atol=cs_delta_atol,
        )
        np.testing.assert_allclose(
            run_cupy.cross_sections.closure_error,
            run_numpy.cross_sections.closure_error,
            rtol=cs_delta_rtol,
            atol=cs_delta_atol,
        )
    else:
        for cross_sections in (run_numpy.cross_sections, run_cupy.cross_sections):
            if cross_sections is None:
                raise AssertionError("Cross-section diagnostics unexpectedly missing.")
            assert np.isfinite(cross_sections.absorption_by_difference)
            assert np.isfinite(cross_sections.closure_error)
            np.testing.assert_allclose(
                cross_sections.absorption_by_difference,
                cross_sections.extinction - cross_sections.scattering,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                cross_sections.closure_error,
                cross_sections.absorption_by_difference - cross_sections.local_absorption,
                rtol=0.0,
                atol=1e-12,
            )
    np.testing.assert_allclose(
        run_cupy.cross_sections.local_absorption,
        run_numpy.cross_sections.local_absorption,
        rtol=0.0,
        atol=cs_local_atol,
    )

    nearfield_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [220.0, 25.0, -60.0],
            [82.0, 0.0, 0.0],
            [0.0, 84.0, 0.0],
            [220.0 + 83.5, 25.0, -60.0],
            [-180.0, 90.0 + 81.5, 70.0],
            [30.0, -15.0, 110.0],
            [220.0, 25.0, -60.0 + 86.0],
        ],
        dtype=float,
    )
    nf_numpy = pcl.compute_near_field(run_numpy, points=nearfield_points, show_progress=False)
    nf_cupy = pcl.compute_near_field(run_cupy, points=nearfield_points, show_progress=False)

    np.testing.assert_array_equal(np.asarray(nf_cupy.inside_mask), np.asarray(nf_numpy.inside_mask))
    np.testing.assert_allclose(
        np.asarray(nf_cupy.E_scattered),
        np.asarray(nf_numpy.E_scattered),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.H_scattered),
        np.asarray(nf_numpy.H_scattered),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.E_initial),
        np.asarray(nf_numpy.E_initial),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.H_initial),
        np.asarray(nf_numpy.H_initial),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.E_total),
        np.asarray(nf_numpy.E_total),
        rtol=nf_rtol,
        atol=nf_atol,
    )
    np.testing.assert_allclose(
        np.asarray(nf_cupy.H_total),
        np.asarray(nf_numpy.H_total),
        rtol=nf_rtol,
        atol=nf_atol,
    )


def test_cupy_mixed_particle_groups_match_numpy_for_solve_and_backscatter() -> None:
    wavelength = 550.0
    n_medium = 1.0 + 0j
    particles = _mixed_cluster_particles()
    source = _plane_wave_source(wavelength, n_medium)

    cfg_numpy = _sim_cfg(
        operator_backend="numpy",
        compute_dtype="complex128",
        wavelength=wavelength,
        n_medium=n_medium,
    )
    cfg_cupy = _sim_cfg(
        operator_backend="cupy",
        compute_dtype="complex128",
        wavelength=wavelength,
        n_medium=n_medium,
    )

    run_numpy = pcl.Simulation(cfg_numpy, particles=particles).run(source, include_farfield=True)
    run_cupy = pcl.Simulation(cfg_cupy, particles=particles).run(source, include_farfield=True)

    np.testing.assert_allclose(run_cupy.coeffs, run_numpy.coeffs, rtol=5e-8, atol=5e-10)

    intensity_numpy = far_field_intensity(run_numpy.farfield.scattered)
    intensity_cupy = far_field_intensity(run_cupy.farfield.scattered)
    kz = np.asarray(run_numpy.farfield.scattered.kz, dtype=float)
    backward_mask = kz <= 0.0
    np.testing.assert_allclose(
        intensity_cupy[backward_mask],
        intensity_numpy[backward_mask],
        rtol=5e-8,
        atol=5e-10,
    )


def _assert_mlfmm_live_results_survive_composition(prepared: Any, x: Any) -> None:
    """Reuse prepared fixtures to check the live vectors needed by LSQR."""
    cupy, _ = import_cupy()
    x = cupy.asarray(x)
    before = x.copy()
    forward = prepared.apply_W(x)
    forward_before = forward.copy()
    adjoint = prepared.apply_adjoint(x)
    adjoint_before = adjoint.copy()
    # A(v) must not clear the workspace backing its live A^H(u) input.
    prepared.apply_A(adjoint)
    np.testing.assert_array_equal(asnumpy(adjoint), asnumpy(adjoint_before))
    np.testing.assert_array_equal(asnumpy(forward), asnumpy(forward_before))
    np.testing.assert_array_equal(asnumpy(x), asnumpy(before))


@pytest.mark.parametrize("out_dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("near_dtype", (np.complex64, np.complex128))
def test_mlfmm_near_far_combination_owns_output_and_keeps_wide_sum(out_dtype, near_dtype) -> None:
    cupy, _ = import_cupy()
    # Narrowing far before addition would erase the small residual entirely.
    far_host = np.asarray([2.0**25 + 0.5, -(2.0**25) + 0.25j], dtype=np.complex128)
    near_host = np.asarray([-(2.0**25), 2.0**25], dtype=near_dtype)
    far = cupy.asarray(far_host).reshape(1, 2, 1)
    near = cupy.asarray(near_host).reshape(1, 2, 1)
    result = _combine_near_far_outputs(
        near, far, out_dtype=np.dtype(out_dtype), far_is_workspace=True, cupy=cupy
    )
    expected = (far_host + near_host.astype(np.complex128)).astype(out_dtype)
    assert result.dtype == np.dtype(out_dtype)
    np.testing.assert_array_equal(asnumpy(result).reshape(-1), expected)
    np.testing.assert_array_equal(asnumpy(far).reshape(-1), far_host)
    np.testing.assert_array_equal(asnumpy(near).reshape(-1), near_host)
    assert not cupy.shares_memory(result, far)
    assert not cupy.shares_memory(result, near)
    far.fill(0)
    near.fill(0)
    np.testing.assert_array_equal(asnumpy(result).reshape(-1), expected)


@pytest.mark.parametrize("near_dtype", (np.complex64, np.complex128))
def test_mlfmm_near_far_combination_transfers_fresh_far_storage(near_dtype) -> None:
    cupy, _ = import_cupy()
    far = cupy.asarray([2.0**25 + 0.5], dtype=cupy.complex128)
    near = cupy.asarray([-(2.0**25)], dtype=near_dtype)
    result = _combine_near_far_outputs(
        near, far, out_dtype=np.dtype(np.complex128), far_is_workspace=False, cupy=cupy
    )
    assert result is far
    np.testing.assert_array_equal(asnumpy(result), np.asarray([0.5]))
    np.testing.assert_array_equal(asnumpy(near), np.asarray([-(2.0**25)], dtype=near_dtype))
