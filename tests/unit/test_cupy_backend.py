from __future__ import annotations

import pickle
from typing import Any, Literal

import numpy as np
import pytest

import pyceles as pcl
from pyceles._optional import asnumpy, import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
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
    _directional_to_box_regular_cupy,
    _upload_directional_transforms,
    _upload_offset_batches,
)
from pyceles.core.operators.mlfmm_directional import (
    box_outgoing_to_directional,
    directional_to_box_regular,
    directional_transforms,
)
from pyceles.core.particles import Particle, spheres_from_arrays
from pyceles.core.translation import RadialLUT
from pyceles.io import far_field_intensity
from pyceles.postprocessing.farfield import (
    local_absorbed_power_components_from_exciting,
    local_absorbed_power_from_exciting,
    local_absorption_cross_section_from_exciting,
)

pytestmark = pytest.mark.gpu


def _small_cluster_particles() -> tuple[Particle, ...]:
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
    return tuple(
        spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=n_particle,
        )
    )


def _random_uniform_sphere_particles(*, n_particles: int, seed: int = 4) -> tuple[Particle, ...]:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1000.0, 1000.0, size=(int(n_particles), 3))
    radii = np.full((positions.shape[0],), 20.0, dtype=float)
    n_particle = np.full((positions.shape[0],), 1.59 + 0.0j, dtype=np.complex128)
    return tuple(
        spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=n_particle,
        )
    )


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
    assert tuple(uploaded.fth_beta.shape) == (transforms.grid.beta.size, transforms.Fth.shape[1])
    assert tuple(uploaded.fph_beta.shape) == (transforms.grid.beta.size, transforms.Fph.shape[1])
    dense_bytes = int(transforms.Fth.nbytes + transforms.Fph.nbytes)
    structured_bytes = int(
        uploaded.fth_beta.nbytes
        + uploaded.fph_beta.nbytes
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


def test_local_absorbed_power_components_from_exciting_accepts_cupy_arrays() -> None:
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
    ref = local_absorbed_power_components_from_exciting(
        e_np,
        x_np,
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
        n_particles=n_particles,
        nmodes_per_particle=nmodes,
    )
    got = local_absorbed_power_components_from_exciting(
        cupy.asarray(e_np),
        cupy.asarray(x_np),
        k0=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0j,
        n_particles=n_particles,
        nmodes_per_particle=nmodes,
    )
    np.testing.assert_allclose(
        float(got["P_abs_local"]), float(ref["P_abs_local"]), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(got["P_abs_local_particles"], dtype=float),
        np.asarray(ref["P_abs_local_particles"], dtype=float),
        rtol=1e-13,
        atol=1e-13,
    )


def _policy_numpy_mlfmm_coupling() -> MLFMMCouplingOperator:
    prepared = prepare_matvec(
        lmax=3,
        k=2.0 * np.pi / 550.0,
        particles=list(_mlfmm_policy_particles()),
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
        mlfmm_options=MLFMMOptions(max_leaf_particles=4, max_depth=4),
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
    positions = np.asarray([np.asarray(p.position, dtype=float) for p in particles], dtype=float)
    radii = np.asarray([float(p.circumscribing_radius()) for p in particles], dtype=float)
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
        particles=list(_mlfmm_policy_particles()),
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


def _mlfmm_transition_particles() -> tuple[Particle, ...]:
    # 27 particles (seed=50) is the smallest deterministic fixture we found that
    # gives the intended stage split for this test surface:
    # - max_leaf_particles=8 -> single_level
    # - max_leaf_particles=4 -> multilevel
    return _random_uniform_sphere_particles(n_particles=27, seed=50)


def _transition_numpy_mlfmm_coupling(*, max_leaf_particles: int) -> MLFMMCouplingOperator:
    lmax = 1
    wavelength = 550.0
    n_medium = 1.0 + 0j
    k = 2.0 * np.pi / wavelength
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=list(_mlfmm_transition_particles()),
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


def _mlfmm_policy_particles() -> tuple[Particle, ...]:
    # Policy/retention tests do not require a stage split; keep this fixture
    # small so host-cache/policy tests stay lightweight.
    return _random_uniform_sphere_particles(n_particles=24, seed=4)


def _sim_cfg(
    *,
    operator_backend: Literal["numpy", "cupy"],
    compute_dtype: Literal["complex64", "complex128"],
    wavelength: float,
    n_medium: complex,
    source: pcl.PlaneWave,
) -> pcl.SimulationConfig:
    return pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=n_medium,
        lmax=3,
        source=source,
        polar_angles=pcl.core.uniform_polar_grid(181),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(36),
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
        source=source,
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
    run = sim.run(include_farfield=True)

    assert run.cross_sections is not None
    assert seen_backend_payloads
    assert seen_backend_payloads[-1] == (False, True)
    assert not hasattr(run.solver_result, "backend_x")
    assert sim._solve_backend_handoffs == {}


def test_cupy_public_solve_sources_does_not_retain_backend_handoff() -> None:
    wavelength = 550.0
    source = _plane_wave_source(wavelength, 1.0 + 0j)
    cfg = pcl.SimulationConfig(
        wavelength=wavelength,
        n_medium=1.0 + 0j,
        lmax=2,
        source=source,
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
    assert sim._solve_backend_handoffs == {}


@pytest.mark.parametrize("collect_stream_stats", [False, True])
def test_cupy_multilevel_stream_stats_collection_is_opt_in(collect_stream_stats: bool) -> None:
    coupling = _transition_numpy_mlfmm_coupling(max_leaf_particles=4)
    assert str(coupling.resolved_plan.stage) == "multilevel"
    runtime = prepare_mlfmm_cupy_coupling(
        coupling,
        host_cache_policy=CuPyMLFMMHostCachePolicy(collect_stream_stats=collect_stream_stats),
    )

    nm = n_modes(1)
    n_particles = len(_mlfmm_transition_particles())
    rng = np.random.default_rng(20260409 + int(collect_stream_stats))
    x = np.asarray(
        rng.standard_normal(n_particles * nm) + 1j * rng.standard_normal(n_particles * nm),
        dtype=np.complex128,
    )
    _ = asnumpy(runtime.apply(x))
    diag = runtime.memory_diagnostics()
    streaming = diag.get("multilevel_streaming")
    rolling = diag.get("multilevel_rolling")
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
    transient_budget = streaming.get("stream_transient_budget_bytes")
    unmodeled_temp_reserve = streaming.get("stream_unmodeled_temp_reserve_bytes")
    chunk_bytes_budget = streaming.get("resolved_streamed_far_chunk_bytes_budget")
    frontier_bytes_budget = streaming.get("resolved_streamed_far_frontier_bytes_budget")
    assert isinstance(chunk_box_cap, int)
    assert isinstance(frontier_box_cap, int)
    assert isinstance(leaf_chunk_min, int)
    assert isinstance(leaf_chunk_max, int)
    assert isinstance(leaf_chunk_by_occupancy, dict)
    assert isinstance(device_limit, int)
    assert isinstance(pool_used_before_plan, int)
    assert isinstance(pool_trimmed_before_plan, bool)
    assert isinstance(transient_budget, int)
    assert isinstance(unmodeled_temp_reserve, int)
    assert isinstance(chunk_bytes_budget, int)
    assert isinstance(frontier_bytes_budget, int)
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
        (8, "single_level"),
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
    particles = _mlfmm_transition_particles()
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
        source=source,
    )
    cfg_cupy = _sim_cfg(
        operator_backend="cupy",
        compute_dtype=compute_dtype,
        wavelength=wavelength,
        n_medium=n_medium,
        source=source,
    )

    run_numpy = pcl.Simulation(cfg_numpy, particles=particles).run(include_farfield=True)
    run_cupy = pcl.Simulation(cfg_cupy, particles=particles).run(include_farfield=True)

    np.testing.assert_allclose(
        run_cupy.coeffs,
        run_numpy.coeffs,
        rtol=coeff_rtol,
        atol=coeff_atol,
    )
    np.testing.assert_allclose(
        run_cupy.farfield.scattered_te["coeff"],
        run_numpy.farfield.scattered_te["coeff"],
        rtol=coeff_rtol,
        atol=coeff_atol,
    )
    np.testing.assert_allclose(
        run_cupy.farfield.scattered_tm["coeff"],
        run_numpy.farfield.scattered_tm["coeff"],
        rtol=coeff_rtol,
        atol=coeff_atol,
    )

    intensity_numpy = far_field_intensity(
        run_numpy.farfield.scattered_te, run_numpy.farfield.scattered_tm
    )
    intensity_cupy = far_field_intensity(
        run_cupy.farfield.scattered_te, run_cupy.farfield.scattered_tm
    )
    kz = np.asarray(run_numpy.farfield.scattered_te["kz"], dtype=float)
    backward_mask = kz <= 0.0
    np.testing.assert_allclose(
        intensity_cupy[backward_mask],
        intensity_numpy[backward_mask],
        rtol=ff_back_rtol,
        atol=ff_back_atol,
    )
    if run_numpy.cross_sections is None or run_cupy.cross_sections is None:
        raise AssertionError("Plane-wave CuPy/NumPy parity run must expose cross sections.")
    for key in (
        "C_ext",
        "C_sca",
        "C_abs",
        "C_ext_raw",
        "C_sca_raw",
    ):
        np.testing.assert_allclose(
            run_cupy.cross_sections[key],
            run_numpy.cross_sections[key],
            rtol=cs_main_rtol,
            atol=cs_main_atol,
        )
    # `C_abs_raw_diff` subtracts two large integrated quantities.  In complex64
    # CPU/GPU parity it is more sensitive to reduction order than the fields or
    # the primary extinction/scattering diagnostics above.
    np.testing.assert_allclose(
        run_cupy.cross_sections["C_abs_raw_diff"],
        run_numpy.cross_sections["C_abs_raw_diff"],
        rtol=cs_delta_rtol,
        atol=cs_delta_atol,
    )
    np.testing.assert_allclose(
        run_cupy.cross_sections["C_abs_local"],
        run_numpy.cross_sections["C_abs_local"],
        rtol=0.0,
        atol=cs_local_atol,
    )
    np.testing.assert_allclose(
        run_cupy.cross_sections["Delta_closure"],
        run_numpy.cross_sections["Delta_closure"],
        rtol=cs_delta_rtol,
        atol=cs_delta_atol,
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
    nf_numpy = pcl.compute_near_field(
        run_numpy, points=nearfield_points, channel="mixed", show_progress=False
    )
    nf_cupy = pcl.compute_near_field(
        run_cupy, points=nearfield_points, channel="mixed", show_progress=False
    )

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
        source=source,
    )
    cfg_cupy = _sim_cfg(
        operator_backend="cupy",
        compute_dtype="complex128",
        wavelength=wavelength,
        n_medium=n_medium,
        source=source,
    )

    run_numpy = pcl.Simulation(cfg_numpy, particles=particles).run(include_farfield=True)
    run_cupy = pcl.Simulation(cfg_cupy, particles=particles).run(include_farfield=True)

    np.testing.assert_allclose(run_cupy.coeffs, run_numpy.coeffs, rtol=5e-8, atol=5e-10)

    intensity_numpy = far_field_intensity(
        run_numpy.farfield.scattered_te,
        run_numpy.farfield.scattered_tm,
    )
    intensity_cupy = far_field_intensity(
        run_cupy.farfield.scattered_te,
        run_cupy.farfield.scattered_tm,
    )
    kz = np.asarray(run_numpy.farfield.scattered_te["kz"], dtype=float)
    backward_mask = kz <= 0.0
    np.testing.assert_allclose(
        intensity_cupy[backward_mask],
        intensity_numpy[backward_mask],
        rtol=5e-8,
        atol=5e-10,
    )
