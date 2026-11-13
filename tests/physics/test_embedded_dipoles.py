"""Particle-local emission composes with the ordinary many-body operator."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyceles.core.operators.mlfmm import MLFMMCouplingOperator, MLFMMOptions
from pyceles.core.operators.mlfmm_cupy import CuPyMLFMMCouplingOperator
from pyceles.core.particles import LayeredSphere, PECSphere, Sphere
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.core.sources.embedded import prepare_embedded_dipole
from pyceles.core.svwf import expansion_fields
from pyceles.postprocessing.dipole_metrics import (
    compute_dipole_reaction_field,
    compute_dipole_source_work,
)
from pyceles.postprocessing.nearfield import compute_electric_field, compute_near_field
from pyceles.simulation import ResultRetention, Simulation, SimulationConfig


def _simulation(particles, *, host=1.33 + 0.08j, solver="direct", dtype="complex128"):
    return Simulation(
        SimulationConfig(
            wavelength=550,
            n_medium=host,
            lmax=3,
            radial_lut_dr=0.25,
            solver_method=solver,
            solver_rtol=1e-10,
            compute_dtype=dtype,
            accum_dtype="complex128",
            verbose=False,
            polar_angles=np.linspace(0, np.pi, 7),
            azimuthal_angles=np.linspace(0, 2 * np.pi, 8, endpoint=False),
        ),
        particles=particles,
    )


@pytest.mark.parametrize("solver", ["direct", "gmres", "bicgstab", "lsqr"])
def test_internal_emission_and_external_illumination_obey_cluster_reciprocity(solver):
    particles = (
        LayeredSphere(
            position=(0, 0, 0),
            layer_radii=(50, 80),
            layer_refractive_indices=(1.6 + 0.05j, 1.9 + 0.12j),
        ),
        PECSphere(position=(300, 0, 0), radius=45),
    )
    simulation = _simulation(particles, solver=solver)
    internal = DipoleSource(
        550,
        medium_n=simulation.config.n_medium,
        position=(60, 0, 0),
        dipole_moment=(1, 0.3j, -0.2),
        radial_lut_dr=0.25,
    )
    external = DipoleSource(
        550,
        medium_n=simulation.config.n_medium,
        position=(600, 0, 0),
        dipole_moment=(-0.1j, 0.6, 0.2j),
        radial_lut_dr=0.25,
    )
    internal_run = simulation.run(internal, include_farfield=False)
    external_run = simulation.run(external, include_farfield=False)
    e_emitted = compute_near_field(
        internal_run, points=external.source_positions(), show_progress=False
    ).E_total[0]
    e_received = compute_near_field(
        external_run, points=internal.source_positions(), show_progress=False
    ).E_total[0]
    np.testing.assert_allclose(
        external.dipole_moments()[0] @ e_emitted,
        internal.dipole_moments()[0] @ e_received,
        rtol=2e-9,
        atol=0,
    )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_mixed_collection_fields_and_coefficients_are_coherent_sums(dtype):
    host = 1.33
    particles = (
        Sphere(position=(0, 0, 0), radius=80, refractive_index=1.7 + 0.15j),
        LayeredSphere(
            position=(300, 0, 0), layer_radii=(40, 60), layer_refractive_indices=(1.5, 1.9 + 0.1j)
        ),
    )
    positions = np.array([[20, 0, 0], [350, 0, 0], [600, 0, 0]])
    moments = np.array([[1, 0.3j, -0.2], [0.1j, 0.5, 0.1], [-0.1j, 0.6, 0.2j]])
    collection = DipoleCollection(
        550, medium_n=host, positions=positions, dipole_moments=moments, radial_lut_dr=0.25
    )
    simulation = _simulation(particles, host=host, dtype=dtype)
    singles = {
        str(i): DipoleSource(
            550, medium_n=host, position=tuple(pos), dipole_moment=tuple(moment), radial_lut_dr=0.25
        )
        for i, (pos, moment) in enumerate(zip(positions, moments, strict=True))
    }
    basis = simulation.run_sources(singles, include_farfield=False)
    combined = simulation.run(
        collection, include_farfield=False, retention=ResultRetention.minimal()
    )
    assert combined.initial_coeffs is None and combined.rhs is None
    tolerance = 2e-6 if dtype == "complex64" else 3e-13
    expected_coeffs = sum(run.coeffs for run in basis.channels.values())
    assert (
        np.linalg.norm(combined.coeffs - expected_coeffs) / np.linalg.norm(expected_coeffs)
        < tolerance
    )
    points = np.array([[0, 0, 0], [21, 0, 0], [300, 0, 0], [350, 1, 0], [700, 0, 0]])
    total = compute_near_field(combined, points=points, show_progress=False)
    fields = [
        compute_near_field(run, points=points, show_progress=False)
        for run in basis.channels.values()
    ]
    for name in ("E_total", "H_total", "E_internal", "H_internal"):
        actual = getattr(total, name)
        expected = sum(getattr(field, name) for field in fields)
        assert actual.dtype == np.complex128
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
    electric = compute_electric_field(combined, points=points, show_progress=False)
    np.testing.assert_array_equal(electric.E_total, total.E_total)


def test_only_embedded_sources_have_no_bare_host_farfield_copy():
    simulation = _simulation((Sphere(position=(0, 0, 0), radius=80, refractive_index=1.7 + 0.15j),))
    source = DipoleSource(550, medium_n=simulation.config.n_medium, position=(20, 0, 0))
    run = simulation.run(source)
    assert run.farfield.initial is not None
    for name in ("coeff_te", "coeff_tm"):
        np.testing.assert_array_equal(getattr(run.farfield.initial, name), 0)
        np.testing.assert_array_equal(
            getattr(run.farfield.total, name), getattr(run.farfield.scattered, name)
        )
    # T-only passive absorption identities do not apply to an emitting particle.
    assert run.power is None
    singular = compute_near_field(run, points=source.source_positions(), show_progress=False)
    assert np.isnan(singular.E_total).all() and np.isnan(singular.H_total).all()


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize(
    "radius,indices,expected",
    [
        (20, (1.5, 1.7 + 0.15j), 1.05551211320675),
        (65, (1.5 + 0.1j, 1.7), 1.00376849150879),
    ],
)
def test_local_reaction_order_converges_independently_of_radiation_order(
    dtype, radius, indices, expected
):
    # Corrected spherical transfer/integral reference, fixed physical dipole
    # normalized to its local bulk power (Moroz, Ann. Phys. 315, 352, 2005).
    particle = LayeredSphere((0, 0, 0), (50, 80), indices)
    simulation = _simulation((particle,), host=1, dtype=dtype)
    run = simulation.run(DipoleSource(550, position=(radius, 0, 0)), include_farfield=False)
    coarse = compute_dipole_source_work(run)
    refined = compute_dipole_source_work(run, reaction_lmax=64)
    converged = compute_dipole_source_work(run, reaction_lmax=96)
    assert np.isfinite(converged.E_reaction_at_dipoles).all()
    assert converged.E_reaction_at_dipoles.dtype == np.complex128
    tolerance = 2e-6 if dtype == "complex64" else 2e-11
    np.testing.assert_allclose(converged.power_ratio, expected, rtol=tolerance)
    np.testing.assert_allclose(refined.power_ratio, converged.power_ratio, rtol=tolerance)
    assert abs(coarse.power_ratio[0] - expected) > 0.01


def test_reaction_field_survives_absorbing_local_material_but_not_absolute_self_power():
    simulation = _simulation((Sphere((0, 0, 0), 80, 1.7 + 0.15j),))
    run = simulation.run(
        DipoleSource(550, medium_n=simulation.config.n_medium, position=(20, 0, 0)),
        include_farfield=False,
    )
    reaction = compute_dipole_reaction_field(run, reaction_lmax=32)
    assert np.isfinite(reaction).all() and np.linalg.norm(reaction) > 0
    with pytest.raises(ValueError, match="nonabsorbing local material"):
        compute_dipole_source_work(run)


def test_absorbing_exterior_does_not_forbid_lossless_local_source_work():
    simulation = _simulation((Sphere((0, 0, 0), 80, 1.7),))
    run = simulation.run(
        DipoleSource(550, medium_n=simulation.config.n_medium, position=(0, 0, 0)),
        include_farfield=False,
    )
    work = compute_dipole_source_work(run)
    np.testing.assert_array_equal(work.local_refractive_indices, [1.7])
    assert work.power_total[0] > 0
    assert np.isfinite(work.power_ratio).all()


@pytest.mark.parametrize(
    "host,backend",
    [
        (1.33, "numpy"),
        (1.33 + 0.08j, "numpy"),
        pytest.param(1.33 + 0.08j, "cupy", marks=pytest.mark.gpu),
    ],
)
def test_embedded_mlfmm_matches_pairwise_for_coefficients_fields_and_work(host, backend, request):
    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    corners = np.array([[x, y, z] for x in (-700, 700) for y in (-700, 700) for z in (-700, 700)])
    positions = np.vstack([corners + i * np.array([150, 100, 80]) for i in range(4)])
    particles = tuple(Sphere(tuple(p), 40, 1.7) for p in positions)
    pairwise = _simulation(particles, host=host, solver="gmres")
    config = replace(pairwise.config, lmax=2)
    pairwise = Simulation(replace(config, cache_translation_blocks=True), particles=particles)
    mlfmm = Simulation(
        replace(
            config,
            coupling_backend="mlfmm",
            operator_backend=backend,
            mlfmm_options=MLFMMOptions(max_leaf_particles=2, max_depth=3, accuracy_level=2),
        ),
        particles=particles,
    )
    source = DipoleCollection(
        550,
        medium_n=host,
        positions=positions[[0, 17]],
        dipole_moments=np.array([[1, 0.2j, 0], [0.3, 1, -0.1j]]),
        radial_lut_dr=0.25,
    )
    expected = pairwise.run(source, include_farfield=False)
    actual = mlfmm.run(source, include_farfield=False)
    assert mlfmm._prepared_operator_cache is not None
    coupling = mlfmm._prepared_operator_cache.coupling
    if backend == "cupy":
        assert isinstance(coupling, CuPyMLFMMCouplingOperator)
        multilevel = coupling.prepared_data.multilevel
        assert multilevel is not None
        assert len(multilevel.levels) == 4
    else:
        assert isinstance(coupling, MLFMMCouplingOperator)
        assert coupling.multilevel is not None
        assert len(coupling.multilevel.levels) == 4
    # Qualify at the same 1e-4 operator accuracy as the public MLFMM workflow;
    # the iterative residual alone does not measure sampled-map truncation.
    assert np.linalg.norm(actual.coeffs - expected.coeffs) / np.linalg.norm(expected.coeffs) < 1e-4
    points = np.vstack([positions[[0, 17]] + [2, 1, 3], [[0, 0, 0], [2600, 0, 0]]])
    actual_fields = compute_near_field(actual, points=points, show_progress=False)
    expected_fields = compute_near_field(expected, points=points, show_progress=False)
    for name in ("E_total", "H_total"):
        a, b = getattr(actual_fields, name), getattr(expected_fields, name)
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 1e-4
    np.testing.assert_allclose(
        compute_dipole_source_work(actual).power_total,
        compute_dipole_source_work(expected).power_total,
        rtol=1e-4,
        atol=0,
    )


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_coherent_embedded_work_matches_enclosing_flux_and_allows_negative_emitter_work(
    backend, request
):
    from numpy.polynomial.legendre import leggauss

    if backend == "cupy":
        request.getfixturevalue("cupy_runtime")
    simulation = _simulation((Sphere((0, 0, 0), 80, 1.5),), host=1)
    simulation = Simulation(
        replace(simulation.config, lmax=8, operator_backend=backend), particles=simulation.particles
    )
    source = DipoleCollection(
        550,
        positions=np.array([[-10, 0, 0], [10, 0, 0]]),
        dipole_moments=np.array([[1, 0, 0], [-0.5, 0, 0]]),
    )
    run = simulation.run(source, include_farfield=False)
    work = compute_dipole_source_work(run)
    assert work.power_total[0] > 0 and work.power_total[1] < 0
    ct, weights = leggauss(12)
    phi = np.arange(24) * 2 * np.pi / 24
    st = np.sqrt(1 - ct**2)
    directions = np.stack(
        np.broadcast_arrays(st[:, None] * np.cos(phi), st[:, None] * np.sin(phi), ct[:, None]),
        axis=-1,
    ).reshape(-1, 3)
    radius = 900
    e, h = expansion_fields(
        radius * directions,
        centre=np.zeros(3),
        regular=np.zeros_like(run.coeffs[0]),
        outgoing=run.coeffs[0],
        k=run.k,
        n=1,
        lmax=8,
        backend=backend,
    )
    radial_flux = 0.5 * np.sum(np.cross(e, h.conj()).real * directions, axis=1)
    flux = radius**2 * np.dot(np.repeat(weights, 24) * 2 * np.pi / 24, radial_flux)
    np.testing.assert_allclose(work.power_total.sum(), flux, rtol=2e-10, atol=0)


def test_embedded_fields_and_reaction_rotate_as_physical_vectors():
    particle = LayeredSphere((13, -24, 5), (50, 80), (1.5, 1.7 + 0.15j))
    centre = np.array(particle.position)
    offset = np.array([55, 10, 20])
    moment = np.array([1, 0.3j, -0.2])
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.2]).as_matrix()
    source = DipoleSource(
        550, medium_n=1.33 + 0.08j, position=tuple(centre + offset), dipole_moment=tuple(moment)
    )
    rotated_source = replace(
        source, position=tuple(centre + rotation @ offset), dipole_moment=tuple(rotation @ moment)
    )
    response = prepare_embedded_dipole(particle, source, lmax=16)
    rotated = prepare_embedded_dipole(particle, rotated_source, lmax=16)
    points = np.array([[0, 0, 0], [20, -5, 3], [65, 4, -3], [150, 20, -70]])
    for a, b in zip(
        response.fields(centre + points), rotated.fields(centre + points @ rotation.T), strict=True
    ):
        expected = a @ rotation.T
        assert np.linalg.norm(b - expected) / np.linalg.norm(expected) < 3e-13
    reaction, _ = response.region_fields(
        source.source_positions(), response.emitter_layer, include_bulk=False
    )
    rotated_reaction, _ = rotated.region_fields(
        rotated_source.source_positions(), rotated.emitter_layer, include_bulk=False
    )
    np.testing.assert_allclose(rotated_reaction, reaction @ rotation.T, rtol=2e-12, atol=0)


@pytest.mark.parametrize("position", [(0, 0, 0), (20, 0, 0), (80, 0, 0)])
def test_transparent_sphere_does_not_change_bulk_emission_or_reaction(position):
    host = 1.33 + 0.08j
    scatterer = Sphere((300, 0, 0), 40, 1.7 + 0.1j)
    transparent = LayeredSphere((0, 0, 0), (50, 80), (host, host))
    source = DipoleSource(
        550, medium_n=host, position=position, dipole_moment=(1, 0.3j, -0.2), radial_lut_dr=0.25
    )
    reference = _simulation((scatterer,), host=host).run(source, include_farfield=False)
    actual = _simulation((transparent, scatterer), host=host).run(source, include_farfield=False)
    np.testing.assert_allclose(actual.coeffs[0], 0, atol=0)
    np.testing.assert_allclose(actual.coeffs[1], reference.coeffs[0], rtol=3e-13, atol=0)
    points = np.array([[0, 0, 0], [30, 20, 10], [100, 0, 0], [300, 0, 0]])
    expected = compute_near_field(reference, points=points, show_progress=False)
    fields = compute_near_field(actual, points=points, show_progress=False)
    for name in ("E_total", "H_total"):
        np.testing.assert_allclose(
            getattr(fields, name), getattr(expected, name), rtol=3e-13, atol=0
        )
    np.testing.assert_allclose(
        compute_dipole_reaction_field(actual),
        compute_dipole_reaction_field(reference),
        rtol=3e-13,
        atol=0,
    )
