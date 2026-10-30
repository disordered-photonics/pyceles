"""Source ownership and affine RHS assembly for embedded electric emitters."""

from dataclasses import replace

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.operators.coupling_pairwise_cupy import CuPyPairwiseCouplingOperator
from pyceles.core.particles import (
    LayeredSphere,
    ParticleCollection,
    PECSphere,
    Sphere,
    Spheroid,
    TMatrixParticle,
)
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.core.sources import embedded as embedded_sources
from pyceles.core.sources.embedded import (
    locate_dipole_source,
    prepare_dipole_source,
    prepare_embedded_dipole,
)
from pyceles.core.tmatrix import particle_T_matrix_block
from pyceles.core.translation import translation_block_rect
from pyceles.postprocessing.dipole_metrics import (
    _prepare_diagnostic,
    compute_dipole_reaction_field,
)
from pyceles.simulation import Simulation, SimulationConfig


@pytest.mark.parametrize(
    "targets,error",
    [
        (np.array([2**32], dtype=np.int64), IndexError),
        (np.array([2**63], dtype=np.uint64), IndexError),
        (np.array([0.5]), TypeError),
        (np.array([[0]]), ValueError),
        (np.empty((0, 2), dtype=int), ValueError),
    ],
)
def test_reception_targets_are_validated_before_gpu_index_narrowing(targets, error):
    operator = CuPyPairwiseCouplingOperator(1, 1, np.zeros((2, 3)), None, None)
    with pytest.raises(error):
        operator.apply_rows(np.zeros(12), targets)


@pytest.mark.api_contract
def test_geometry_infers_local_material_and_ignores_zero_emitters():
    particles = ParticleCollection.from_particles(
        (
            LayeredSphere(
                position=(0, 0, 0), layer_radii=(50, 80), layer_refractive_indices=(1.5, 1.8 + 0.1j)
            ),
            PECSphere(position=(300, 0, 0), radius=45),
        )
    )
    source = DipoleCollection(
        550,
        medium_n=1.33 + 0.08j,
        positions=np.array([[10, 0, 0], [60, 0, 0], [600, 0, 0], [300, 0, 0]]),
        dipole_moments=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        amplitude=0.7 - 0.2j,
    )
    prepared = prepare_dipole_source(particles, source, lmax=2)
    assert [emitter.particle_index for emitter in prepared.embedded] == [0, 0]
    assert [emitter.emitter_index for emitter in prepared.embedded] == [0, 1]
    np.testing.assert_array_equal(prepared.particle_indices, [0, 0, -1, -1])
    assert [emitter.response.local_source.medium_n for emitter in prepared.embedded] == [
        1.5,
        1.8 + 0.1j,
    ]
    assert source.medium_n == 1.33 + 0.08j
    np.testing.assert_array_equal(prepared.host_source.source_positions(), source.positions[2:])
    for index, emitter in enumerate(prepared.embedded):
        response = emitter.response
        np.testing.assert_array_equal(
            response.local_source.dipole_moments()[0], source.dipole_moments_array()[index]
        )


def test_placement_is_order_independent_and_keeps_owned_source_geometry(monkeypatch):
    particle = Sphere((0, 0, 0), 50, 1.7 + 0.1j)
    particles = ParticleCollection.from_particles((particle,))
    source = DipoleCollection(
        550,
        positions=np.array([[10, 0, 0], [150, 20, 0]]),
        dipole_moments=np.array([[1, 0.2j, 0], [0, 0.7, 0.1j]]),
    )

    def no_matching(*args, **kwargs):
        raise AssertionError("Geometry-only placement must not solve material interfaces")

    monkeypatch.setattr(embedded_sources, "prepare_embedded_dipole", no_matching)
    placement = locate_dipole_source(particles, source)
    assert placement.positions is source.positions
    np.testing.assert_array_equal(placement.local_refractive_indices, [1.7 + 0.1j, 1])
    np.testing.assert_array_equal(placement.particle_indices, [0, -1])
    np.testing.assert_array_equal(placement.host_source.source_positions(), [[150, 20, 0]])
    for array in (
        placement.positions,
        placement.moments,
        placement.particle_indices,
        placement.local_refractive_indices,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_refining_native_response_reuses_placement_without_mutating_previous_order(monkeypatch):
    particles = ParticleCollection.from_particles((Sphere((0, 0, 0), 50, 1.7),))
    source = DipoleSource(550, position=(10, 0, 0))
    initial = prepare_dipole_source(particles, source, lmax=2)
    original = initial.embedded[0].response.regular.copy()

    def no_placement(*args, **kwargs):
        raise AssertionError("Order refinement must not reclassify geometry")

    monkeypatch.setattr(embedded_sources, "locate_dipole_source", no_placement)
    refined = prepare_dipole_source(particles, source, lmax=4, placement=initial.placement)
    assert refined.placement is initial.placement
    assert refined.embedded[0].response.lmax == 4
    np.testing.assert_array_equal(initial.embedded[0].response.regular, original)


def test_completed_channel_reuses_response_and_only_refines_local_order(monkeypatch):
    source = DipoleSource(550, position=(10, 0, 0))
    run = Simulation(
        SimulationConfig(wavelength=550, lmax=2, solver_method="direct", verbose=False),
        particles=(Sphere((0, 0, 0), 50, 1.7),),
    ).run(source, include_farfield=False)
    cached = run._dipole_source
    assert cached is not None

    def no_placement(*args, **kwargs):
        raise AssertionError("A completed channel must retain its source placement")

    monkeypatch.setattr(embedded_sources, "locate_dipole_source", no_placement)
    unchanged, _, _, _ = _prepare_diagnostic(run, None)
    assert unchanged is cached
    refined, _, _, _ = _prepare_diagnostic(run, 4)
    assert refined is not cached and refined.placement is cached.placement
    assert refined.lmax == 4 and cached.lmax == 2


def test_rebinding_channel_source_drops_dependent_response():
    simulation = Simulation(
        SimulationConfig(wavelength=550, lmax=2, solver_method="direct", verbose=False),
        particles=(Sphere((0, 0, 0), 50, 1.7),),
    )
    source = DipoleSource(550, position=(10, 0, 0))
    run = simulation.run(source, include_farfield=False)
    assert run._dipole_source is not None
    other = replace(source, position=(15, 0, 0), dipole_moment=(0, 1, 0))
    rebound = replace(run, source=other)
    assert rebound._dipole_source is None
    # One isolated particle has no external reception. Its regular reaction
    # depends only on the local source/interface response, not the old solve.
    fresh = simulation.run(other, include_farfield=False)
    np.testing.assert_allclose(
        compute_dipole_reaction_field(rebound),
        compute_dipole_reaction_field(fresh),
        rtol=2e-13,
        atol=0,
    )


def test_periodic_outgoing_orders_need_placement_without_preparing_local_response(monkeypatch):
    from pyceles.core.lattice import RectangularLattice2D
    from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
    from pyceles.postprocessing.farfield.periodic import compute_periodic_outgoing_amplitudes
    from pyceles.simulation import ResultRetention

    host = 1.33 + 0.08j
    source = DipoleCollection(
        550,
        medium_n=host,
        positions=np.array([[10, 0, 0], [180, 120, 20]]),
        dipole_moments=np.array([[1, 0.2j, 0], [0, 0.7, 0.1j]]),
    )
    run = Simulation(
        SimulationConfig(
            wavelength=550,
            n_medium=host,
            lmax=1,
            solver_method="direct",
            verbose=False,
            periodic=PeriodicSpec(
                RectangularLattice2D(600, 650),
                PeriodicOptions(eta=0.003, real_shells=3, reciprocal_shells=4),
                k_parallel=(0.0013, -0.0007),
            ),
        ),
        particles=(Sphere((0, 0, 0), 35, 1.7),),
    ).run(source, include_farfield=False, retention=ResultRetention.minimal())
    assert run._dipole_source is None

    def no_matching(*args, **kwargs):
        raise AssertionError("Outgoing orders only need the host-source subset")

    monkeypatch.setattr(embedded_sources, "prepare_embedded_dipole", no_matching)
    outgoing = compute_periodic_outgoing_amplitudes(run, output_bmax=0.02)
    assert np.isfinite(outgoing.up_amplitudes).all()
    assert np.linalg.norm(outgoing.up_amplitudes) > 0


@pytest.mark.api_contract
def test_matching_material_boundaries_do_not_restrict_placement():
    particle = LayeredSphere(
        position=(0, 0, 0), layer_radii=(50, 80), layer_refractive_indices=(1.5, 1.5)
    )
    part = ParticleCollection.from_particles((particle,))
    source = DipoleSource(550, position=(50, 0, 0))
    assert prepare_dipole_source(part, source, lmax=2).embedded
    with pytest.raises(ValueError, match="material interface"):
        prepare_dipole_source(part, DipoleSource(550, position=(80, 0, 0)), lmax=2)
    homogeneous = ParticleCollection.from_particles(
        (Sphere(position=(0, 0, 0), radius=80, refractive_index=source.medium_n),)
    )
    prepared = prepare_dipole_source(homogeneous, source, lmax=2)
    assert not prepared.embedded and prepared.host_source is source


def test_transparent_fractional_centre_in_another_cell_does_not_receive_singular_illumination():
    from pyceles.core.lattice import RectangularLattice2D

    centre = np.array([-0.1, 20.2, 0.3])
    particle = Sphere(tuple(centre), 35, 1.33 + 0.08j)
    source = DipoleSource(
        550,
        medium_n=particle.refractive_index,
        position=tuple(centre + np.array([600, -650, 0])),
    )
    placement = locate_dipole_source(
        ParticleCollection.from_particles((particle,)),
        source,
        lattice=RectangularLattice2D(600, 650),
        k_parallel=np.zeros(2),
    )
    assert placement.nonreceiving_particles == (0,)
    assert placement.host_source is source and not placement.embedded


def test_nonemitting_imported_particles_retain_dense_mode_coupling():
    lmax = 2
    matrix = np.eye(n_modes(lmax), dtype=complex) * (0.01 + 0.02j)
    matrix[0, 7] = 0.03 - 0.01j
    matrix[5, 2] = -0.02j
    particles = (
        Sphere(position=(0, 0, 0), radius=50, refractive_index=1.7 + 0.1j),
        TMatrixParticle(position=(300, 0, 0), radius=40, lmax=lmax, t_matrix=matrix),
    )
    source = DipoleSource(550, position=(10, 0, 0), dipole_moment=(1, 0.3j, -0.2))
    simulation = Simulation(
        SimulationConfig(
            wavelength=550, lmax=lmax, solver_method="direct", radial_lut_dr=0.25, verbose=False
        ),
        particles=particles,
    )
    run = simulation.run(source, include_farfield=False)
    k = 2 * np.pi / 550
    w10 = translation_block_rect(lmax, lmax, k, np.array([300, 0, 0]))
    w01 = translation_block_rect(lmax, lmax, k, np.array([-300, 0, 0]))
    emission = prepare_embedded_dipole(particles[0], source, lmax=lmax).exterior_coefficients
    sphere_t = particle_T_matrix_block(lmax, k, particles[0])
    for actual, expected in (
        (run.coeffs[1], matrix @ w10 @ run.coeffs[0]),
        (run.coeffs[0], sphere_t @ w01 @ run.coeffs[1] + emission),
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-23)
    with pytest.raises(NotImplementedError, match="TMatrixParticle"):
        prepare_dipole_source(
            simulation.particles, DipoleSource(550, position=(300, 0, 0)), lmax=lmax
        )


def test_host_emitter_must_be_outside_nonspherical_circumscribing_sphere():
    particles = ParticleCollection.from_particles(
        (
            Spheroid(
                position=(0, 0, 0), equatorial_radius=20, polar_radius=100, refractive_index=1.7
            ),
        )
    )
    # Outside the prolate surface, but a regular incident expansion about the
    # particle centre still cannot represent this source over the whole body.
    with pytest.raises(ValueError, match="circumscribing"):
        prepare_dipole_source(particles, DipoleSource(550, position=(30, 0, 0)), lmax=2)
