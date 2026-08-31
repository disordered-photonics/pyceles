from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    AxisymmetricTGroup,
    CompositeParticleTOperator,
    DenseTGroup,
    DiagonalTGroup,
    MLFMMOptions,
    PairwiseCouplingOperator,
    ParticleTGroupFactories,
    PreparedOperator,
    apply_A_numpy,
    assemble_dense_A_numpy,
    estimate_translation_cache_bytes,
    make_axisymmetric_block_group_factory,
    make_axisymmetric_group_factory,
    make_dense_group_factory,
    plan_particle_t_groups,
    precompute_T_diagonal,
    prepare_matvec,
    require_pairwise_coupling,
    rhs_Tb_numpy,
)
from pyceles.core.particles import (
    LayeredSphere,
    Particle,
    ParticleCollection,
    ParticleTRepresentation,
    PECSphere,
    Sphere,
    Spheroid,
    pec_spheres_from_arrays,
    spheres_from_arrays,
)
from pyceles.core.tmatrix import particle_T_diagonal, sphere_T_diagonal
from pyceles.core.translation import RadialLUT, translation_ab5_table


@dataclass(frozen=True)
class _DenseTestParticle(Particle):
    circumscribing: float = 1.0
    scale: complex = 1.0 + 0.0j

    def circumscribing_radius(self) -> float:
        return float(self.circumscribing)

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        return "dense"


@dataclass(frozen=True)
class _ScalingCoupling:
    scale: complex

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.scale * np.asarray(x, dtype=np.complex128), dtype=np.complex128)


@dataclass(frozen=True)
class _ForcedRepresentationParticle(Particle):
    base_particle: Particle
    representation: ParticleTRepresentation

    def circumscribing_radius(self) -> float:
        return float(self.base_particle.circumscribing_radius())

    @property
    def t_operator_representation(self) -> ParticleTRepresentation:
        return self.representation


def _wrapped_particle_t_diag(group_particles, context):
    diag_rows = []
    for wrapped in group_particles:
        base = wrapped.base_particle
        Td = particle_T_diagonal(
            lmax=context.lmax,
            k_medium=context.k,
            particle=base,
            n_medium=context.n_medium,
        )
        diag_rows.append(
            np.concatenate(
                [
                    np.repeat(Td[1][1:], 2 * np.arange(1, context.lmax + 1) + 1),
                    np.repeat(Td[2][1:], 2 * np.arange(1, context.lmax + 1) + 1),
                ]
            )
        )
    return np.asarray(diag_rows, dtype=context.dtype)


def _sample_problem():
    lmax = 3
    k = 2 * np.pi / 550.0
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [120.0, 15.0, -40.0],
            [-60.0, 45.0, 35.0],
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
    n_medium = 1.0 + 0j
    Nm = n_modes(lmax)
    Ns = positions.shape[0]
    rng = np.random.default_rng(7)
    x = rng.standard_normal(Ns * Nm) + 1j * rng.standard_normal(Ns * Nm)
    b = rng.standard_normal(Ns * Nm) + 1j * rng.standard_normal(Ns * Nm)
    return lmax, k, positions, radii, n_particle, particles, n_medium, x, b


def _mlfmm_single_level_problem():
    lmax = 2
    k = 2 * np.pi / 550.0
    positions = np.array(
        [
            [-90.0, -90.0, -90.0],
            [-72.0, -74.0, -88.0],
            [-90.0, -90.0, 90.0],
            [-72.0, -88.0, 74.0],
            [90.0, 90.0, -90.0],
            [72.0, 88.0, -74.0],
            [90.0, 90.0, 90.0],
            [88.0, 72.0, 74.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 11.0, dtype=float)
    n_particle = np.full((positions.shape[0],), 1.59 + 0.0j, dtype=np.complex128)
    particles = spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )
    return lmax, k, positions, particles


def _mlfmm_direct_fallback_problem():
    lmax = 2
    k = 2 * np.pi / 550.0
    positions = np.array(
        [
            [-220.0, 0.0, 0.0],
            [220.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    radii = np.full((positions.shape[0],), 80.0, dtype=float)
    n_particle = np.full((positions.shape[0],), 1.59 + 0.0j, dtype=np.complex128)
    particles = spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )
    return lmax, k, positions, particles


def test_prepare_matvec_matches_explicit_operator_kernels():
    lmax, k, positions, _, _, particles, n_medium, x, b = _sample_problem()
    T_M, T_N = precompute_T_diagonal(lmax=lmax, k=k, particles=particles, n_medium=n_medium)
    ab5 = translation_ab5_table(lmax)
    rmax = float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)))
    lut = RadialLUT(lmax=lmax, k=k, r_max=rmax, dr=0.5)

    A_ref = apply_A_numpy(lmax, k, positions, x, T_M=T_M, T_N=T_N, ab5=ab5, radial_lut=lut)
    rhs_ref = rhs_Tb_numpy(lmax, b, T_M=T_M, T_N=T_N)

    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=True,
    )

    A_prepared = prepared.apply_A(x)
    rhs_prepared = prepared.rhs_Tb(b)

    np.testing.assert_allclose(A_prepared, A_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rhs_prepared, rhs_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pec", [False, True])
def test_array_native_sphere_t_preparation_matches_explicit_descriptors(pec: bool):
    positions = np.array(
        [[-140.0, 0.0, 0.0], [140.0, 0.0, 0.0], [0.0, 170.0, 0.0]],
        dtype=float,
    )
    radii = np.array([55.0, 60.0, 55.0])
    explicit_particles: list[Particle]
    if pec:
        array_particles = pec_spheres_from_arrays(positions=positions, radii=radii)
        explicit_particles = [
            PECSphere(
                position=(float(position[0]), float(position[1]), float(position[2])),
                radius=float(radius),
            )
            for position, radius in zip(positions, radii, strict=True)
        ]
    else:
        indices = np.array([1.5 + 0.0j, 1.7 + 0.02j, 1.5 + 0.0j])
        array_particles = spheres_from_arrays(
            positions=positions,
            radii=radii,
            refractive_indices=indices,
        )
        explicit_particles = [
            Sphere(
                position=(float(position[0]), float(position[1]), float(position[2])),
                radius=float(radius),
                refractive_index=complex(index),
            )
            for position, radius, index in zip(positions, radii, indices, strict=True)
        ]

    array_prepared = prepare_matvec(
        particles=array_particles,
        lmax=3,
        k=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=1.0,
    )
    explicit_prepared = prepare_matvec(
        particles=explicit_particles,
        lmax=3,
        k=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=1.0,
    )
    array_diagonal = array_prepared.particle_t.mode_diagonal()
    explicit_diagonal = explicit_prepared.particle_t.mode_diagonal()

    assert array_diagonal is not None
    assert explicit_diagonal is not None
    np.testing.assert_allclose(array_diagonal, explicit_diagonal, rtol=1e-13, atol=1e-13)


def test_mixed_array_particle_batches_match_explicit_t_preparation():
    sphere = spheres_from_arrays(
        positions=np.array([[-140.0, 0.0, 0.0]]),
        radii=np.array([55.0]),
        refractive_indices=np.array([1.6 + 0.01j]),
    )
    pec = pec_spheres_from_arrays(
        positions=np.array([[140.0, 0.0, 0.0]]),
        radii=np.array([60.0]),
    )
    mixed = ParticleCollection.concatenate(sphere, pec)
    explicit = [sphere[0], pec[0]]
    mixed_prepared = prepare_matvec(
        particles=mixed,
        lmax=2,
        k=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=1.0,
    )
    explicit_prepared = prepare_matvec(
        particles=explicit,
        lmax=2,
        k=2.0 * np.pi / 550.0,
        n_medium=1.0 + 0.0j,
        radial_lut_dr=1.0,
    )
    mixed_diagonal = mixed_prepared.particle_t.mode_diagonal()
    explicit_diagonal = explicit_prepared.particle_t.mode_diagonal()

    assert mixed_diagonal is not None
    assert explicit_diagonal is not None
    np.testing.assert_allclose(mixed_diagonal, explicit_diagonal, rtol=1e-13, atol=1e-13)


def test_precompute_t_diagonal_reuses_identical_particle_kernels(monkeypatch):
    particles = [
        LayeredSphere(
            position=(0.0, 0.0, 0.0),
            layer_radii=(40.0, 90.0),
            layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
        ),
        LayeredSphere(
            position=(140.0, 0.0, 0.0),
            layer_radii=(40.0, 90.0),
            layer_refractive_indices=(1.6 + 0.0j, 1.4 + 0.0j),
        ),
    ]

    calls = {"count": 0}
    original = particle_T_diagonal

    def counted_particle_t_diagonal(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "pyceles.core.operators.prepare.particle_T_diagonal",
        counted_particle_t_diagonal,
    )

    T_M, T_N = precompute_T_diagonal(
        lmax=3,
        k=2 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
    )

    assert calls["count"] == 1
    np.testing.assert_allclose(T_M[0], T_M[1], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(T_N[0], T_N[1], rtol=1e-13, atol=1e-13)


def test_prepare_matvec_exposes_composite_particle_t_operator():
    lmax, k, _, _, _, particles, n_medium, x, _ = _sample_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )

    particle_t = prepared.particle_t
    assert isinstance(particle_t, CompositeParticleTOperator)
    assert len(particle_t.groups) == 1
    group = particle_t.groups[0]
    assert isinstance(group, DiagonalTGroup)
    np.testing.assert_array_equal(group.particle_indices, np.arange(len(particles), dtype=np.int64))

    Nm = n_modes(lmax)
    x2 = np.asarray(x).reshape(len(particles), Nm)
    np.testing.assert_allclose(
        prepared.particle_t.apply(x),
        (group.T_diag * x2).reshape(-1),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("coupling_backend", ["pairwise", "mlfmm"])
def test_prepare_matvec_accepts_pec_spheres_in_coupling_backends(
    coupling_backend: Literal["pairwise", "mlfmm"],
):
    lmax = 1
    k = 2 * np.pi / 550.0
    particles = [
        PECSphere(position=(-140.0, 0.0, 0.0), radius=55.0),
        PECSphere(position=(140.0, 0.0, 0.0), radius=60.0),
    ]
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        coupling_backend=coupling_backend,
        mlfmm_options=MLFMMOptions(max_leaf_particles=1, max_depth=2),
    )
    x = np.arange(len(particles) * n_modes(lmax), dtype=np.float64).astype(np.complex128)
    y = prepared.apply_A(x)
    rhs = prepared.rhs_Tb(x)

    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(rhs))
    assert np.linalg.norm(y) > 0.0
    assert np.linalg.norm(rhs) > 0.0


def test_plan_particle_t_groups_marks_axisymmetric_particles_separately():
    _, _, positions, radii, n_particle, _, _, _, _ = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        LayeredSphere(
            position=tuple(positions[1].tolist()),
            layer_radii=(40.0, float(radii[1])),
            layer_refractive_indices=(1.7 + 0j, complex(n_particle[1])),
        ),
        Spheroid(
            position=tuple(positions[2].tolist()),
            equatorial_radius=float(radii[2]),
            polar_radius=float(radii[2]) * 1.5,
            refractive_index=complex(n_particle[2]),
        ),
    ]
    plans = plan_particle_t_groups(particles)

    assert len(plans) == 2
    assert plans[0].representation == "diagonal"
    assert plans[1].representation == "axisymmetric"
    np.testing.assert_array_equal(plans[0].particle_indices, np.array([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(plans[1].particle_indices, np.array([2], dtype=np.int64))


def test_prepare_matvec_accepts_rotated_spheroids_with_default_axisymmetric_path():
    lmax, k, positions, radii, n_particle, _, n_medium, _, _ = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        Spheroid(
            position=tuple(positions[1].tolist()),
            equatorial_radius=float(radii[1]),
            polar_radius=float(radii[1]) * 1.25,
            refractive_index=complex(n_particle[1]),
            euler_angles=(0.1, 0.0, 0.0),
        ),
    ]

    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )

    assert isinstance(prepared.particle_t, CompositeParticleTOperator)
    assert len(prepared.particle_t.groups) == 2
    assert isinstance(prepared.particle_t.groups[0], DiagonalTGroup)
    assert isinstance(prepared.particle_t.groups[1], AxisymmetricTGroup)
    assert prepared.particle_t.groups[1].T_blocks is not None


def test_prepare_matvec_accepts_default_aligned_spheroid_axisymmetric_path():
    lmax = 3
    k = 2 * np.pi / 550.0
    n_medium = 1.0 + 0j
    particles: list[Particle] = [
        Sphere(
            position=(0.0, 0.0, 0.0),
            radius=80.0,
            refractive_index=1.5 + 0.0j,
        ),
        Spheroid(
            position=(220.0, 0.0, 0.0),
            equatorial_radius=70.0,
            polar_radius=100.0,
            refractive_index=1.4 + 0.0j,
        ),
    ]

    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )

    assert isinstance(prepared.particle_t, CompositeParticleTOperator)
    assert len(prepared.particle_t.groups) == 2
    assert isinstance(prepared.particle_t.groups[0], DiagonalTGroup)
    assert isinstance(prepared.particle_t.groups[1], AxisymmetricTGroup)
    assert prepared.particle_t.groups[1].T_blocks is not None


def test_prepare_matvec_accepts_custom_dense_group_factory():
    lmax, k, positions, radii, n_particle, _, n_medium, x, b = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        _DenseTestParticle(
            position=tuple(positions[1].tolist()),
            circumscribing=float(radii[1]),
            scale=1.3 + 0.2j,
        ),
    ]

    def provide_dense_blocks(group_particles, context):
        nm = context.n_modes
        assert len(group_particles) == 1
        scale = complex(group_particles[0].scale)
        base = np.eye(nm, dtype=context.dtype) * scale
        base[0, 1] = 0.25 - 0.1j
        return base[None, :, :]

    factories = ParticleTGroupFactories(dense=make_dense_group_factory(provide_dense_blocks))
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        particle_t_group_factories=factories,
    )

    assert isinstance(prepared.particle_t, CompositeParticleTOperator)
    assert isinstance(prepared.particle_t.groups[1], DenseTGroup)
    rhs = prepared.rhs_Tb(b[: 2 * n_modes(lmax)])
    y_dense = (
        assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
        @ x[: 2 * n_modes(lmax)]
    )
    y_mv = prepared.apply_A(x[: 2 * n_modes(lmax)])
    np.testing.assert_allclose(y_mv, y_dense, rtol=1e-12, atol=1e-12)
    assert rhs.shape == (2 * n_modes(lmax),)


def test_prepare_matvec_accepts_custom_axisymmetric_group_factory():
    lmax, k, positions, radii, n_particle, _, n_medium, x, b = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        Spheroid(
            position=tuple(positions[1].tolist()),
            equatorial_radius=float(radii[1]),
            polar_radius=float(radii[1]) * 1.25,
            refractive_index=complex(n_particle[1]),
            euler_angles=(0.2, 0.4, 0.1),
        ),
    ]
    nm = n_modes(lmax)
    axis_matrix = np.eye(nm, dtype=np.complex128) * (0.8 + 0.1j)
    axis_matrix[0, 2] = 0.15 - 0.05j

    factories = ParticleTGroupFactories(
        axisymmetric=make_axisymmetric_group_factory(
            apply_subset=lambda x_subset, _particles, _context: x_subset @ axis_matrix.T,
            apply_local_block=lambda local_i, block, _particles, _context: axis_matrix @ block,
            metadata_builder=lambda group_particles, _context: {
                "euler_angles": tuple(group_particles[0].euler_angles)  # type: ignore[attr-defined]
            },
        )
    )
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        particle_t_group_factories=factories,
    )

    assert isinstance(prepared.particle_t, CompositeParticleTOperator)
    assert isinstance(prepared.particle_t.groups[1], AxisymmetricTGroup)
    assert prepared.particle_t.groups[1].body_metadata is not None
    y_dense = (
        assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
        @ x[: 2 * nm]
    )
    y_mv = prepared.apply_A(x[: 2 * nm])
    rhs = prepared.rhs_Tb(b[: 2 * nm])
    np.testing.assert_allclose(y_mv, y_dense, rtol=1e-12, atol=1e-12)
    assert rhs.shape == (2 * nm,)


def test_prepare_matvec_accepts_axisymmetric_block_group_factory():
    lmax, k, positions, radii, n_particle, _, n_medium, x, b = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        Spheroid(
            position=tuple(positions[1].tolist()),
            equatorial_radius=float(radii[1]),
            polar_radius=float(radii[1]) * 1.25,
            refractive_index=complex(n_particle[1]),
            euler_angles=(0.2, 0.4, 0.1),
        ),
    ]
    nm = n_modes(lmax)
    axis_matrix = np.eye(nm, dtype=np.complex128) * (0.9 - 0.05j)
    axis_matrix[1, 4] = -0.1 + 0.03j

    factories = ParticleTGroupFactories(
        axisymmetric=make_axisymmetric_block_group_factory(
            lambda _group_particles, _context: axis_matrix[None, :, :],
            metadata_builder=lambda group_particles, _context: {
                "euler_angles": tuple(group_particles[0].euler_angles)  # type: ignore[attr-defined]
            },
        )
    )
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        particle_t_group_factories=factories,
    )

    assert isinstance(prepared.particle_t, CompositeParticleTOperator)
    assert isinstance(prepared.particle_t.groups[1], AxisymmetricTGroup)
    assert prepared.particle_t.groups[1].T_blocks is not None
    y_dense = (
        assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
        @ x[: 2 * nm]
    )
    y_mv = prepared.apply_A(x[: 2 * nm])
    rhs = prepared.rhs_Tb(b[: 2 * nm])
    np.testing.assert_allclose(y_mv, y_dense, rtol=1e-12, atol=1e-12)
    assert rhs.shape == (2 * nm,)


@pytest.mark.reference
def test_sphere_and_layered_match_when_forced_through_dense_factory():
    lmax, k, positions, radii, n_particle, _, n_medium, x, b = _sample_problem()
    base_particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        LayeredSphere(
            position=tuple(positions[1].tolist()),
            layer_radii=(40.0, float(radii[1])),
            layer_refractive_indices=(1.7 + 0j, complex(n_particle[1])),
        ),
    ]
    reference = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=base_particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )
    forced_particles = [
        _ForcedRepresentationParticle(
            position=particle.position,
            base_particle=particle,
            representation="dense",
        )
        for particle in base_particles
    ]

    factories = ParticleTGroupFactories(
        dense=make_dense_group_factory(
            lambda group_particles, context: np.stack(
                [np.diag(row) for row in _wrapped_particle_t_diag(group_particles, context)],
                axis=0,
            )
        )
    )
    forced = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=forced_particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        particle_t_group_factories=factories,
    )

    n = len(base_particles) * n_modes(lmax)
    np.testing.assert_allclose(
        forced.apply_A(x[:n]), reference.apply_A(x[:n]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        forced.rhs_Tb(b[:n]), reference.rhs_Tb(b[:n]), rtol=1e-12, atol=1e-12
    )


@pytest.mark.reference
def test_sphere_and_layered_match_when_forced_through_axisymmetric_factory():
    lmax, k, positions, radii, n_particle, _, n_medium, x, b = _sample_problem()
    base_particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        LayeredSphere(
            position=tuple(positions[1].tolist()),
            layer_radii=(40.0, float(radii[1])),
            layer_refractive_indices=(1.7 + 0j, complex(n_particle[1])),
        ),
    ]
    reference = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=base_particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )
    forced_particles = [
        _ForcedRepresentationParticle(
            position=particle.position,
            base_particle=particle,
            representation="axisymmetric",
        )
        for particle in base_particles
    ]

    factories = ParticleTGroupFactories(
        axisymmetric=make_axisymmetric_group_factory(
            apply_subset=lambda x_subset, group_particles, context: (
                _wrapped_particle_t_diag(group_particles, context) * x_subset
            ),
            apply_local_block=lambda local_i, block, group_particles, context: (
                _wrapped_particle_t_diag(group_particles, context)[local_i][:, None] * block
            ),
            metadata_builder=lambda group_particles, context: {
                "forced_from_diagonal": True,
                "n_particles": len(group_particles),
                "n_modes": context.n_modes,
            },
        )
    )
    forced = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=forced_particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        particle_t_group_factories=factories,
    )

    n = len(base_particles) * n_modes(lmax)
    assert isinstance(forced.particle_t, CompositeParticleTOperator)
    assert isinstance(forced.particle_t.groups[0], AxisymmetricTGroup)
    np.testing.assert_allclose(
        forced.apply_A(x[:n]), reference.apply_A(x[:n]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        forced.rhs_Tb(b[:n]), reference.rhs_Tb(b[:n]), rtol=1e-12, atol=1e-12
    )


def test_prepare_matvec_rejects_dense_representation_without_canonical_block_dispatch():
    lmax, k, positions, radii, n_particle, _, n_medium, _x, _b = _sample_problem()
    base_particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        LayeredSphere(
            position=tuple(positions[1].tolist()),
            layer_radii=(40.0, float(radii[1])),
            layer_refractive_indices=(1.7 + 0j, complex(n_particle[1])),
        ),
    ]
    forced_particles = [
        _ForcedRepresentationParticle(
            position=particle.position,
            base_particle=particle,
            representation="dense",
        )
        for particle in base_particles
    ]

    with pytest.raises(NotImplementedError):
        prepare_matvec(
            lmax=lmax,
            k=k,
            particles=forced_particles,
            n_medium=n_medium,
            radial_lut_dr=0.5,
            cache_translation_blocks=False,
        )


@pytest.mark.reference
def test_precompute_t_diagonal_matches_per_sphere_reference():
    lmax, k, _, radii, n_particle, particles, n_medium, _, _ = _sample_problem()
    T_M, T_N = precompute_T_diagonal(lmax=lmax, k=k, particles=particles, n_medium=n_medium)

    for i in range(len(particles)):
        Td = sphere_T_diagonal(
            lmax=lmax,
            k_medium=k,
            radius=float(radii[i]),
            n_particle=complex(n_particle[i]),
            n_medium=n_medium,
        )
        np.testing.assert_allclose(T_M[i], Td[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(T_N[i], Td[2], rtol=1e-12, atol=1e-12)


def test_prepare_matvec_accepts_mixed_sphere_and_layered():
    lmax, k, positions, radii, n_particle, _, n_medium, x, _ = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        LayeredSphere(
            position=tuple(positions[1].tolist()),
            layer_radii=(40.0, float(radii[1])),
            layer_refractive_indices=(1.7 + 0j, complex(n_particle[1])),
        ),
        Sphere(
            position=tuple(positions[2].tolist()),
            radius=float(radii[2]),
            refractive_index=complex(n_particle[2]),
        ),
    ]
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )
    y = prepared.apply_A(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


@pytest.mark.gpu
def test_prepare_matvec_cupy_matches_numpy_for_mixed_particle_groups():
    lmax, k, positions, radii, n_particle, _, n_medium, _, _ = _sample_problem()
    particles: list[Particle] = [
        Sphere(
            position=tuple(positions[0].tolist()),
            radius=float(radii[0]),
            refractive_index=complex(n_particle[0]),
        ),
        Spheroid(
            position=tuple(positions[1].tolist()),
            equatorial_radius=float(radii[1]),
            polar_radius=float(radii[1]) * 1.25,
            refractive_index=complex(n_particle[1]),
        ),
    ]
    rng = np.random.default_rng(11)
    x = np.asarray(
        rng.standard_normal(len(particles) * n_modes(lmax))
        + 1j * rng.standard_normal(len(particles) * n_modes(lmax)),
        dtype=np.complex128,
    )

    prepared_numpy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        backend="numpy",
    )
    prepared_cupy = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        backend="cupy",
    )

    np.testing.assert_allclose(
        prepared_cupy.apply_A(x),
        prepared_numpy.apply_A(x),
        rtol=5e-9,
        atol=5e-10,
    )


@pytest.mark.gpu
def test_prepare_matvec_cupy_rejects_translation_block_cache():
    lmax, k, _, _, _, particles, n_medium, _, _ = _sample_problem()

    with pytest.raises(NotImplementedError, match="direct raw-kernel coupling path"):
        prepare_matvec(
            lmax=lmax,
            k=k,
            particles=particles,
            n_medium=n_medium,
            radial_lut_dr=0.5,
            cache_translation_blocks=True,
            backend="cupy",
        )


def test_apply_A_lookup_vs_direct_coupling_agree():
    """Lookup-coupling path should match explicit per-distance coupling evaluation."""
    lmax, k, positions, _, _, particles, n_medium, x, _ = _sample_problem()
    T_M, T_N = precompute_T_diagonal(lmax=lmax, k=k, particles=particles, n_medium=n_medium)
    ab5 = translation_ab5_table(lmax)
    rmax = float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)))
    # Use a dense lookup spacing so interpolation error stays below strict
    # lookup-vs-direct tolerance.
    lut = RadialLUT(lmax=lmax, k=k, r_max=rmax, dr=0.05)

    y_lut = apply_A_numpy(lmax, k, positions, x, T_M=T_M, T_N=T_N, ab5=ab5, radial_lut=lut)
    y_direct = apply_A_numpy(lmax, k, positions, x, T_M=T_M, T_N=T_N, ab5=ab5, radial_lut=None)
    np.testing.assert_allclose(y_lut, y_direct, rtol=5e-6, atol=1e-9)


def test_block_cache_fills_once_and_reuses():
    lmax, k, positions, _, _, particles, n_medium, x, _ = _sample_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=True,
    )

    Ns = positions.shape[0]
    expected_blocks = Ns * (Ns - 1)
    coupling = require_pairwise_coupling(prepared.coupling)

    _ = prepared.apply_A(x)
    cache_size_after_first = len(coupling._W_cache)
    _ = prepared.apply_A(x)
    cache_size_after_second = len(coupling._W_cache)

    assert cache_size_after_first == expected_blocks
    assert cache_size_after_second == expected_blocks


def test_prepared_operator_accepts_generic_coupling_protocol():
    lmax, k, _positions, _, _, particles, n_medium, x, _ = _sample_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )
    generic_prepared = PreparedOperator(
        lmax=prepared.lmax,
        k=prepared.k,
        positions=prepared.positions,
        particle_t=prepared.particle_t,
        coupling=_ScalingCoupling(0.0 + 0.0j),
        dtype=np.dtype(prepared.dtype),
    )
    np.testing.assert_allclose(generic_prepared.apply_A(x), x, rtol=1e-12, atol=1e-12)


def test_estimate_translation_cache_bytes():
    n = 500
    lmax = 3
    nm = n_modes(lmax)
    expected = n * (n - 1) * nm * nm * np.dtype(np.complex128).itemsize
    assert estimate_translation_cache_bytes(n, lmax) == expected


def test_assemble_dense_A_matches_apply_A():
    lmax, k, positions, _, _, particles, n_medium, _, _ = _sample_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=n_medium,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
    )
    A = assemble_dense_A_numpy(prepared, show_progress=False, use_cache=False, store_blocks=False)
    Nm = n_modes(lmax)
    n = positions.shape[0] * Nm
    assert A.shape == (n, n)

    rng = np.random.default_rng(3)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y_dense = A @ x
    y_mv = prepared.apply_A(x)
    np.testing.assert_allclose(y_dense, y_mv, rtol=1e-12, atol=1e-12)


def test_prepare_matvec_mlfmm_direct_stage_returns_pairwise_coupling() -> None:
    lmax, k, _, particles = _mlfmm_direct_fallback_problem()
    prepared = prepare_matvec(
        lmax=lmax,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=0.5,
        cache_translation_blocks=False,
        coupling_backend="mlfmm",
    )

    assert isinstance(prepared.coupling, PairwiseCouplingOperator)


def test_prepare_matvec_shares_diagonal_operator_across_layered_instances():
    particles = ParticleCollection.from_particles(
        [
            LayeredSphere(
                position=(float(index) * 180.0, 0.0, 0.0),
                layer_radii=(30.0, 60.0),
                layer_refractive_indices=(1.8 + 0j, 1.5 + 0.01j),
            )
            for index in range(3)
        ]
    )
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
    )

    particle_t = prepared.particle_t
    assert isinstance(particle_t, CompositeParticleTOperator)
    group = particle_t.groups[0]
    assert isinstance(group, DiagonalTGroup)
    assert group.T_diag.shape[0] == 1
    np.testing.assert_array_equal(group.operator_indices, np.zeros(3, dtype=np.int64))

    x = np.arange(3 * n_modes(1), dtype=float).reshape(3, n_modes(1)).astype(np.complex128)
    expected = group.T_diag[0] * x
    np.testing.assert_allclose(particle_t.apply(x.reshape(-1)), expected.reshape(-1))
    block = np.column_stack((x.reshape(-1), (0.25 - 0.5j) * x.reshape(-1)))
    expected_block = np.column_stack((expected.reshape(-1), (0.25 - 0.5j) * expected.reshape(-1)))
    np.testing.assert_allclose(particle_t.apply(block), expected_block)
    np.testing.assert_allclose(particle_t.rhs(block), expected_block)


def test_dense_factory_receives_unique_archetypes_and_reuses_blocks():
    particles = ParticleCollection.from_particles(
        [
            _DenseTestParticle(
                position=(float(index) * 100.0, 0.0, 0.0),
                circumscribing=20.0,
                scale=1.3 + 0.2j,
            )
            for index in range(3)
        ]
    )
    calls = {"count": 0}

    def provide_dense_blocks(group_archetypes, context):
        calls["count"] += 1
        assert len(group_archetypes) == 1
        assert group_archetypes[0].position == (0.0, 0.0, 0.0)
        block = np.eye(context.n_modes, dtype=context.dtype) * group_archetypes[0].scale
        block[0, 1] = 0.25 - 0.1j
        return block[None, :, :]

    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        particle_t_group_factories=ParticleTGroupFactories(
            dense=make_dense_group_factory(provide_dense_blocks)
        ),
    )

    particle_t = cast(CompositeParticleTOperator, prepared.particle_t)
    group = particle_t.groups[0]
    assert isinstance(group, DenseTGroup)
    assert calls["count"] == 1
    assert group.T_blocks.shape[0] == 1
    np.testing.assert_array_equal(group.operator_indices, np.zeros(3, dtype=np.int64))

    x = np.arange(3 * n_modes(1), dtype=float).reshape(3, n_modes(1)).astype(np.complex128)
    expected = np.einsum("ij,gj->gi", group.T_blocks[0], x)
    np.testing.assert_allclose(particle_t.apply(x.reshape(-1)), expected.reshape(-1))
    block = np.column_stack((x.reshape(-1), 2j * x.reshape(-1)))
    expected_block = np.column_stack((expected.reshape(-1), 2j * expected.reshape(-1)))
    np.testing.assert_allclose(particle_t.apply(block), expected_block)


def test_shared_operator_maps_handle_repeated_noncontiguous_archetypes():
    particles = ParticleCollection.from_particles(
        [
            Sphere(position=(0.0, 0.0, 0.0), radius=20.0, refractive_index=1.5 + 0j),
            Sphere(position=(100.0, 0.0, 0.0), radius=25.0, refractive_index=1.6 + 0j),
            Sphere(position=(200.0, 0.0, 0.0), radius=20.0, refractive_index=1.5 + 0j),
        ]
    )
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
    )

    particle_t = cast(CompositeParticleTOperator, prepared.particle_t)
    group = particle_t.groups[0]
    assert isinstance(group, DiagonalTGroup)
    assert group.T_diag.shape[0] == 2
    np.testing.assert_array_equal(group.operator_indices, np.array([0, 1, 0]))

    x = np.arange(3 * n_modes(1), dtype=float).reshape(3, n_modes(1)).astype(np.complex128)
    expected = np.empty_like(x)
    expected[0] = group.T_diag[0] * x[0]
    expected[1] = group.T_diag[1] * x[1]
    expected[2] = group.T_diag[0] * x[2]
    np.testing.assert_allclose(particle_t.apply(x.reshape(-1)), expected.reshape(-1))
    np.testing.assert_allclose(particle_t.apply(x), expected)
    np.testing.assert_allclose(particle_t.rhs(x), expected)
    block = np.column_stack((x.reshape(-1), (0.5 + 0.25j) * x.reshape(-1)))
    expected_block = np.column_stack((expected.reshape(-1), (0.5 + 0.25j) * expected.reshape(-1)))
    np.testing.assert_allclose(particle_t.apply(block), expected_block)
    np.testing.assert_allclose(particle_t.rhs(block), expected_block)


@pytest.mark.fake_gpu
def test_cupy_group_wrappers_preserve_shared_operator_maps(monkeypatch):
    from pyceles.core.operators import single_body_cupy

    monkeypatch.setattr(single_body_cupy, "import_cupy", lambda: (np, None))
    monkeypatch.setattr(
        single_body_cupy,
        "coerce_array",
        lambda value, *, dtype, prefer_cupy: np.asarray(value, dtype=dtype),
    )

    particle_indices = np.arange(3, dtype=np.int64)
    operator_indices = np.array([0, 1, 0], dtype=np.int64)
    diagonal = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    diagonal_group = single_body_cupy.CuPyDiagonalTGroup(
        particle_indices=particle_indices,
        T_M=np.ones((2, 2), dtype=np.complex128),
        T_N=np.ones((2, 2), dtype=np.complex128),
        T_diag=diagonal,
        operator_indices=operator_indices,
    )
    x = np.arange(6, dtype=float).reshape(3, 2).astype(np.complex128)
    np.testing.assert_allclose(
        np.asarray(diagonal_group.apply_subset(x)),
        diagonal[operator_indices] * x,
    )

    blocks = np.stack([np.eye(2, dtype=np.complex128), 2.0 * np.eye(2, dtype=np.complex128)])
    dense_group = single_body_cupy.CuPyDenseTGroup(
        particle_indices=particle_indices,
        T_blocks=blocks,
        operator_indices=operator_indices,
    )
    np.testing.assert_allclose(
        np.asarray(dense_group.apply_subset(x)),
        np.einsum("gij,gj->gi", blocks[operator_indices], x),
    )


def test_prepare_matvec_rejects_unknown_coupling_before_cupy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import() -> None:
        raise AssertionError("CuPy import must not run for an invalid coupling backend")

    monkeypatch.setattr("pyceles.core.operators.prepare.import_cupy", fail_import)
    particle = Sphere(position=(0.0, 0.0, 0.0), radius=20.0, refractive_index=1.5)

    with pytest.raises(ValueError, match="Unknown coupling backend"):
        prepare_matvec(
            lmax=1,
            k=2.0 * np.pi / 550.0,
            particles=[particle],
            radial_lut_dr=1.0,
            coupling_backend="invalid",  # type: ignore[arg-type]
            backend="cupy",
        )


def test_prepare_matvec_normalizes_numpy_backend_name() -> None:
    particle = Sphere(position=(0.0, 0.0, 0.0), radius=20.0, refractive_index=1.5)
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[particle],
        radial_lut_dr=1.0,
        coupling_backend="PAIRWISE",  # type: ignore[arg-type]
        backend="NUMPY",  # type: ignore[arg-type]
    )

    assert isinstance(prepared.coupling, PairwiseCouplingOperator)
