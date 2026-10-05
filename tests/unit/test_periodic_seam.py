from __future__ import annotations

import importlib
from typing import Literal

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.indexing import n_modes
from pyceles.core.operators import (
    CuPyPeriodicCouplingOperator,
    PeriodicCouplingOperator,
    apply_W_numpy,
    prepare_matvec,
)
from pyceles.core.periodic.ewald import periodic_ewald_block
from pyceles.core.translation import translation_ab5_table
from pyceles.io import load_periodic_h5, save_periodic_h5
from pyceles.simulation import Simulation, SimulationConfig
from pyceles.simulation.solve import (
    _assemble_dense_operator_for_prepared,
    periodic_shared_k_parallel,
)


def _required_float(value: float | None) -> float:
    assert value is not None
    return value


def _plane_wave(
    *,
    polar_angle: float = 0.0,
    azimuthal_angle: float = 0.0,
    polarization: Literal["TE", "TM"] | tuple[complex, complex] = "TE",
) -> pcl.PlaneWave:
    return pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=polarization,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle,
        amplitude=1.0,
    )


def _sphere(*, radius: float = 10.0, x: float = 0.0) -> pcl.Sphere:
    return pcl.Sphere(position=(x, 0.0, 0.0), radius=radius, refractive_index=1.5 + 0j)


def test_rectangular_lattice_geometry_and_diffraction_orders() -> None:
    lattice = pcl.RectangularLattice2D(ax=100.0, ay=200.0)

    np.testing.assert_allclose(lattice.a1, [100.0, 0.0, 0.0])
    np.testing.assert_allclose(lattice.a2, [0.0, 200.0, 0.0])
    np.testing.assert_allclose(lattice.b1, [2.0 * np.pi / 100.0, 0.0])
    np.testing.assert_allclose(lattice.b2, [0.0, 2.0 * np.pi / 200.0])
    assert lattice.area == 20_000.0

    orders = lattice.diffraction_orders(k_parallel=np.zeros((2,)), k=0.02, max_order=1)
    order00 = next(order for order in orders if order.m == 0 and order.n == 0)
    order10 = next(order for order in orders if order.m == 1 and order.n == 0)
    assert order00.propagating
    assert order00.kz == 0.02 + 0.0j
    assert not order10.propagating
    assert order10.kz.imag > 0.0


@pytest.mark.parametrize(("kwargs", "match"), [({"ax": 0.0}, "ax"), ({"ay": -1.0}, "ay")])
def test_rectangular_lattice_rejects_invalid_periods(kwargs: dict[str, float], match: str) -> None:
    params = {"ax": 100.0, "ay": 100.0}
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        pcl.RectangularLattice2D(**params)


def test_periodic_config_accepts_rectangular_lattice_spec() -> None:
    cfg = SimulationConfig(
        periodic=pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0)),
        verbose=False,
    )

    assert cfg.periodic is not None
    assert cfg.periodic.lattice.area == 120_000.0


def test_periodic_options_default_to_adaptive_ewald_shells() -> None:
    options = pcl.PeriodicOptions()

    assert options.real_shells is None
    assert options.reciprocal_shells is None
    assert options.shell_tolerance == pytest.approx(1.0e-10)
    assert options.max_shells == 32


def test_periodic_options_reject_invalid_numerical_policy() -> None:
    with pytest.raises(ValueError, match="eta"):
        pcl.PeriodicOptions(eta=-0.1)
    with pytest.raises(ValueError, match="real_shells"):
        pcl.PeriodicOptions(real_shells=-1)
    with pytest.raises(ValueError, match="reciprocal_shells"):
        pcl.PeriodicOptions(reciprocal_shells=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="directsum_window"):
        pcl.PeriodicOptions(directsum_window=-1)
    with pytest.raises(ValueError, match="shell_tolerance"):
        pcl.PeriodicOptions(shell_tolerance=0.0)
    with pytest.raises(ValueError, match="max_shells"):
        pcl.PeriodicOptions(max_shells=-1)
    with pytest.raises(ValueError, match="max_shells"):
        pcl.PeriodicOptions(max_shells=0)
    with pytest.raises(ValueError, match="output_bmax"):
        pcl.PeriodicOptions(output_bmax=0.0)
    with pytest.raises(ValueError, match="rayleigh_z_cut"):
        pcl.PeriodicOptions(rayleigh_z_cut=0.0)
    with pytest.raises(ValueError, match="rayleigh_reciprocal_shells"):
        pcl.PeriodicOptions(rayleigh_reciprocal_shells=-1)


def test_periodic_config_accepts_cupy_operator_backend() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0))

    cfg = SimulationConfig(
        periodic=spec,
        operator_backend="cupy",
        verbose=False,
    )

    assert cfg.periodic == spec
    assert cfg.operator_backend == "cupy"


def test_periodic_config_rejects_cupy_directsum_method() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 400.0),
        options=pcl.PeriodicOptions(method="directsum"),
    )
    with pytest.raises(NotImplementedError, match="Ewald or Rayleigh"):
        SimulationConfig(periodic=spec, operator_backend="cupy", verbose=False)


def test_prepare_matvec_rejects_cupy_directsum_before_backend_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import() -> None:
        raise AssertionError("CuPy import must not run for rejected direct-sum preparation")

    monkeypatch.setattr("pyceles.core.operators.prepare.import_cupy", fail_import)
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 400.0),
        options=pcl.PeriodicOptions(method="directsum"),
    )

    with pytest.raises(NotImplementedError, match="Ewald or Rayleigh"):
        prepare_matvec(
            lmax=1,
            k=2.0 * np.pi / 550.0,
            particles=[_sphere()],
            radial_lut_dr=1.0,
            periodic=periodic,
            k_parallel=np.zeros(2),
            backend="cupy",
        )


def test_periodic_config_accepts_cupy_rayleigh_method() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 400.0),
        options=pcl.PeriodicOptions(method="rayleigh"),
    )
    cfg = SimulationConfig(periodic=spec, operator_backend="cupy", verbose=False)

    assert cfg.periodic == spec


def test_prepare_matvec_rejects_periodic_mlfmm_before_backend_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import() -> None:
        raise AssertionError("CuPy import must not run for rejected periodic MLFMM")

    monkeypatch.setattr("pyceles.core.operators.prepare.import_cupy", fail_import)
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 400.0),
        options=pcl.PeriodicOptions(method="rayleigh"),
    )

    with pytest.raises(NotImplementedError, match="Periodic MLFMM is not implemented"):
        prepare_matvec(
            lmax=1,
            k=2.0 * np.pi / 550.0,
            particles=[_sphere()],
            radial_lut_dr=1.0,
            coupling_backend="mlfmm",
            periodic=periodic,
            k_parallel=np.zeros(2),
            backend="cupy",
        )


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_periodic_config_rejects_mlfmm_backend(
    backend: Literal["numpy", "cupy"],
) -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 400.0),
        options=pcl.PeriodicOptions(method="rayleigh"),
    )
    with pytest.raises(NotImplementedError, match="Periodic MLFMM is not implemented"):
        SimulationConfig(
            periodic=spec,
            coupling_backend="mlfmm",
            operator_backend=backend,
            verbose=False,
        )


def test_periodic_config_rejects_non_plane_wave_embedded_source() -> None:
    source = pcl.GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        beam_width=1000.0,
        focal_point=(0.0, 0.0, 0.0),
    )
    sim = Simulation(
        SimulationConfig(
            periodic=pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0)),
            verbose=False,
        ),
        particles=[_sphere(radius=10.0)],
    )
    with pytest.raises(NotImplementedError, match="PlaneWave excitation"):
        sim.run(source)


def test_periodic_overlap_validator_checks_self_images() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(100.0, 100.0))
    cfg = SimulationConfig(periodic=spec, verbose=False)

    with pytest.raises(ValueError, match="lattice shift"):
        Simulation(cfg, particles=[_sphere(radius=60.0)])


@pytest.mark.reference
def test_periodic_overlap_validator_accepts_separated_reference_cell() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 300.0))
    cfg = SimulationConfig(periodic=spec, verbose=False)

    sim = Simulation(cfg, particles=[_sphere(radius=10.0), _sphere(radius=10.0, x=80.0)])

    assert sim.n_particles == 2


def test_periodic_overlap_validator_uses_minimum_image_for_unwrapped_positions() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(100.0, 200.0))
    cfg = SimulationConfig(periodic=spec, verbose=False)
    particles = [
        pcl.Sphere(position=(0.0, 0.0, 0.0), radius=30.0, refractive_index=1.5 + 0j),
        pcl.Sphere(position=(250.0, 0.0, 0.0), radius=30.0, refractive_index=1.5 + 0j),
    ]

    with pytest.raises(ValueError, match="lattice shift"):
        Simulation(cfg, particles=particles)


def test_periodic_shared_k_parallel_accepts_matching_plane_wave_sources() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 300.0))
    cfg = SimulationConfig(periodic=spec, verbose=False)
    sim = Simulation(cfg, particles=[_sphere()])
    src_te = _plane_wave(polar_angle=0.3, azimuthal_angle=0.4, polarization="TE")
    src_tm = _plane_wave(polar_angle=0.3, azimuthal_angle=0.4, polarization="TM")

    kp = periodic_shared_k_parallel(sim, {"te": src_te, "tm": src_tm})

    assert kp is not None
    expected = pcl.core.plane_wave_k_parallel(src_te)
    np.testing.assert_allclose(kp, expected)


def test_periodic_shared_k_parallel_rejects_mismatched_sources() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 300.0))
    cfg = SimulationConfig(periodic=spec, verbose=False)
    sim = Simulation(cfg, particles=[_sphere()])

    with pytest.raises(NotImplementedError, match="one in-plane Bloch wavevector"):
        periodic_shared_k_parallel(
            sim,
            {
                "a": _plane_wave(polar_angle=0.2, azimuthal_angle=0.0),
                "b": _plane_wave(polar_angle=0.3, azimuthal_angle=0.0),
            },
        )


def test_prepare_matvec_periodic_returns_ewald_operator() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 400.0),
        options=pcl.PeriodicOptions(method="ewald", eta=0.02, real_shells=6, reciprocal_shells=6),
    )
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3)
    particles = [_sphere(radius=10.0)]
    k = 2.0 * np.pi / 550.0

    prepared = prepare_matvec(
        lmax=1,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )

    assert isinstance(prepared.coupling, PeriodicCouplingOperator)
    assert prepared.coupling.periodic is spec
    np.testing.assert_allclose(
        prepared.coupling.k_parallel,
        pcl.core.plane_wave_k_parallel(source),
    )
    x = np.arange(6, dtype=np.float64).astype(np.complex128) + 0.25j
    expected = (
        periodic_ewald_block(
            lmax=1,
            k=k,
            destination=np.asarray(particles[0].position, dtype=float),
            source=np.asarray(particles[0].position, dtype=float),
            lattice=spec.lattice,
            k_parallel=pcl.core.plane_wave_k_parallel(source),
            eta=0.02,
            real_shells=6,
            reciprocal_shells=6,
            ab5=translation_ab5_table(1, dtype=np.complex128),
            dtype=np.complex128,
            exclude_zero_shift=True,
        )
        @ x
    )
    np.testing.assert_allclose(prepared.apply_W(x), expected, rtol=1e-12, atol=1e-12)


def test_prepare_matvec_periodic_complex64_contraction_follows_operator_dtype() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(360.0, 390.0),
        options=pcl.PeriodicOptions(
            method="ewald",
            eta=0.02,
            real_shells=2,
            reciprocal_shells=2,
        ),
    )
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.1)
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[_sphere(radius=10.0)],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=True,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        operator_dtype=np.complex64,
        accum_dtype=np.complex128,
        show_progress=False,
    )

    assert isinstance(prepared.coupling, PeriodicCouplingOperator)
    coupling = prepared.coupling
    assert coupling.ab5.dtype == np.dtype(np.complex64)
    # Exercise the policy even if a caller supplies a wider translation table.
    coupling.ab5 = translation_ab5_table(1, dtype=np.complex128)
    assert coupling._contraction_tensor().dtype == np.dtype(np.complex64)

    prepared.populate_coupling(show_progress=False)
    assert coupling._ewald_block_cache
    assert all(
        block.dtype == np.dtype(np.complex64) for block in coupling._ewald_block_cache.values()
    )

    x = np.arange(n_modes(1), dtype=np.float32).astype(np.complex64) + np.complex64(0.25j)
    y = prepared.apply_W(x)
    assert y.dtype == np.dtype(np.complex64)


def test_periodic_ewald_operator_matches_explicit_blocks_for_two_particle_cell() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(360.0, 390.0),
        options=pcl.PeriodicOptions(method="ewald", eta=0.02, real_shells=4, reciprocal_shells=4),
    )
    source = _plane_wave(polar_angle=0.25, azimuthal_angle=0.4)
    particles = [
        _sphere(radius=10.0),
        pcl.Sphere(position=(85.0, 24.0, 31.0), radius=9.0, refractive_index=1.45 + 0j),
    ]
    k = 2.0 * np.pi / 550.0
    prepared = prepare_matvec(
        lmax=1,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )
    positions = np.asarray([p.position for p in particles], dtype=float)
    nm = n_modes(1)
    x = (np.arange(2 * nm, dtype=np.float64) + 0.5j).astype(np.complex128)
    arr = x.reshape(2, nm)
    ab5 = translation_ab5_table(1, dtype=np.complex128)
    expected = np.zeros_like(arr)
    for i in range(2):
        for j in range(2):
            expected[i] += (
                periodic_ewald_block(
                    lmax=1,
                    k=k,
                    destination=positions[i],
                    source=positions[j],
                    lattice=spec.lattice,
                    k_parallel=pcl.core.plane_wave_k_parallel(source),
                    eta=0.02,
                    real_shells=4,
                    reciprocal_shells=4,
                    ab5=ab5,
                    dtype=np.complex128,
                    exclude_zero_shift=(i == j),
                )
                @ arr[j]
            )

    np.testing.assert_allclose(
        prepared.apply_W(x), expected.reshape(2 * nm), rtol=1e-12, atol=1e-12
    )


def test_periodic_ewald_coupling_populate_fills_private_block_cache() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(360.0, 390.0),
        options=pcl.PeriodicOptions(method="ewald", eta=0.02, real_shells=4, reciprocal_shells=4),
    )
    source = _plane_wave(polar_angle=0.25, azimuthal_angle=0.4)
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[
            _sphere(radius=10.0),
            pcl.Sphere(position=(85.0, 24.0, 31.0), radius=9.0, refractive_index=1.45 + 0j),
        ],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=True,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )
    assert isinstance(prepared.coupling, PeriodicCouplingOperator)

    prepared.populate_coupling(show_progress=False)

    assert set(prepared.coupling._ewald_block_cache) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_periodic_ewald_apply_keeps_block_cache_empty_when_disabled() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(360.0, 390.0),
        options=pcl.PeriodicOptions(method="ewald", eta=0.02, real_shells=2, reciprocal_shells=2),
    )
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.1)
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[_sphere(radius=10.0), _sphere(radius=9.0, x=70.0)],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )
    assert isinstance(prepared.coupling, PeriodicCouplingOperator)
    assert prepared.coupling._ewald_block_cache == {}
    x = (np.arange(12, dtype=np.float64) + 0.3j).astype(np.complex128)
    _ = prepared.apply_W(x)
    assert prepared.coupling._ewald_block_cache == {}


def test_periodic_dense_assembly_streams_without_populating_cache() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(360.0, 390.0),
        options=pcl.PeriodicOptions(method="ewald", eta=0.02, real_shells=2, reciprocal_shells=2),
    )
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.1)
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[_sphere(radius=10.0), _sphere(radius=9.0, x=70.0)],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        cache_translation_blocks=False,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )
    assert isinstance(prepared.coupling, PeriodicCouplingOperator)
    sentinel_cache = {(99, 99): np.eye(n_modes(1), dtype=np.complex128)}
    prepared.coupling._ewald_block_cache = dict(sentinel_cache)

    dense = _assemble_dense_operator_for_prepared(
        prepared=prepared,
        A_mv=prepared.apply_A,
        n=2 * n_modes(1),
        dtype=np.dtype(np.complex128),
        backend="numpy",
        show_progress=False,
    )
    eye = np.eye(2 * n_modes(1), dtype=np.complex128)
    expected = np.column_stack([prepared.apply_A(eye[:, j]) for j in range(eye.shape[1])])

    assert dense.shape == (2 * n_modes(1), 2 * n_modes(1))
    np.testing.assert_allclose(dense, expected, rtol=1e-12, atol=1e-12)
    assert prepared.coupling.cache_blocks is False
    assert set(prepared.coupling._ewald_block_cache) == set(sentinel_cache)
    np.testing.assert_array_equal(
        prepared.coupling._ewald_block_cache[(99, 99)], sentinel_cache[(99, 99)]
    )


def test_periodic_dense_assembly_applies_nondiagonal_particle_t_blocks() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(900.0, 850.0),
        options=pcl.PeriodicOptions(
            method="ewald",
            eta=0.002,
            real_shells=1,
            reciprocal_shells=1,
        ),
    )
    source = _plane_wave()
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[
            pcl.Spheroid(
                position=(0.0, 0.0, 40.0),
                equatorial_radius=40.0,
                polar_radius=50.0,
                refractive_index=1.5 + 0j,
            )
        ],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )
    assert isinstance(prepared.coupling, PeriodicCouplingOperator)
    assert prepared.particle_t.mode_diagonal() is None

    nm = n_modes(1)
    dense = _assemble_dense_operator_for_prepared(
        prepared=prepared,
        A_mv=prepared.apply_A,
        n=nm,
        dtype=np.dtype(np.complex128),
        backend="numpy",
        show_progress=False,
    )
    eye = np.eye(nm, dtype=np.complex128)
    expected = np.column_stack([prepared.apply_A(eye[:, j]) for j in range(nm)])

    np.testing.assert_allclose(dense, expected, rtol=1e-12, atol=1e-12)
    assert prepared.coupling.cache_blocks is False
    assert prepared.coupling._ewald_block_cache == {}


def test_prepare_matvec_periodic_does_not_build_radial_lut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0))
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3)
    prepare_module = importlib.import_module("pyceles.core.operators.prepare")

    def fail_radial_lut(*args: object, **kwargs: object) -> None:
        raise AssertionError("periodic preparation should not build a RadialLUT")

    monkeypatch.setattr(prepare_module, "RadialLUT", fail_radial_lut)

    prepared = prepare_module.prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[_sphere(radius=10.0)],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )

    assert isinstance(prepared.coupling, PeriodicCouplingOperator)


@pytest.mark.reference
def test_periodic_direct_sum_window_zero_matches_pairwise_reference() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(500.0, 500.0),
        options=pcl.PeriodicOptions(method="directsum", directsum_window=0),
    )
    particles = [_sphere(radius=10.0), _sphere(radius=10.0, x=90.0)]
    k = 2.0 * np.pi / 550.0
    k_parallel = np.array([0.001, 0.002])
    periodic_prepared = prepare_matvec(
        lmax=1,
        k=k,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        periodic=spec,
        k_parallel=k_parallel,
        show_progress=False,
    )
    positions = np.asarray([p.position for p in particles], dtype=float)
    x = np.arange(12, dtype=np.float64).astype(np.complex128) + 0.5j

    np.testing.assert_allclose(
        periodic_prepared.apply_W(x),
        apply_W_numpy(
            1,
            k,
            positions,
            x,
            translation_ab5_table(1, dtype=np.complex128),
            dtype=np.complex128,
            radial_lut=None,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_periodic_direct_sum_includes_self_images() -> None:
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 320.0),
        options=pcl.PeriodicOptions(method="directsum", directsum_window=1),
    )
    source = _plane_wave(polar_angle=0.25, azimuthal_angle=0.4)
    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=[_sphere(radius=10.0)],
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )
    x = np.arange(6, dtype=np.float64).astype(np.complex128) + 1.0j

    y = prepared.apply_W(x)

    assert np.linalg.norm(y) > 0.0


def test_periodic_direct_sum_solve_runs_through_dense_fallback() -> None:
    source = _plane_wave()
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 300.0),
        options=pcl.PeriodicOptions(method="directsum", directsum_window=0),
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=spec,
        solver_method="direct",
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])

    solved = sim.solve_sources({"pw": source})

    np.testing.assert_allclose(solved.coeffs["pw"], solved.rhs["pw"])


def test_periodic_ewald_solve_runs_through_dense_fallback() -> None:
    source = _plane_wave()
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 300.0),
        options=pcl.PeriodicOptions(method="ewald", eta=0.02, real_shells=6, reciprocal_shells=6),
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=spec,
        solver_method="direct",
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])

    solved = sim.solve_sources({"pw": source})

    assert np.all(np.isfinite(solved.coeffs["pw"]))


def test_cupy_periodic_dense_batches_do_not_use_cache_off_matvec_budget() -> None:
    """Dense assembly must bound its transient blocks independently of matvec batches."""

    lmax = 3
    positions = np.zeros((500, 3), dtype=float)
    coupling = CuPyPeriodicCouplingOperator(
        lmax=lmax,
        k=2.0 * np.pi / 550.0,
        positions=positions,
        ab5=translation_ab5_table(lmax, dtype=np.complex128),
        periodic=pcl.PeriodicSpec(
            lattice=pcl.RectangularLattice2D(3000.0, 3000.0),
            options=pcl.PeriodicOptions(method="ewald"),
        ),
        k_parallel=np.zeros(2, dtype=float),
        dtype=np.dtype(np.complex128),
        cache_blocks=False,
    )

    matvec_batch_size = coupling._source_batch_size()
    dense_batch_size = coupling._source_batch_size(for_dense_assembly=True)

    assert matvec_batch_size > dense_batch_size
    assert dense_batch_size <= 16


def test_periodic_postprocess_populates_periodic_result_payload() -> None:
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3, polarization="TE")
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 320.0),
        options=pcl.PeriodicOptions(method="ewald", eta=0.02, real_shells=4, reciprocal_shells=4),
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=spec,
        solver_method="direct",
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])

    solved = sim.solve_sources({"pw": source})
    multi = sim.postprocess_sources(solved, include_farfield=False)
    run = multi["pw"]

    assert run.periodic is not None
    periodic = run.periodic
    assert periodic.order_mn.shape[1] == 2
    assert periodic.order_k_parallel.shape[0] == periodic.order_mn.shape[0]
    assert periodic.order_kz.shape[0] == periodic.order_mn.shape[0]
    assert periodic.reflected_amplitudes.shape == periodic.transmitted_amplitudes.shape
    assert periodic.reflected_amplitudes.shape[1] == 2
    assert np.any((periodic.order_mn[:, 0] == 0) & (periodic.order_mn[:, 1] == 0))
    assert bool(np.all(periodic.order_propagating))
    assert periodic.output_bmax is None
    assert run.power is periodic.power
    reflectance = _required_float(periodic.power.reflectance)
    transmittance = _required_float(periodic.power.transmittance)
    local_absorptance = _required_float(periodic.power.local_absorptance)
    closure_error_fraction = _required_float(periodic.power.closure_error_fraction)
    flux_defect_fraction = _required_float(periodic.power.flux_defect_fraction)
    assert np.isfinite(reflectance)
    assert np.isfinite(transmittance)
    assert np.isfinite(local_absorptance)
    assert np.isfinite(closure_error_fraction)
    np.testing.assert_allclose(
        flux_defect_fraction,
        local_absorptance + closure_error_fraction,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert run.farfield.scattered.coeff_te.shape == (0, 0)


def test_periodic_postprocess_output_bmax_includes_evanescent_orders() -> None:
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3, polarization="TE")
    spec = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(300.0, 320.0),
        options=pcl.PeriodicOptions(output_bmax=0.05),
    )
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=spec,
        solver_method="direct",
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])

    run = sim.run(source, include_farfield=False)
    assert run.periodic is not None
    periodic = run.periodic
    assert periodic.output_bmax == pytest.approx(0.05)
    assert periodic.order_mn.shape[0] > int(np.count_nonzero(periodic.order_propagating))


def test_periodic_polarization_result_matches_direct_mixed_channel() -> None:
    source = _plane_wave(
        polar_angle=0.2,
        azimuthal_angle=0.3,
        polarization=(0.6 + 0.2j, -0.3 + 0.7j),
    )
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 320.0))
    sim = Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=1,
            periodic=spec,
            solver_method="direct",
            verbose=False,
        ),
        particles=[_sphere(radius=10.0)],
    )

    direct = sim.run(source)
    polarized = sim.run_polarizations(source)
    mixed = polarized.mixed

    assert polarized.solver_result.rhs_count == 2
    assert not hasattr(polarized.te, "solver_result")
    assert not hasattr(polarized.tm, "solver_result")
    np.testing.assert_allclose(mixed.coeffs, direct.coeffs, rtol=2e-12, atol=2e-13)
    assert mixed.periodic is not None
    assert direct.periodic is not None
    np.testing.assert_allclose(
        mixed.periodic.reflected_amplitudes,
        direct.periodic.reflected_amplitudes,
        rtol=2e-12,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        mixed.periodic.transmitted_amplitudes,
        direct.periodic.transmitted_amplitudes,
        rtol=2e-12,
        atol=2e-13,
    )
    assert mixed.power is not None
    assert direct.power is not None
    assert mixed.power.reflectance == pytest.approx(direct.power.reflectance, rel=2e-12)
    assert mixed.power.transmittance == pytest.approx(direct.power.transmittance, rel=2e-12)


@pytest.mark.api_contract
def test_periodic_polarization_reuses_basis_order_payloads(monkeypatch) -> None:
    postprocess_module = importlib.import_module("pyceles.simulation.postprocess")
    source = _plane_wave(
        polar_angle=0.2,
        azimuthal_angle=0.3,
        polarization=(0.6 + 0.2j, -0.3 + 0.7j),
    )
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 320.0))
    sim = Simulation(
        SimulationConfig(
            wavelength=550.0,
            n_medium=1.0 + 0j,
            lmax=1,
            periodic=spec,
            solver_method="direct",
            verbose=False,
        ),
        particles=[_sphere(radius=10.0)],
    )
    build_periodic_result = postprocess_module._build_periodic_result
    calls = 0

    def _counted_build_periodic_result(*args, **kwargs):
        nonlocal calls
        calls += 1
        return build_periodic_result(*args, **kwargs)

    monkeypatch.setattr(
        postprocess_module,
        "_build_periodic_result",
        _counted_build_periodic_result,
    )

    polarized = sim.run_polarizations(source)

    assert calls == 2
    assert polarized.mixed.periodic is not None


@pytest.mark.hdf5
def test_periodic_hdf5_roundtrip(tmp_path) -> None:
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3, polarization="TE")
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 320.0))
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=spec,
        solver_method="direct",
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])
    run = sim.run(source, include_farfield=False)
    assert run.periodic is not None

    path = tmp_path / "periodic_payload.h5"
    save_periodic_h5(path, periodic=run.periodic, group="periodic", mode="w")
    loaded = load_periodic_h5(path, group="periodic")

    assert "order_mn" in loaded
    assert "incident_flux" in loaded
    assert "reflected_flux_per_order" in loaded
    assert "transmitted_flux_per_order" in loaded
    assert "power" in loaded
    assert "reflectance" in loaded["power"]
    assert "transmittance" in loaded["power"]
    assert "flux_defect_fraction" in loaded["power"]
    assert "local_absorptance" in loaded["power"]
    assert "closure_error_fraction" in loaded["power"]
    assert int(np.asarray(loaded["order_mn"]).shape[1]) == 2

    orders_only_path = tmp_path / "periodic_orders_only.h5"
    save_periodic_h5(
        orders_only_path,
        periodic=run.periodic,
        group="periodic",
        mode="w",
        include_power=False,
    )
    orders_only = load_periodic_h5(orders_only_path, group="periodic")
    assert "power" not in orders_only
