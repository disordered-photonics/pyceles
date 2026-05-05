from __future__ import annotations

import importlib
from typing import Literal

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.indexing import n_modes
from pyceles.core.operators import PeriodicCouplingOperator, apply_W_numpy, prepare_matvec
from pyceles.core.periodic.ewald import periodic_ewald_block
from pyceles.core.translation import translation_ab5_table
from pyceles.io import load_periodic_h5, save_periodic_h5
from pyceles.simulation import Simulation, SimulationConfig
from pyceles.simulation.solve import periodic_shared_k_parallel


def _plane_wave(
    *,
    polar_angle: float = 0.0,
    azimuthal_angle: float = 0.0,
    polarization: Literal["TE", "TM"] = "TE",
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
        source=_plane_wave(),
        periodic=pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0)),
        verbose=False,
    )

    assert cfg.periodic is not None
    assert cfg.periodic.lattice.area == 120_000.0


def test_periodic_options_reject_invalid_numerical_policy() -> None:
    with pytest.raises(ValueError, match="eta"):
        pcl.PeriodicOptions(eta=-0.1)
    with pytest.raises(ValueError, match="real_shells"):
        pcl.PeriodicOptions(real_shells=-1)
    with pytest.raises(ValueError, match="reciprocal_shells"):
        pcl.PeriodicOptions(reciprocal_shells=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="directsum_window"):
        pcl.PeriodicOptions(directsum_window=-1)
    with pytest.raises(ValueError, match="output_bmax"):
        pcl.PeriodicOptions(output_bmax=0.0)


def test_periodic_config_rejects_unimplemented_backend_combinations() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0))
    with pytest.raises(NotImplementedError, match="CPU/NumPy-only"):
        SimulationConfig(
            source=_plane_wave(), periodic=spec, operator_backend="cupy", verbose=False
        )
    with pytest.raises(NotImplementedError, match="Periodic MLFMM"):
        SimulationConfig(
            source=_plane_wave(), periodic=spec, coupling_backend="mlfmm", verbose=False
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
    with pytest.raises(NotImplementedError, match="PlaneWave excitation"):
        SimulationConfig(
            source=source,
            periodic=pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0)),
            verbose=False,
        )


def test_periodic_overlap_validator_checks_self_images() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(100.0, 100.0))
    cfg = SimulationConfig(source=_plane_wave(), periodic=spec, verbose=False)

    with pytest.raises(ValueError, match="lattice shift"):
        Simulation(cfg, particles=[_sphere(radius=60.0)])


def test_periodic_overlap_validator_accepts_separated_reference_cell() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 300.0))
    cfg = SimulationConfig(source=_plane_wave(), periodic=spec, verbose=False)

    sim = Simulation(cfg, particles=[_sphere(radius=10.0), _sphere(radius=10.0, x=80.0)])

    assert sim.n_particles == 2


def test_periodic_overlap_validator_uses_minimum_image_for_unwrapped_positions() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(100.0, 200.0))
    cfg = SimulationConfig(source=_plane_wave(), periodic=spec, verbose=False)
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
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )
    assert isinstance(prepared.coupling, PeriodicCouplingOperator)

    prepared.populate_coupling(show_progress=False)

    assert set(prepared.coupling._ewald_block_cache) == {(0, 0), (0, 1), (1, 0), (1, 1)}


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
        source=source,
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
        source=source,
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])

    solved = sim.solve_sources({"pw": source})

    assert np.all(np.isfinite(solved.coeffs["pw"]))


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
        source=source,
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
    assert np.isfinite(periodic.reflectance)
    assert np.isfinite(periodic.transmittance)
    assert np.isfinite(periodic.absorptance)
    assert run.farfield.scattered_te["coeff"].shape == (0, 0)


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
        source=source,
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])

    run = sim.run(include_farfield=False)
    assert run.periodic is not None
    periodic = run.periodic
    assert periodic.output_bmax == pytest.approx(0.05)
    assert periodic.order_mn.shape[0] > int(np.count_nonzero(periodic.order_propagating))


def test_periodic_run_rejects_polarization_basis_mode() -> None:
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3, polarization="TE")
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 320.0))
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=spec,
        solver_method="direct",
        source=source,
        solve_polarization_basis=True,
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])
    with pytest.raises(NotImplementedError, match="solve_polarization_basis"):
        sim.run(include_farfield=False)


def test_periodic_hdf5_roundtrip(tmp_path) -> None:
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3, polarization="TE")
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 320.0))
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=1,
        periodic=spec,
        solver_method="direct",
        source=source,
        verbose=False,
    )
    sim = Simulation(cfg, particles=[_sphere(radius=10.0)])
    run = sim.run(include_farfield=False)
    assert run.periodic is not None

    path = tmp_path / "periodic_payload.h5"
    save_periodic_h5(path, periodic=run.periodic, group="periodic", mode="w")
    loaded = load_periodic_h5(path, group="periodic")

    assert "order_mn" in loaded
    assert "reflectance" in loaded
    assert int(np.asarray(loaded["order_mn"]).shape[1]) == 2
