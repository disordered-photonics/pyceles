from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

import pyceles as pcl
from pyceles.core.operators import PeriodicCouplingOperator, prepare_matvec
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


def test_prepare_matvec_periodic_returns_operator_stub() -> None:
    spec = pcl.PeriodicSpec(lattice=pcl.RectangularLattice2D(300.0, 400.0))
    source = _plane_wave(polar_angle=0.2, azimuthal_angle=0.3)
    particles = [_sphere(radius=10.0)]

    prepared = prepare_matvec(
        lmax=1,
        k=2.0 * np.pi / 550.0,
        particles=particles,
        n_medium=1.0 + 0j,
        radial_lut_dr=1.0,
        periodic=spec,
        k_parallel=pcl.core.plane_wave_k_parallel(source),
        show_progress=False,
    )

    assert isinstance(prepared.coupling, PeriodicCouplingOperator)
    np.testing.assert_allclose(
        prepared.coupling.k_parallel,
        pcl.core.plane_wave_k_parallel(source),
    )
    with pytest.raises(NotImplementedError, match="Periodic coupling application"):
        prepared.apply_W(np.zeros((6,), dtype=np.complex128))
