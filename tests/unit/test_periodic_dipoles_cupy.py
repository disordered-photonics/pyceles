"""Shared periodic CuPy reception and native dipole field/work contracts."""

from dataclasses import replace

import numpy as np
import pytest

from pyceles.core.indexing import n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import LayeredSphere, Sphere, TMatrixParticle
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.periodic.reception import apply_periodic_reception
from pyceles.core.periodic.reception_cupy import apply_periodic_reception_cupy
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.postprocessing.dipole_metrics import (
    compute_dipole_reaction_field,
    compute_dipole_source_work,
)
from pyceles.postprocessing.nearfield import compute_periodic_near_field
from pyceles.simulation import ResultRetention, Simulation, SimulationConfig

pytestmark = pytest.mark.gpu


def _config(method, dtype, host):
    return SimulationConfig(
        wavelength=550,
        n_medium=host,
        lmax=2,
        verbose=False,
        solver_method="gmres",
        solver_rtol=2e-6 if dtype == "complex64" else 1e-10,
        compute_dtype=dtype,
        accum_dtype="complex128",
        operator_backend="cupy",
        periodic=PeriodicSpec(
            RectangularLattice2D(600, 650),
            PeriodicOptions(
                method=method,
                eta=0.002,
                real_shells=4,
                reciprocal_shells=6,
                rayleigh_z_cut=200,
                rayleigh_reciprocal_shells=12,
            ),
            k_parallel=(0.0013, -0.0007),
        ),
    )


@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("host", [1.33, 1.33 + 0.08j])
def test_periodic_dipoles_gpu_matches_cpu_for_solve_fields_and_reaction(
    cupy_runtime, method, dtype, host
):
    cfg = _config(method, dtype, host)
    matrix = np.eye(n_modes(2), dtype=complex) * (0.001 + 0.002j)
    matrix[0, 7], matrix[5, 2] = 0.003 - 0.001j, -0.002j
    particles = (
        LayeredSphere((590, 20, 0), (20, 35), (1.5, 1.8 + 0.1j)),
        Sphere((160, 140, 720), 30, 1.65 + 0.03j),
        TMatrixParticle((320, 300, -700), 25, 2, matrix),
    )
    source = DipoleCollection(
        550,
        medium_n=host,
        positions=np.array([[600, 20, 0], [230, 210, 220]]),
        dipole_moments=np.array([[1, 0.2j, -0.3], [0.1j, 0.6, 0.2]]),
    )
    cpu = Simulation(replace(cfg, operator_backend="numpy"), particles=particles).run(source)
    gpu = Simulation(cfg, particles=particles).run(source, retention=ResultRetention.minimal())
    tol = 5e-5 if dtype == "complex64" else 2e-8
    np.testing.assert_allclose(gpu.coeffs, cpu.coeffs, rtol=tol, atol=1e-19)
    points = np.array(
        [
            [605, 23, 3],
            [615, 22, 2],
            [162, 145, 725],
            [240, 200, 180],
            [240, 200, 270],
            [100, 50, -1800],
            [210, 120, 1800],
        ]
    )
    fields_cpu, fields_gpu = (compute_periodic_near_field(run, points=points) for run in (cpu, gpu))
    for name in (
        "E_initial",
        "H_initial",
        "E_scattered",
        "H_scattered",
        "E_internal",
        "H_internal",
        "E_total",
        "H_total",
    ):
        actual, expected = getattr(fields_gpu, name), getattr(fields_cpu, name)
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=1e-18)
    np.testing.assert_array_equal(fields_cpu.inside_mask, fields_gpu.inside_mask)
    np.testing.assert_allclose(
        compute_dipole_reaction_field(gpu), compute_dipole_reaction_field(cpu), rtol=tol, atol=1e-18
    )
    if complex(host).imag == 0:
        np.testing.assert_allclose(
            compute_dipole_source_work(gpu).power_total,
            compute_dipole_source_work(cpu).power_total,
            rtol=tol,
            atol=0,
        )
    else:
        with pytest.raises(ValueError, match="nonabsorbing local"):
            compute_dipole_source_work(gpu)


@pytest.mark.parametrize("orders", [(1, 3), (3, 1), (3, 3)])
@pytest.mark.parametrize(
    "dtype,accum",
    [(np.complex64, np.complex64), (np.complex64, np.complex128), (np.complex128, np.complex128)],
)
def test_rectangular_gpu_reception_keeps_self_images_and_owned_outputs(
    cupy_runtime, orders, dtype, accum
):
    out, inc = orders
    boundary = PeriodicSpec(
        RectangularLattice2D(600, 650),
        PeriodicOptions(method="ewald", real_shells=4, reciprocal_shells=5),
        k_parallel=(0.0013, -0.0007),
    )
    rng = np.random.default_rng(33)
    positions = np.array([[10, 20, 30], [590, 25, 100], [230, 250, -200]])
    points = np.r_[positions, [[240, 130, 40], [620, 23, -50], [180, 140, 210]]]
    coefficients = (
        rng.normal(size=(3, n_modes(inc))) + 1j * rng.normal(size=(3, n_modes(inc)))
    ).astype(dtype)
    original = coefficients.copy()
    kwargs = dict(
        lmax_out=out,
        k=2 * np.pi / 550 * (1.33 + 0.1j),
        periodic=boundary,
        k_parallel=np.asarray(boundary.k_parallel),
        exclude_self=True,
        compute_dtype=dtype,
        accum_dtype=accum,
    )
    expected = apply_periodic_reception(points, positions, coefficients, lmax_in=inc, **kwargs)
    result = apply_periodic_reception_cupy(
        points=points,
        positions=positions,
        coefficients=coefficients,
        lmax_in=inc,
        point_batch_size=2,
        source_batch_size=2,
        **kwargs,
    )
    saved = result.copy()
    next_result = apply_periodic_reception_cupy(
        points=points[::-1],
        positions=np.r_[positions, [[30, 25, 1e6]]],
        coefficients=np.r_[-coefficients, np.zeros((1, n_modes(inc)), dtype=dtype)],
        lmax_in=inc,
        point_batch_size=1,
        source_batch_size=1,
        **kwargs,
    )
    np.testing.assert_array_equal(result, saved)
    np.testing.assert_array_equal(coefficients, original)
    assert result.dtype == np.dtype(accum)
    tolerance = 5e-6 if accum == np.complex64 else 2e-10
    assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < tolerance
    assert np.linalg.norm(next_result + expected[::-1]) / np.linalg.norm(expected) < tolerance


@pytest.mark.parametrize("solver", ["direct", "bicgstab", "lsqr"])
def test_periodic_embedded_gpu_sources_use_existing_solver_interfaces(cupy_runtime, solver):
    cfg = replace(_config("ewald", "complex128", 1.33 + 0.08j), solver_method=solver)
    particles = (LayeredSphere((590, 20, 0), (20, 35), (1.5, 1.8 + 0.1j)),)
    source = DipoleSource(
        550, medium_n=cfg.n_medium, position=(600.1, 20.3, 0.7), dipole_moment=(1, 0.2j, -0.3)
    )
    cpu = Simulation(
        replace(cfg, operator_backend="numpy", solver_method="direct"), particles=particles
    ).run(source)
    gpu = Simulation(cfg, particles=particles).run(source)
    np.testing.assert_allclose(gpu.coeffs, cpu.coeffs, rtol=2e-9, atol=1e-21)
    points = np.stack(
        (source.source_positions()[0], source.source_positions()[0] + np.array([600, -650, 0]))
    )
    fields = compute_periodic_near_field(gpu, points=points)
    assert np.all(np.isnan(fields.E_total)) and np.all(np.isnan(fields.H_total))


def test_periodic_gpu_source_blocks_share_one_bloch_boundary(cupy_runtime):
    cfg = _config("rayleigh", "complex64", 1.33 + 0.08j)
    particles = (LayeredSphere((590, 20, 0), (20, 35), (1.5, 1.8 + 0.1j)),)
    source = DipoleSource(550, medium_n=cfg.n_medium, position=(615.1, 20.3, 0.7))
    sources = source.cartesian_basis_sources()
    cpu = Simulation(replace(cfg, operator_backend="numpy"), particles=particles).run_sources(
        sources
    )
    gpu = Simulation(cfg, particles=particles).run_sources(
        sources, retention=ResultRetention.minimal()
    )
    for label in sources:
        actual, expected = gpu.channels[label].coeffs, cpu.channels[label].coeffs
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 5e-5
    active = next(iter(gpu.channels.values()))
    assert np.all(np.isfinite(compute_dipole_reaction_field(active)))
    with pytest.raises(ValueError, match="nonabsorbing local"):
        compute_dipole_source_work(active)


@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
def test_periodic_gpu_reception_rejects_active_source_origins(cupy_runtime, method):
    cfg = _config(method, "complex128", 1.33 + 0.08j)
    positions = np.asarray([[10.0, 20.0, 30.0]])
    points = positions + np.asarray([600.0, -650.0, 0.0])
    kwargs = dict(
        lmax_in=1,
        k=2 * np.pi * cfg.n_medium / cfg.wavelength,
        periodic=cfg.periodic,
        k_parallel=np.asarray(cfg.periodic.k_parallel),
    )
    # Test the direct GPU entry point as well as high-level backend dispatch.
    with pytest.raises(ValueError, match="singular at its source position"):
        apply_periodic_reception_cupy(points, positions, np.ones((1, 6)), **kwargs)
    np.testing.assert_array_equal(
        apply_periodic_reception_cupy(points, positions, np.zeros((1, 6)), **kwargs),
        np.zeros((1, 6)),
    )
