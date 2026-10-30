"""Periodic source boundaries, affine emission, and local Green reconstruction."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import LayeredSphere, Sphere, Spheroid
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.periodic.reception import apply_periodic_reception
from pyceles.core.periodic.types import periodic_source_k_parallel
from pyceles.core.sources import DipoleCollection, DipoleSource, PlaneWave
from pyceles.core.svwf import expansion_fields
from pyceles.core.translation import translation_block_rect
from pyceles.postprocessing.dipole_metrics import (
    compute_dipole_ldos_enhancement,
    compute_dipole_reaction_field,
    compute_dipole_source_work,
)
from pyceles.postprocessing.nearfield import compute_periodic_near_field
from pyceles.simulation import ChannelResult, ResultRetention, Simulation, SimulationConfig


def _config(method="directsum", *, host=1.33 + 0.25j, q=(0.0013, -0.0007), lmax=2):
    return SimulationConfig(
        wavelength=550,
        n_medium=host,
        lmax=lmax,
        verbose=False,
        solver_method="direct",
        polar_angles=np.linspace(0, np.pi, 3),
        azimuthal_angles=np.linspace(0, 2 * np.pi, 4, endpoint=False),
        periodic=PeriodicSpec(
            RectangularLattice2D(600, 650),
            PeriodicOptions(
                method=method,
                directsum_window=3,
                real_shells=3,
                reciprocal_shells=4,
            ),
            k_parallel=q,
        ),
    )


@pytest.mark.api_contract
def test_bloch_boundary_resolves_sources_and_rejects_conflicting_waves():
    periodic = PeriodicSpec(RectangularLattice2D(600, 650))
    np.testing.assert_array_equal(periodic_source_k_parallel(periodic, DipoleSource(550)), [0, 0])
    np.testing.assert_array_equal(periodic_source_k_parallel(periodic, PlaneWave(550)), [0, 0])
    with pytest.raises(ValueError, match="disagrees"):
        periodic_source_k_parallel(replace(periodic, k_parallel=(0.01, 0)), PlaneWave(550))
    for q in ((1j, 0), (np.inf, 0), (0, 0, 0)):
        with pytest.raises(ValueError, match="k_parallel must"):
            replace(periodic, k_parallel=cast(Any, q))


@pytest.mark.parametrize("orders", [(1, 3), (3, 1), (2, 3)])
def test_rectangular_reception_matches_independent_image_blocks(orders):
    out, inc = orders
    cfg = _config()
    pos = np.array([[10, 20, 30]])
    target = np.array([[230, 130, -10], [270, 230, 50]])
    rng = np.random.default_rng(33)
    coeffs = rng.normal(size=(1, 2 * inc * (inc + 2))) + 1j * rng.normal(
        size=(1, 2 * inc * (inc + 2))
    )
    k = 2 * np.pi / cfg.wavelength * cfg.n_medium
    actual = apply_periodic_reception(
        target,
        pos,
        coeffs,
        lmax_out=out,
        lmax_in=inc,
        k=k,
        periodic=cfg.periodic,
        k_parallel=np.asarray(cfg.periodic.k_parallel),
    )
    expected = np.zeros_like(actual)
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            shift = cfg.periodic.lattice.lattice_vector(ix, iy)
            phase = np.exp(1j * (shift[:2] @ cfg.periodic.k_parallel))
            for j, point in enumerate(target):
                expected[j] += phase * (
                    translation_block_rect(out, inc, k, point - pos[0] - shift) @ coeffs[0]
                )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("method", ["directsum", "ewald", "rayleigh"])
def test_host_and_embedded_sources_solve_and_reconstruct_bloch_fields(method):
    cfg = _config(method)
    particles = (LayeredSphere((590, 20, 0), (20, 35), (1.5, 1.8 + 0.1j)),)
    source = DipoleCollection(
        550,
        medium_n=cfg.n_medium,
        positions=np.array([[615, 20, 0], [230, 220, 100]]),
        dipole_moments=np.array([[1, 0.2j, -0.3], [0.1j, 0.6, 0.2]]),
    )
    run = Simulation(cfg, particles=particles).run(source)
    assert run.power is None and run.periodic is None
    points = np.array([[620, 23, 3], [200, 150, 0], [240, 200, 150], [100, 50, -100]])
    nf = compute_periodic_near_field(run, points=points)
    assert np.all(np.isfinite(nf.E_total)) and np.all(np.isfinite(nf.H_total))
    shift = np.array([600, -650, 0])
    phase = np.exp(1j * (shift[:2] @ cfg.periodic.k_parallel))
    moved = compute_periodic_near_field(run, points=points + shift)
    for a, b in ((moved.E_total, nf.E_total), (moved.H_total, nf.H_total)):
        np.testing.assert_allclose(a, phase * b, rtol=2e-12, atol=1e-23)
    np.testing.assert_array_equal(nf.inside_mask, [True, False, False, False])
    singular = compute_periodic_near_field(run, points=source.positions)
    assert np.all(np.isnan(singular.E_total))
    assert np.all(np.isfinite(compute_dipole_reaction_field(run)))
    with pytest.raises(ValueError, match="nonabsorbing local"):
        compute_dipole_source_work(run)


def test_direct_image_fields_include_host_sources_above_the_particle_slab():
    cfg = _config()
    source = DipoleSource(
        550, medium_n=cfg.n_medium, position=(200, 210, 170), dipole_moment=(0.1j, 0.6, 0.2)
    )
    run = Simulation(cfg, particles=()).run(source)
    points = np.array([[220, 230, 190], [220, 230, 140], [100, 20, -40]])
    nf = compute_periodic_near_field(run, points=points)
    e, h = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            shift = cfg.periodic.lattice.lattice_vector(ix, iy)
            phase = np.exp(1j * (shift[:2] @ cfg.periodic.k_parallel))
            ei, hi = expansion_fields(
                points,
                centre=np.array(source.position) + shift,
                regular=np.zeros(6),
                outgoing=source.outgoing_coeffs()[0] * phase,
                k=run.k,
                n=cfg.n_medium,
                lmax=1,
            )
            e += ei
            h += hi
    np.testing.assert_allclose(nf.E_total, e, rtol=3e-13, atol=1e-23)
    np.testing.assert_allclose(nf.H_total, h, rtol=3e-13, atol=1e-23)


def test_periodic_embedded_work_is_invariant_to_emitter_representative():
    cfg = _config("ewald")
    particle = Sphere((590, 20, 0), 35, 1.5)
    source = DipoleSource(
        550, medium_n=cfg.n_medium, position=(610, 20, 0), dipole_moment=(1, 0.2j, -0.3)
    )
    shift = np.array([600, -650, 0])
    phase = np.exp(1j * (shift[:2] @ cfg.periodic.k_parallel))
    moved = replace(
        source,
        position=tuple(np.array(source.position) + shift),
        dipole_moment=tuple(np.array(source.dipole_moment) * phase),
    )
    a, b = (Simulation(cfg, particles=(particle,)).run(s) for s in (source, moved))
    wa, wb = (compute_dipole_source_work(r, reaction_lmax=8) for r in (a, b))
    np.testing.assert_allclose(a.coeffs, b.coeffs, rtol=1e-13, atol=1e-23)
    np.testing.assert_allclose(wa.power_total, wb.power_total, rtol=2e-13)
    np.testing.assert_allclose(
        wb.E_reaction_at_dipoles, phase * wa.E_reaction_at_dipoles, rtol=3e-13, atol=1e-23
    )
    with pytest.raises(ValueError, match="collective"):
        compute_dipole_ldos_enhancement(a)


@pytest.mark.parametrize("solver", ["gmres", "bicgstab", "lsqr"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_periodic_source_blocks_preserve_coherent_emission(solver, dtype):
    cfg = replace(
        _config("ewald"),
        solver_method=solver,
        compute_dtype=dtype,
        accum_dtype="complex128",
        solver_rtol=1e-5 if dtype == "complex64" else 1e-10,
    )
    embedded = DipoleSource(
        550, medium_n=cfg.n_medium, position=(10, 0, 0), dipole_moment=(1, 0.2j, -0.3)
    )
    host = replace(embedded, position=(230, 200, 50), dipole_moment=(0.1j, 0.6, 0.2))
    combined = DipoleCollection(
        550,
        medium_n=cfg.n_medium,
        positions=np.array([embedded.position, host.position]),
        dipole_moments=np.array([embedded.dipole_moment, host.dipole_moment]),
    )
    simulation = Simulation(cfg, particles=(Sphere((0, 0, 0), 35, 1.5),))
    channels: tuple[ChannelResult, ...]
    if solver == "lsqr":
        channels = tuple(
            simulation.run(s, retention=ResultRetention.minimal())
            for s in (embedded, host, combined)
        )
    else:
        result = simulation.run_sources(
            {"embedded": embedded, "host": host, "combined": combined},
            retention=ResultRetention.minimal(),
        )
        channels = tuple(result.channels[label] for label in ("embedded", "host", "combined"))
    a, b, c = channels
    rtol = 5e-5 if dtype == "complex64" else 2e-9
    np.testing.assert_allclose(c.coeffs, a.coeffs + b.coeffs, rtol=rtol, atol=1e-22)
    points = np.array([[5, 3, 2], [120, 200, 80]])
    fields = [compute_periodic_near_field(run, points=points) for run in (a, b, c)]
    np.testing.assert_allclose(
        fields[2].E_total, fields[0].E_total + fields[1].E_total, rtol=rtol, atol=1e-22
    )
    np.testing.assert_allclose(
        fields[2].H_total, fields[0].H_total + fields[1].H_total, rtol=rtol, atol=1e-22
    )


@pytest.mark.parametrize("method", ["directsum", "ewald"])
def test_transparent_periodic_particle_does_not_modify_emitter_field_or_work(method):
    cfg = _config(method, host=1.33)
    source = DipoleSource(550, medium_n=cfg.n_medium, position=(590, 20, 0))
    a, b = (
        Simulation(cfg, particles=particles).run(source)
        for particles in ((), (Sphere(source.position, 35, cfg.n_medium),))
    )
    points = np.array([[590, 21, 2], [620, 50, 100]])
    np.testing.assert_allclose(
        compute_periodic_near_field(a, points=points).E_total,
        compute_periodic_near_field(b, points=points).E_total,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        compute_dipole_source_work(a).power_total,
        compute_dipole_source_work(b).power_total,
        rtol=0,
        atol=0,
    )


def test_neighbouring_spheroid_image_retains_the_circumsphere_guard():
    cfg = _config("ewald")
    source = DipoleSource(550, medium_n=cfg.n_medium, position=(15, 20, 0))
    particle = Spheroid(
        position=(590, 20, 0), equatorial_radius=20, polar_radius=100, refractive_index=1.5
    )
    with pytest.raises(ValueError, match="circumscribing sphere"):
        Simulation(cfg, particles=(particle,)).run(source)


@pytest.mark.parametrize("inside", [False, True])
def test_noninteger_source_point_is_singular_in_its_periodic_images(inside):
    cfg = _config("ewald")
    source = DipoleSource(550, medium_n=cfg.n_medium, position=(10.1, 20.3, 0.7))
    particles = (Sphere((10, 20, 0), 35, 1.5),) if inside else ()
    run = Simulation(cfg, particles=particles).run(source)
    shift = cfg.periodic.lattice.lattice_vector(1, -1)
    fields = compute_periodic_near_field(run, points=np.array(source.position) + shift)
    assert np.all(np.isnan(fields.E_total))
    assert np.all(np.isnan(fields.H_total))


def test_work_rejects_coincident_fractional_sources_in_different_cells():
    cfg = _config("ewald", host=1.33)
    point = np.array([10.1, 20.3, 0.7])
    source = DipoleCollection(
        550,
        medium_n=cfg.n_medium,
        positions=np.stack((point, point + cfg.periodic.lattice.lattice_vector(1, -1))),
        dipole_moments=np.array([[1, 0, 0], [0, 1, 0]]),
    )
    run = Simulation(cfg, particles=()).run(source)
    with pytest.raises(ValueError, match="distinct active"):
        compute_dipole_source_work(run)


@pytest.mark.parametrize("method", ["directsum", "ewald", "rayleigh"])
def test_reception_requires_explicit_reference_self_subtraction(method):
    cfg = _config(method)
    positions = np.asarray([[10.0, 20.0, 30.0]])
    # An exactly representable image must obey the same source-domain rule.
    points = positions + np.asarray([600.0, -650.0, 0.0])
    kwargs = dict(
        lmax_in=1,
        k=2 * np.pi * cfg.n_medium / cfg.wavelength,
        periodic=cfg.periodic,
        k_parallel=np.asarray(cfg.periodic.k_parallel),
    )
    with pytest.raises(ValueError, match="singular at its source position"):
        apply_periodic_reception(points, positions, np.ones((1, 6)), **kwargs)
    np.testing.assert_array_equal(
        apply_periodic_reception(points, positions, np.zeros((1, 6)), **kwargs),
        np.zeros((1, 6)),
    )


def test_reception_rejects_unknown_backend_even_for_empty_input():
    cfg = _config()
    with pytest.raises(ValueError, match="backend must"):
        apply_periodic_reception(
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 6)),
            lmax_in=1,
            k=1,
            periodic=cfg.periodic,
            k_parallel=np.zeros(2),
            backend="cuppy",
        )


@pytest.mark.parametrize("method", ["ewald", "rayleigh"])
def test_inactive_sources_do_not_set_reception_geometry_policy(method):
    cfg = _config(method)
    positions = np.array([[10, 20, 30]])
    points = np.array([[230, 130, -10], [270, 230, 900]])
    coefficients = np.ones((1, 6), dtype=complex)
    kwargs = dict(
        lmax_in=1,
        k=2 * np.pi * cfg.n_medium / cfg.wavelength,
        periodic=cfg.periodic,
        k_parallel=np.asarray(cfg.periodic.k_parallel),
    )
    reference = apply_periodic_reception(
        points, positions, coefficients, circumscribing_radii=np.array([30]), **kwargs
    )
    mixed = apply_periodic_reception(
        points,
        np.r_[positions, [[10, 20, 1e6]]],
        np.r_[coefficients, np.zeros((1, 6))],
        circumscribing_radii=np.array([30, 1e6]),
        **kwargs,
    )
    np.testing.assert_array_equal(mixed, reference)
