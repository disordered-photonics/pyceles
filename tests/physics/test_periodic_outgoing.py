"""Finite-reference outgoing fields and flux in an absorbing periodic host."""

from dataclasses import replace

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.sources import DipoleSource
from pyceles.postprocessing.farfield import (
    compute_periodic_outgoing_amplitudes,
    periodic_outgoing_flux,
)
from pyceles.postprocessing.nearfield import compute_periodic_near_field
from pyceles.simulation import Simulation, SimulationConfig


def _analytic_image_fields(points, source, lattice, q, window):
    """Elementary electric-dipole Green tensor, independent of SVWF/Ewald code."""
    ix, iy = np.meshgrid(np.arange(-window, window + 1), np.arange(-window, window + 1))
    shifts = np.column_stack((ix.ravel() * lattice.ax, iy.ravel() * lattice.ay, np.zeros(ix.size)))
    k0 = 2 * np.pi / source.wavelength
    k = k0 * source.medium_n
    moment = source.dipole_moments()[0]
    displacement = points[:, None, :] - np.asarray(source.position) - shifts
    radius = np.linalg.norm(displacement, axis=-1)
    direction = displacement / radius[..., None]
    longitudinal = direction * (direction @ moment)[..., None]
    green = np.exp(1j * (k * radius + shifts[:, :2] @ q)) / (4 * np.pi * radius)
    e = (
        k0**2
        * green[..., None]
        * (
            moment
            - longitudinal
            + (1 / (k * radius) ** 2 - 1j / (k * radius))[..., None] * (3 * longitudinal - moment)
        )
    )
    h = (
        source.medium_n
        * k0**2
        * green[..., None]
        * (1 + 1j / (k * radius))[..., None]
        * np.cross(direction, moment)
    )
    return np.sum(e, axis=1), np.sum(h, axis=1)


def test_absorbing_array_flux_matches_cell_poynting_integral_at_finite_planes():
    cfg = SimulationConfig(
        wavelength=550,
        n_medium=1.33 + 0.08j,
        lmax=2,
        verbose=False,
        periodic=PeriodicSpec(
            RectangularLattice2D(600, 650),
            PeriodicOptions(method="ewald", eta=0.003, real_shells=4, reciprocal_shells=6),
            k_parallel=(0.0013, -0.0007),
        ),
    )
    assert cfg.periodic is not None
    source = DipoleSource(
        550, medium_n=cfg.n_medium, position=(210, 170, 1e6), dipole_moment=(1, 0.3j, -0.2)
    )
    run = Simulation(cfg, particles=()).run(source)
    outgoing = compute_periodic_outgoing_amplitudes(run, output_bmax=0.1)
    assert outgoing.reference_z == (1e6, 1e6)
    assert not np.any(outgoing.orders.propagating)
    assert np.all(np.isfinite(outgoing.up_amplitudes))
    z_above, z_below = 1e6 + 350, 1e6 - 350
    flux = periodic_outgoing_flux(
        outgoing, k=run.k, n_medium=cfg.n_medium, z_above=z_above, z_below=z_below
    )
    x, y = np.meshgrid(np.arange(24) * 600 / 24, np.arange(24) * 650 / 24, indexing="ij")
    for sign, z, branch in ((1, z_above, flux[0]), (-1, z_below, flux[1])):
        points = np.stack((x, y, np.full_like(x, z)), axis=-1)
        fields = compute_periodic_near_field(run, points=points)
        order_fields = compute_periodic_near_field(run, points=points, field_bmax=0.1)
        np.testing.assert_allclose(order_fields.E_total, fields.E_total, rtol=2e-10, atol=1e-21)
        np.testing.assert_allclose(order_fields.H_total, fields.H_total, rtol=2e-10, atol=1e-21)
        integral = (
            sign
            * cfg.periodic.lattice.area
            / 2
            * np.mean(np.cross(fields.E_total, fields.H_total.conj())[..., 2].real)
        )
        np.testing.assert_allclose(np.sum(branch), integral, rtol=2e-10, atol=0)
    probes = np.array([[450, 325, z_above], [425, 350, z_above], [300, 100, z_above]])
    previous = _analytic_image_fields(
        probes, source, cfg.periodic.lattice, cfg.periodic.k_parallel, 48
    )
    images = _analytic_image_fields(
        probes, source, cfg.periodic.lattice, cfg.periodic.k_parallel, 64
    )
    fields = compute_periodic_near_field(run, points=probes, field_bmax=0.1)
    for actual, reference, earlier in zip(
        (fields.E_total, fields.H_total), images, previous, strict=True
    ):
        assert np.linalg.norm(reference - earlier) / np.linalg.norm(reference) < 1e-10
        np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=1e-21)
    further = periodic_outgoing_flux(
        outgoing, k=run.k, n_medium=cfg.n_medium, z_above=z_above + 200, z_below=z_below - 200
    )
    assert 0 < sum(np.sum(f) for f in further) < sum(np.sum(f) for f in flux)
    centered = Simulation(cfg, particles=()).run(replace(source, position=(210, 170, 0)))
    centered_orders = compute_periodic_outgoing_amplitudes(centered, output_bmax=0.1)
    np.testing.assert_allclose(
        outgoing.up_amplitudes, centered_orders.up_amplitudes, rtol=1e-14, atol=0
    )
    np.testing.assert_allclose(
        outgoing.down_amplitudes, centered_orders.down_amplitudes, rtol=1e-14, atol=0
    )
    with pytest.raises(ValueError, match="finite"):
        periodic_outgoing_flux(outgoing, k=run.k, n_medium=cfg.n_medium, z_above=np.inf)
