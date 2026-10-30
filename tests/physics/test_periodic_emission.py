"""Radiation and reciprocal work of a Bloch-periodic electric emitter array."""

from dataclasses import replace

import numpy as np
import pytest

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.particles import LayeredSphere
from pyceles.core.periodic import PeriodicOptions, PeriodicSpec
from pyceles.core.sources import DipoleCollection, DipoleSource
from pyceles.postprocessing.dipole_metrics import (
    compute_dipole_reaction_field,
    compute_dipole_source_work,
)
from pyceles.postprocessing.farfield.periodic import (
    compute_periodic_outgoing_amplitudes,
    periodic_outgoing_flux,
)
from pyceles.postprocessing.nearfield import compute_periodic_near_field
from pyceles.simulation import Simulation, SimulationConfig


def _config(q):
    return SimulationConfig(
        wavelength=550,
        n_medium=1.33,
        lmax=4,
        verbose=False,
        solver_method="direct",
        polar_angles=np.linspace(0, np.pi, 3),
        azimuthal_angles=np.linspace(0, 2 * np.pi, 4, endpoint=False),
        periodic=PeriodicSpec(
            RectangularLattice2D(600, 650),
            PeriodicOptions(
                method="ewald",
                real_shells=4,
                reciprocal_shells=5,
            ),
            k_parallel=q,
        ),
    )


def test_bare_array_work_equals_analytic_plane_wave_radiation_per_cell():
    """The sheet Green tensor gives an independent propagating-order flux sum."""
    cfg = _config((0.0013, -0.0007))
    source = DipoleSource(
        550, medium_n=cfg.n_medium, position=(200, 210, 50), dipole_moment=(1, 0.3j, -0.2)
    )
    run = Simulation(cfg, particles=()).run(source)
    total = compute_dipole_source_work(run).power_total.sum()
    area = cfg.periodic.lattice.area
    moment = np.asarray(source.dipole_moment)
    flux = 0.0
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            q = np.asarray(cfg.periodic.k_parallel) + cfg.periodic.lattice.reciprocal_vector(ix, iy)
            kz2 = run.k.real**2 - q @ q
            if kz2 <= 0:
                continue
            for sign in (-1, 1):
                K = np.r_[q, sign * np.sqrt(kz2)]
                e = (
                    1j
                    * run.k0**2
                    / (2 * area * np.sqrt(kz2))
                    * (moment - K * (K @ moment) / run.k**2)
                )
                h = cfg.n_medium * np.cross(K / run.k, e)
                flux += sign * area / 2 * np.cross(e, h.conj())[2].real
    np.testing.assert_allclose(total, flux, rtol=3e-12, atol=0)


def test_lossless_embedded_array_work_equals_outgoing_radiation_per_cell():
    cfg = replace(_config((0.0013, -0.0007)), lmax=6)
    source = DipoleCollection(
        550,
        medium_n=cfg.n_medium,
        positions=np.array([[40, 20, 0], [230, 210, 80]]),
        dipole_moments=np.array([[1, 0.3j, -0.2], [0.2j, 0.6, 0.3]]),
    )
    particles = (LayeredSphere((30, 20, 0), (20, 35), (1.5, 1.8)),)
    run = Simulation(cfg, particles=particles).run(source)
    work = compute_dipole_source_work(run, reaction_lmax=10).power_total.sum()
    outgoing = compute_periodic_outgoing_amplitudes(run)
    flux = sum(
        np.sum(branch)
        for branch in periodic_outgoing_flux(outgoing, k=run.k, n_medium=cfg.n_medium)
    )
    np.testing.assert_allclose(work, flux, rtol=1e-10, atol=0)


def test_embedded_periodic_reaction_obeys_bilinear_reciprocity_at_reversed_q():
    cfg = _config((0.0013, -0.0007))
    cfg = replace(cfg, n_medium=1.33 + 0.1j)
    particles = (LayeredSphere((590, 20, 0), (20, 35), (1.5 + 0.02j, 1.8 + 0.1j)),)
    embedded = DipoleSource(
        550, medium_n=cfg.n_medium, position=(610, 25, 0), dipole_moment=(1, 0.3j, -0.2)
    )
    host = replace(embedded, position=(230, 210, 80), dipole_moment=(0.2j, 0.6, 0.3))
    a = Simulation(cfg, particles=particles).run(embedded)
    reverse = replace(
        cfg, periodic=replace(cfg.periodic, k_parallel=tuple(-np.asarray(cfg.periodic.k_parallel)))
    )
    b = Simulation(reverse, particles=particles).run(host)
    ea = compute_periodic_near_field(a, points=np.asarray(host.position)).E_total
    eb = compute_periodic_near_field(b, points=np.asarray(embedded.position)).E_total
    np.testing.assert_allclose(
        np.dot(host.dipole_moment, ea), np.dot(embedded.dipole_moment, eb), rtol=2e-9, atol=1e-22
    )
    assert np.all(np.isfinite(compute_dipole_reaction_field(a)))
    with pytest.raises(ValueError, match="nonabsorbing local"):
        compute_dipole_source_work(a)
