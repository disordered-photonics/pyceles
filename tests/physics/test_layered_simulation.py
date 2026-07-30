from __future__ import annotations

import numpy as np

from pyceles import compute_near_field
from pyceles.core.particles import LayeredSphere
from pyceles.core.sources import PlaneWave
from pyceles.simulation import Simulation, SimulationConfig


def test_layered_sphere_simulation_runs_and_nearfield_dispatches():
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        polar_angles=np.linspace(0.0, np.pi, 41),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False),
        check_circumscribing_sphere_overlap=True,
        verbose=False,
    )
    sim = Simulation(
        cfg,
        particles=[
            LayeredSphere(
                position=(0.0, 0.0, 0.0),
                layer_radii=(80.0, 120.0),
                layer_refractive_indices=(1.8 + 0j, 1.35 + 0.02j),
            )
        ],
    )
    assert sim.n_particles == 1
    run = sim.run(
        PlaneWave(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            polarization="TE",
            polar_angle=0.0,
            azimuthal_angle=0.0,
            amplitude=1.0,
            focal_point=(0.0, 0.0, 0.0),
        ),
        include_farfield=False,
    )
    assert run.n_particles == 1
    assert run.particles is not None
    assert len(run.particles) == 1
    assert run.coeffs.shape[0] == 1
    assert np.all(np.isfinite(run.coeffs))

    nf = compute_near_field(
        run,
        points=np.array([[10.0, 0.0, 0.0], [200.0, 0.0, 0.0]], dtype=float),
        show_progress=False,
    )
    assert nf.inside_mask.tolist() == [True, False]
    assert np.all(np.isfinite(nf.E_total))
