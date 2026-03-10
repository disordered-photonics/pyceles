from typing import Any, cast

import numpy as np

from pyceles.core.fields import PlaneWave
from pyceles.core.particles import Sphere
from pyceles.postprocessing.nearfield import compute_near_field_slice
from pyceles.simulation import Simulation, SimulationConfig


def _make_run():
    cfg = SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        source=PlaneWave(
            wavelength=550.0,
            medium_n=1.0 + 0j,
            polarization="TE",
            polar_angle=0.0,
            azimuthal_angle=0.0,
            amplitude=1.0,
            focal_point=(0.0, 0.0, 0.0),
        ),
        polar_angles=np.linspace(0.0, np.pi, 31),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 25, endpoint=False),
        verbose=False,
    )
    sim = Simulation(
        cfg,
        particles=[Sphere(position=(0.0, 0.0, 0.0), radius=120.0, refractive_index=1.5 + 0.0j)],
    )
    return sim.run(include_farfield=False)


def test_compute_near_field_slice_uses_axis_bounds():
    run = _make_run()
    out = compute_near_field_slice(
        run,
        plane="y",
        plane_value=0.0,
        axis_0_min=-200.0,
        axis_0_max=200.0,
        axis_1_min=-100.0,
        axis_1_max=100.0,
        dx=100.0,
        show_progress=False,
        center_pixel_policy="none",
    )
    assert out.axis_0.shape == out.axis_1.shape
    assert out.inside.shape == out.axis_0.shape


def test_compute_near_field_slice_rejects_legacy_xyz_bounds_kwargs():
    run = _make_run()
    fn = cast(Any, compute_near_field_slice)
    with np.testing.assert_raises(TypeError):
        fn(
            run,
            x_min=-200.0,
            x_max=200.0,
            z_min=-100.0,
            z_max=100.0,
            dx=100.0,
            plane="y",
            plane_value=0.0,
            show_progress=False,
        )
