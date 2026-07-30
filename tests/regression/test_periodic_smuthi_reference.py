"""Regression against a local SMUTHI homogeneous periodic one-sphere reference."""

from __future__ import annotations

import numpy as np
import pytest

import pyceles as pcl

pytestmark = pytest.mark.reference

_SMUTHI_ONE_SPHERE_COEFFS = np.array(
    [
        -0.014915920033791091 + 0.027152509508112207j,
        -4.0822571554230608e-19 + 2.8194275611729213e-19j,
        -0.014915920033791091 + 0.027152509508112207j,
        3.6296364344320786e-22 - 3.5831234584712418e-22j,
        -0.00079509267592868169 - 0.0004533111207563875j,
        -5.988513973263101e-22 - 8.8469389859193623e-22j,
        -0.00079509267592868169 - 0.0004533111207563875j,
        1.4017248761481089e-21 + 1.7010240683846519e-21j,
        0.14195327316620174 - 0.20637402973002295j,
        1.6888937781684991e-19 - 1.547864043600622e-19j,
        -0.14195327316620174 + 0.20637402973002295j,
        1.099385094050113e-19 - 2.0543949892521325e-19j,
        0.013035579841932222 + 0.0070093191670349449j,
        2.1791863069450358e-20 + 5.9549516013335858e-20j,
        -0.013035579841932222 - 0.0070093191670349449j,
        -1.1153187305014339e-19 - 1.816930655089988e-19j,
    ],
    dtype=np.complex128,
)


def test_one_sphere_periodic_coefficients_match_smuthi_reference() -> None:
    src = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(ax=700.0, ay=700.0),
        options=pcl.PeriodicOptions(method="ewald"),
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=2,
        periodic=periodic,
        solver_method="direct",
        verbose=False,
    )
    sim = pcl.Simulation(
        cfg,
        particles=[
            pcl.Sphere(
                position=(0.0, 0.0, 180.0),
                radius=80.0,
                refractive_index=1.5 + 0j,
            )
        ],
    )

    coeffs = np.asarray(sim.solve_sources({"source": src}).coeffs["source"]).reshape(-1)

    np.testing.assert_allclose(coeffs, _SMUTHI_ONE_SPHERE_COEFFS, rtol=1e-13, atol=1e-13)
