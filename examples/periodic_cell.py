"""Solve a compact 500-sphere periodic cell with the Rayleigh backend.

The cell reuses the canonical CELES sphere set and wraps its lateral
coordinates into a 4 micrometre square lattice.  CuPy is selected when a CUDA
device is available; otherwise the same public workflow falls back to NumPy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

import pyceles as pcl

Backend = Literal["numpy", "cupy"]
ROOT = Path(__file__).resolve().parents[1]
WAVELENGTH = 550.0
N_PARTICLES = 500
CELL_SIDE = 4000.0


def _backend() -> Backend:
    """Use CuPy when a CUDA device is available, otherwise use NumPy."""
    try:
        import cupy

        if int(cupy.cuda.runtime.getDeviceCount()) > 0:
            return "cupy"
    except Exception:  # pragma: no cover - depends on the local installation
        pass
    return "numpy"


def _particles() -> list[pcl.Particle]:
    """Load and wrap the first 500 spheres into the periodic lateral cell."""
    data = np.loadtxt(ROOT / "examples" / "sphere_parameters.txt")[:N_PARTICLES].copy()
    data[:, 1:3] = np.mod(data[:, 1:3] + 0.5 * CELL_SIDE, CELL_SIDE) - 0.5 * CELL_SIDE
    return list(
        pcl.spheres_from_arrays(
            positions=data[:, 1:4],
            radii=data[:, 0],
            refractive_indices=data[:, 4] + 1j * data[:, 5],
        )
    )


def main() -> None:
    backend = _backend()
    source = pcl.PlaneWave(
        wavelength=WAVELENGTH,
        medium_n=1.0 + 0.0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(ax=CELL_SIDE, ay=CELL_SIDE),
        options=pcl.PeriodicOptions(
            method="rayleigh",
            rayleigh_reciprocal_shells=32,
        ),
    )
    config = pcl.SimulationConfig(
        wavelength=WAVELENGTH,
        n_medium=1.0 + 0.0j,
        lmax=3,
        polar_angles=pcl.core.uniform_polar_grid(181),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(180),
        solver_method="gmres",
        solver_rtol=1.0e-4,
        solver_restart=80,
        solver_maxiter=800,
        periodic=periodic,
        operator_backend=backend,
        postprocessing_backend="inherit",
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=True,
    )
    result = pcl.Simulation(config, particles=_particles()).run(source)
    solver = result.solver_result
    print(
        f"Periodic Rayleigh: N={result.n_particles}, backend={backend}, "
        f"iterations={solver.iterations}, residual={solver.relative_residual:.3e}"
    )
    if result.periodic is None:
        raise RuntimeError("The periodic example did not produce order diagnostics.")
    power = result.periodic.power
    propagating = int(np.count_nonzero(result.periodic.order_propagating))
    print(
        f"Orders: {propagating}/{result.periodic.order_mn.shape[0]} propagating; "
        f"R={power.reflectance:.6f}, T={power.transmittance:.6f}, "
        f"A={power.local_absorptance:.6f}, defect={power.flux_defect_fraction:.3e}"
    )


if __name__ == "__main__":
    main()
