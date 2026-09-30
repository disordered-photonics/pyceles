"""Solve the canonical 500-sphere finite cluster with the MLFMM backend.

The geometry is shared with the CELES-style notebook and lives in
``examples/sphere_parameters.txt``.  CuPy is selected when a CUDA device is
available; otherwise the same public workflow falls back to NumPy.
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
    """Load the first 500 spheres from the bundled CELES geometry."""
    data = np.loadtxt(ROOT / "examples" / "sphere_parameters.txt")[:N_PARTICLES]
    return list(
        pcl.spheres_from_arrays(
            positions=data[:, 1:4],
            radii=data[:, 0],
            refractive_indices=data[:, 4] + 1j * data[:, 5],
        )
    )


def main() -> None:
    backend = _backend()
    source = pcl.GaussianBeam(
        wavelength=WAVELENGTH,
        medium_n=1.0 + 0.0j,
        polarization="TE",
        beam_width=2000.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )
    config = pcl.SimulationConfig(
        wavelength=WAVELENGTH,
        n_medium=1.0 + 0.0j,
        lmax=3,
        polar_angles=pcl.core.uniform_polar_grid(181),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(180),
        solver_method="bicgstab",
        solver_rtol=1.0e-4,
        solver_maxiter=800,
        operator_backend=backend,
        coupling_backend="mlfmm",
        postprocessing_backend="inherit",
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=True,
    )
    result = pcl.Simulation(config, particles=_particles()).run(source)
    solver = result.solver_result
    print(
        f"Finite MLFMM: N={result.n_particles}, backend={backend}, "
        f"iterations={solver.iterations}, residual={solver.relative_residual:.3e}"
    )
    if result.cross_sections is not None:
        print(
            "Cross sections: "
            f"C_ext={result.cross_sections.extinction:.6e}, "
            f"C_sca={result.cross_sections.scattering:.6e}, "
            f"C_abs={result.cross_sections.local_absorption:.6e}"
        )


if __name__ == "__main__":
    main()
