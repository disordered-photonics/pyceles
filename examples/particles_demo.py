"""Render the 500-particle particle-family demonstration used in the notebook.

The same geometry is solved three times: as concentric core--shell spheres,
randomly oriented prolate spheroids, and perfect-electric-conductor spheres.
Each case produces one near-field canvas and one scattered far-field canvas in
``outputs/particles_demo``.  The script deliberately keeps the physical setup
fixed so that the effect of changing the particle representation is easy to
inspect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np

import pyceles as pcl

Backend = Literal["numpy", "cupy"]

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "particles_demo"
WAVELENGTH = 550.0
N_PARTICLES = 500


def _backend() -> Backend:
    """Use CuPy when a CUDA device is available, otherwise use NumPy."""
    try:
        from pyceles._optional import import_cupy

        cupy, _ = import_cupy()
        if int(cupy.cuda.runtime.getDeviceCount()) > 0:
            return "cupy"
    except Exception:  # pragma: no cover - depends on the local installation
        pass
    return "numpy"


def _load_reference_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(ROOT / "examples" / "sphere_parameters.txt")[:N_PARTICLES]
    positions = np.asarray(data[:, 1:4], dtype=float)
    radii = np.asarray(data[:, 0], dtype=float)
    refractive_indices = np.asarray(data[:, 4] + 1j * data[:, 5], dtype=np.complex128)
    return positions, radii, refractive_indices


def _particle_families(
    positions: np.ndarray,
    radii: np.ndarray,
    refractive_indices: np.ndarray,
) -> dict[str, list[pcl.Particle]]:
    rng = np.random.default_rng(7)
    layered = [
        pcl.LayeredSphere(
            position=tuple(position),
            layer_radii=(0.6 * float(radius), float(radius)),
            layer_refractive_indices=(1.0 + 0.0j, complex(index)),
        )
        for position, radius, index in zip(positions, radii, refractive_indices, strict=True)
    ]
    spheroids = [
        pcl.Spheroid(
            position=tuple(position),
            equatorial_radius=0.6 * float(radius),
            polar_radius=float(radius),
            refractive_index=complex(index),
            euler_angles=tuple(rng.uniform(0.0, 2.0 * np.pi, size=3)),
        )
        for position, radius, index in zip(positions, radii, refractive_indices, strict=True)
    ]
    pec = [
        pcl.PECSphere(position=tuple(position), radius=float(radius))
        for position, radius in zip(positions, radii, strict=True)
    ]
    return {
        "core_shell": cast(list[pcl.Particle], layered),
        "spheroids": cast(list[pcl.Particle], spheroids),
        "pec": cast(list[pcl.Particle], pec),
    }


def _configuration(backend: Backend) -> pcl.SimulationConfig:
    return pcl.SimulationConfig(
        wavelength=WAVELENGTH,
        n_medium=1.0 + 0.0j,
        lmax=3,
        polar_angles=pcl.core.uniform_polar_grid(721),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(360),
        solver_method="gmres",
        solver_rtol=1.0e-4,
        solver_restart=25,
        solver_maxiter=100,
        compute_dtype="complex64",
        accum_dtype="complex128",
        operator_backend=backend,
        postprocessing_backend=backend,
        verbose=True,
    )


def _source() -> pcl.LaguerreGaussianBeam:
    return pcl.LaguerreGaussianBeam(
        wavelength=WAVELENGTH,
        medium_n=1.0 + 0.0j,
        polarization=(1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)),
        radial_order_p=0,
        azimuthal_order_l=1,
        polar_angle=0.32,
        azimuthal_angle=0.21,
        beam_width=1800.0,
        focal_point=(0.0, 0.0, 0.0),
        amplitude=1.0,
    )


def _render_case(
    label: str,
    simulation: pcl.Simulation,
    source: pcl.LaguerreGaussianBeam,
) -> None:
    result = simulation.run(source)
    near_field = pcl.compute_near_field_slice(
        result,
        axis_0_min=-1500.0,
        axis_0_max=1500.0,
        axis_1_min=0.0,
        axis_1_max=3000.0,
        dx=30.0,
        plane="y",
        plane_value=0.0,
        show_progress=True,
    )
    electric, magnetic = near_field.field_maps["total"]
    figure_near, _ = pcl.io.plot_nearfield_panels(
        near_field.axis_0,
        near_field.axis_1,
        electric,
        magnetic,
        particles=result.particles,
        plane=near_field.plane,
        plane_value=near_field.plane_value,
        real_limits=(-2.0, 2.0),
        abs_limits=(0.0, 2.0),
    )
    figure_near.suptitle(f"{label}: total near field", y=1.02)
    figure_near.savefig(OUTPUT_DIR / f"{label}_nearfield.png", dpi=200, bbox_inches="tight")
    plt.close(figure_near)

    far_field = result.farfield.scattered
    intensity = pcl.io.far_field_intensity(far_field)
    figure_far, _ = pcl.io.plot_farfield_hemispheres(
        polar_angles=far_field.beta,
        azimuthal_angles=far_field.alpha,
        intensity=intensity,
        cmap="inferno",
        independent_scales=True,
        title=f"{label}: scattered far-field intensity",
    )
    figure_far.savefig(OUTPUT_DIR / f"{label}_farfield.png", dpi=200, bbox_inches="tight")
    plt.close(figure_far)

    solver = result.solver_result
    power = result.power
    if power is None:
        raise RuntimeError("The particle demo requires far-field power diagnostics.")
    print(
        f"{label}: iterations={solver.iterations}, "
        f"relative residual={solver.relative_residual:.3e}, "
        f"T={power.transmittance:.6f}, R={power.reflectance:.6f}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    positions, radii, refractive_indices = _load_reference_geometry()
    families = _particle_families(positions, radii, refractive_indices)
    backend = _backend()
    config = _configuration(backend)
    source = _source()
    print(f"Particle demo: {N_PARTICLES} particles, backend={backend}")
    for label, particles in families.items():
        _render_case(label, pcl.Simulation(config, particles=particles), source)


if __name__ == "__main__":
    main()
