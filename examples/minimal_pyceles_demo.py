from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyceles as pcl


def main() -> None:
    # 4 non-overlapping spheres in the y=0 plane (good for y-slice near-field plotting)
    positions = np.array(
        [
            [-360.0, 0.0, -120.0],
            [-80.0, 0.0, 100.0],
            [180.0, 0.0, -80.0],
            [420.0, 0.0, 140.0],
        ],
        dtype=float,
    )
    radii = np.array([110.0, 90.0, 120.0, 80.0], dtype=float)
    n_particle = np.array([1.5 + 0.0j, 2.5 + 0.0j, 1.5 + 0.1j, 2.5 + 0.2j], dtype=np.complex128)

    source = pcl.PlaneWave(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=0.0,
        azimuthal_angle=0.0,
        amplitude=1.0,
    )
    cfg = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=3,
        source=source,
        compute_dtype="complex64",
        accum_dtype="complex128",
        polar_angles=np.linspace(0.0, np.pi, 3601, endpoint=True),
        azimuthal_angles=np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False),
        solver_method="auto",
        verbose=True,
    )

    sim = pcl.Simulation(cfg, positions=positions, radii=radii, n_particle=n_particle)
    run = sim.run()
    nf = pcl.compute_near_field_slice(
        run,
        x_min=-700.0,
        x_max=700.0,
        z_min=-500.0,
        z_max=500.0,
        dx=10.0,
        plane="y",
        plane_value=0.0,
        show_progress=True,
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pcl.save_simulation_h5(run, nf, out_dir / "minimal_pyceles_demo.h5")

    E, H = nf.field_maps["total"]
    fig_nf, _ = pcl.io.plot_nearfield_panels(
        nf.axis_0,
        nf.axis_1,
        E,
        H,
        run.positions,
        run.radii,
        plane=nf.plane,
        plane_value=nf.plane_value,
    )
    fig_nf.savefig(out_dir / "minimal_pyceles_demo_nearfield.png", dpi=200)
    plt.close(fig_nf)

    ff = run.farfield
    intensity = pcl.io.far_field_intensity(ff.scattered_te, ff.scattered_tm)
    fig_ff, _ = pcl.io.plot_farfield_hemispheres(
        polar_angles=ff.scattered_te["beta"],
        azimuthal_angles=ff.scattered_te["alpha"],
        intensity=intensity,
        cmap="inferno",
        independent_scales=True,
    )
    fig_ff.savefig(out_dir / "minimal_pyceles_demo_farfield.png", dpi=200)
    plt.close(fig_ff)


if __name__ == "__main__":
    main()
