from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import pyceles as pcl


def _render_outputs(
    run: pcl.SimulationResult,
    nf: pcl.postprocessing.NearFieldSlice,
    *,
    out_dir: Path,
    stem: str,
    auto_nearfield_limits: bool = False,
) -> None:
    pcl.save_simulation_h5(run, nf, out_dir / f"{stem}.h5")

    E, H = nf.field_maps["total"]
    real_limits = (-2.0, 2.0)
    abs_limits = (0.0, 2.0)
    if auto_nearfield_limits:
        # Use robust percentiles to avoid one-point near-singular spikes
        # dominating the full color scale in dipole slice visualizations.
        real_samples = np.concatenate(
            [np.abs(np.real(E)).reshape(-1), np.abs(np.real(H)).reshape(-1)]
        )
        real_samples = real_samples[np.isfinite(real_samples)]
        abs_samples = np.concatenate(
            [
                np.sqrt(np.sum(np.abs(E) ** 2, axis=-1)).reshape(-1),
                np.sqrt(np.sum(np.abs(H) ** 2, axis=-1)).reshape(-1),
            ]
        )
        abs_samples = abs_samples[np.isfinite(abs_samples)]
        real_peak = float(np.percentile(real_samples, 99.5)) if real_samples.size > 0 else 1e-12
        abs_peak = float(np.percentile(abs_samples, 99.5)) if abs_samples.size > 0 else 1e-12
        real_peak = max(real_peak, 1e-12)
        abs_peak = max(abs_peak, 1e-12)
        real_limits = (-real_peak, real_peak)
        abs_limits = (0.0, abs_peak)
    fig_nf, _ = pcl.io.plot_nearfield_panels(
        nf.axis_0,
        nf.axis_1,
        E,
        H,
        run.positions,
        run.radii,
        plane=nf.plane,
        plane_value=nf.plane_value,
        real_limits=real_limits,
        abs_limits=abs_limits,
    )
    fig_nf.savefig(out_dir / f"{stem}_nearfield.png", dpi=200)
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
    fig_ff.savefig(out_dir / f"{stem}_farfield.png", dpi=200)
    plt.close(fig_ff)


def _render_quick_ldos_map(
    sim: pcl.Simulation,
    *,
    out_dir: Path,
    stem: str,
    x_min: float = -700.0,
    x_max: float = 700.0,
    z_min: float = -500.0,
    z_max: float = 500.0,
    dx: float = 100.0,
    plane_value: float = 0.0,
    moment_magnitude: complex = 1.0 + 0j,
) -> None:
    """Compute and plot a coarse dipole-LDOS map on a y-slice."""
    cfg = sim.config
    sim_map = pcl.Simulation(
        replace(cfg, verbose=False),
        particles=list(sim.particles),
    )
    x = np.arange(float(x_min), float(x_max) + 0.5 * float(dx), float(dx), dtype=float)
    z = np.arange(float(z_min), float(z_max) + 0.5 * float(dx), float(dx), dtype=float)
    xx, zz = np.meshgrid(x, z)
    points = np.stack([xx, np.full_like(xx, float(plane_value)), zz], axis=-1)
    points_flat = points.reshape(-1, 3)

    inside = np.zeros((points_flat.shape[0],), dtype=bool)
    if sim_map.positions.shape[0] > 0:
        dr = points_flat[:, None, :] - sim_map.positions[None, :, :]
        inside = np.any(np.linalg.norm(dr, axis=2) < sim_map.radii[None, :], axis=1)
    inside = inside.reshape(xx.shape)

    ldos_px = np.full(xx.shape, np.nan, dtype=float)
    ldos_py = np.full(xx.shape, np.nan, dtype=float)
    ldos_pz = np.full(xx.shape, np.nan, dtype=float)

    valid_ids = np.argwhere(~inside)
    for i, j in tqdm(valid_ids, desc="LDOS map pixels", unit="px", leave=False):
        probe = pcl.DipoleSource(
            wavelength=float(cfg.wavelength),
            medium_n=complex(cfg.n_medium),
            position=(float(xx[i, j]), float(plane_value), float(zz[i, j])),
            dipole_moment=(moment_magnitude, 0.0 + 0j, 0.0 + 0j),
        )
        solved = sim_map.solve_sources(
            probe.cartesian_basis_sources(
                labels=("px", "py", "pz"),
                moment_magnitude=moment_magnitude,
            ),
            solver_compute_final_residual=False,
        )
        multi = sim_map.postprocess_sources(solved, include_farfield=False)
        ldos_px[i, j] = float(
            pcl.compute_dipole_ldos_enhancement(multi["px"], channel="mixed", show_progress=False)
        )
        ldos_py[i, j] = float(
            pcl.compute_dipole_ldos_enhancement(multi["py"], channel="mixed", show_progress=False)
        )
        ldos_pz[i, j] = float(
            pcl.compute_dipole_ldos_enhancement(multi["pz"], channel="mixed", show_progress=False)
        )

    ldos_avg = (ldos_px + ldos_py + ldos_pz) / 3.0
    panel_data = [
        (ldos_px, "Projected LDOS/Purcell enhancement (px)"),
        (ldos_py, "Projected LDOS/Purcell enhancement (py)"),
        (ldos_pz, "Projected LDOS/Purcell enhancement (pz)"),
        (ldos_avg, "Orientation-averaged enhancement"),
    ]
    finite_vals = np.concatenate([arr[np.isfinite(arr)] for arr, _ in panel_data])
    if finite_vals.size > 0:
        delta = max(1e-6, float(np.max(np.abs(finite_vals - 1.0))))
    else:
        delta = 1e-3
    vmin = 1.0 - delta
    vmax = 1.0 + delta
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6), constrained_layout=True)
    panel_data = [
        (ldos_px, "LDOS enhancement (px)"),
        (ldos_py, "LDOS enhancement (py)"),
        (ldos_pz, "LDOS enhancement (pz)"),
        (ldos_avg, "LDOS enhancement (avg)"),
    ]
    for ax, (data, title) in zip(np.ravel(axes), panel_data):
        im = pcl.io.plot_field_component(
            ax,
            xx,
            zz,
            data,
            title=title,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            axis_0_label="x",
            axis_1_label="z",
        )
        im.set_norm(norm)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        pcl.io.plot_spheres(
            ax,
            sim_map.positions,
            sim_map.radii,
            plane="y",
            plane_value=float(plane_value),
            alpha=0.7,
            color="w",
        )
    fig.savefig(out_dir / f"{stem}_ldos.png", dpi=200)
    plt.close(fig)


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
        polar_angles=pcl.core.uniform_polar_grid(721),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(360),
        solver_method="auto",
        verbose=True,
    )

    particles = pcl.core.spheres_from_arrays(
        positions=positions,
        radii=radii,
        refractive_indices=n_particle,
    )
    sim = pcl.Simulation(cfg, particles=particles)
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

    # Same 4-particle geometry, now excited by three local dipoles with
    # distinct positions/orientations (x/y/z moments).
    dipoles = pcl.DipoleCollection(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        positions=np.array(
            [
                [-310.0, 0.0, 60.0],
                [0.0, 0.0, 0.0],
                [390.0, 0.0, -30.0],
            ],
            dtype=float,
        ),
        dipole_moments=np.array(
            [
                [1.0 + 0j, 0.0 + 0j, 0.0 + 0j],  # px
                [0.0 + 0j, 1.0 + 0j, 0.0 + 0j],  # py
                [0.0 + 0j, 0.0 + 0j, 1.0 + 0j],  # pz
            ],
            dtype=np.complex128,
        ),
    )
    cfg_dip = pcl.SimulationConfig(
        wavelength=550.0,
        n_medium=1.0 + 0j,
        lmax=4,
        source=dipoles,
        compute_dtype="complex128",
        accum_dtype="complex128",
        polar_angles=pcl.core.uniform_polar_grid(721),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(360),
        solver_method="direct",
        solver_compute_final_residual=False,
        verbose=True,
    )
    sim_dip = pcl.Simulation(cfg_dip, particles=particles)
    run_dip = sim_dip.run()
    nf_dip = pcl.compute_near_field_slice(
        run_dip,
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
    _render_outputs(run, nf, out_dir=out_dir, stem="minimal_pyceles_demo_planewave")
    _render_outputs(
        run_dip,
        nf_dip,
        out_dir=out_dir,
        stem="minimal_pyceles_demo_dipoles",
        auto_nearfield_limits=True,
    )
    _render_quick_ldos_map(
        sim_dip,
        out_dir=out_dir,
        stem="minimal_pyceles_demo_dipoles",
        plane_value=0.0,
        dx=100.0,
    )


if __name__ == "__main__":
    main()
