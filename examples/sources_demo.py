"""Render the curated incident-source family demonstration.

The fixed setup generates near-field panels for the Gaussian,
Laguerre--Gaussian, focused, Bessel, and Cartesian-polarized source models.
Outputs are written to ``outputs/sources_demo`` so the example remains a small,
reproducible illustration of the source API rather than a configurable
benchmark driver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

import pyceles as pcl

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "sources_demo"

SourceName = Literal[
    "plane_wave",
    "gaussian",
    "laguerre_gaussian",
    "focused_laguerre_gaussian",
    "focused_laguerre_gaussian_cartesian",
    "bessel",
    "bessel_cartesian",
]


def _derived_bessel_cone_angle(*, wavelength: float, gaussian_beam_width: float) -> float:
    """Return a Bessel cone angle roughly matching Gaussian angular spread."""
    k0 = 2.0 * np.pi / float(wavelength)
    sigma_theta = 2.0 / (k0 * float(gaussian_beam_width))
    return float(np.clip(sigma_theta, 0.03, 0.7))


def _derived_focused_na(
    *,
    wavelength: float,
    gaussian_beam_width: float,
    n_medium: complex,
) -> float:
    """Return a focused-LG NA matched to the Gaussian/Bessel reference spread."""
    alpha_ref = _derived_bessel_cone_angle(
        wavelength=float(wavelength),
        gaussian_beam_width=float(gaussian_beam_width),
    )
    na = float(np.real(complex(n_medium))) * float(np.sin(alpha_ref))
    return float(np.clip(na, 0.03, float(np.real(complex(n_medium))) * 0.95))


def _make_no_particle_simulation(
    *,
    wavelength: float,
    n_medium: complex,
    lmax: int,
    n_polar: int,
    n_azimuth: int,
    verbose: bool,
) -> pcl.Simulation:
    """Create a no-particle simulation carrying only the selected source."""
    cfg = pcl.SimulationConfig(
        wavelength=float(wavelength),
        n_medium=complex(n_medium),
        lmax=int(lmax),
        polar_angles=pcl.core.uniform_polar_grid(int(n_polar)),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(int(n_azimuth)),
        solver_method="gmres",
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=bool(verbose),
    )
    return pcl.Simulation(
        cfg,
        particles=[],
    )


def _build_source(
    name: SourceName,
    *,
    wavelength: float,
    n_medium: complex,
    gaussian_beam_width: float,
    laguerre_radial_order: int,
    laguerre_azimuthal_order: int,
    focused_focal_length: float,
    focused_numerical_aperture: float,
    bessel_cone_angle: float,
    bessel_order: int = 0,
) -> tuple[str, pcl.core.Source]:
    """Build one source definition for the showcase run."""
    if name == "plane_wave":
        return (
            "plane_wave",
            pcl.PlaneWave(
                wavelength=float(wavelength),
                medium_n=complex(n_medium),
                amplitude=1.0,
                polarization=(1.0 + 0.0j, 0.35j),
                polar_angle=0.0,
                azimuthal_angle=0.0,
                focal_point=(0.0, 0.0, 0.0),
            ),
        )
    if name == "gaussian":
        return (
            "gaussian",
            pcl.GaussianBeam(
                wavelength=float(wavelength),
                medium_n=complex(n_medium),
                amplitude=1.0,
                polarization=(1.0 + 0.0j, 0.35j),
                polar_angle=0.0,
                azimuthal_angle=0.0,
                beam_width=float(gaussian_beam_width),
                focal_point=(0.0, 0.0, 0.0),
            ),
        )
    if name == "laguerre_gaussian":
        return (
            f"laguerre_gaussian_p{int(laguerre_radial_order)}_l{int(laguerre_azimuthal_order)}",
            pcl.LaguerreGaussianBeam(
                wavelength=float(wavelength),
                medium_n=complex(n_medium),
                amplitude=1.0,
                polarization=(1.0 + 0.0j, 0.35j),
                radial_order_p=int(laguerre_radial_order),
                azimuthal_order_l=int(laguerre_azimuthal_order),
                polar_angle=0.0,
                azimuthal_angle=0.0,
                beam_width=float(gaussian_beam_width),
                focal_point=(0.0, 0.0, 0.0),
                azimuthal_phase=0.0,
            ),
        )
    if name == "focused_laguerre_gaussian":
        return (
            (
                f"focused_laguerre_gaussian_p{int(laguerre_radial_order)}"
                f"_l{int(laguerre_azimuthal_order)}"
            ),
            pcl.FocusedLaguerreGaussianBeam(
                wavelength=float(wavelength),
                medium_n=complex(n_medium),
                amplitude=1.0,
                polarization=(1.0 + 0.0j, 0.35j),
                radial_order_p=int(laguerre_radial_order),
                azimuthal_order_l=int(laguerre_azimuthal_order),
                polar_angle=0.0,
                azimuthal_angle=0.0,
                beam_width=float(gaussian_beam_width),
                focal_length=float(focused_focal_length),
                numerical_aperture=float(focused_numerical_aperture),
                focal_point=(0.0, 0.0, 0.0),
                azimuthal_phase=0.0,
                sine_condition_apodization=True,
            ),
        )
    if name == "focused_laguerre_gaussian_cartesian":
        return (
            (
                f"focused_laguerre_gaussian_cartesian_p{int(laguerre_radial_order)}"
                f"_l{int(laguerre_azimuthal_order)}"
            ),
            pcl.CartesianPolarizedFocusedLaguerreGaussianBeam(
                wavelength=float(wavelength),
                medium_n=complex(n_medium),
                amplitude=1.0,
                radial_order_p=int(laguerre_radial_order),
                azimuthal_order_l=int(laguerre_azimuthal_order),
                polar_angle=0.0,
                azimuthal_angle=0.0,
                global_polarization=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
                beam_width=float(gaussian_beam_width),
                focal_length=float(focused_focal_length),
                numerical_aperture=float(focused_numerical_aperture),
                focal_point=(0.0, 0.0, 0.0),
                azimuthal_phase=0.0,
                sine_condition_apodization=True,
            ),
        )
    if name == "bessel":
        return (
            f"bessel_m{int(bessel_order)}",
            pcl.BesselBeam(
                wavelength=float(wavelength),
                medium_n=complex(n_medium),
                amplitude=1.0,
                polarization=(1.0 + 0.0j, 0.35j),
                order_m=int(bessel_order),
                cone_angle=float(bessel_cone_angle),
                polar_angle=0.0,
                azimuthal_angle=0.0,
                center=(0.0, 0.0, 0.0),
                azimuthal_phase=0.0,
                forward_only=True,
            ),
        )
    if name == "bessel_cartesian":
        return (
            f"bessel_cartesian_m{int(bessel_order)}",
            pcl.CartesianPolarizedBesselBeam(
                wavelength=float(wavelength),
                medium_n=complex(n_medium),
                amplitude=1.0,
                order_m=int(bessel_order),
                cone_angle=float(bessel_cone_angle),
                polar_angle=0.0,
                azimuthal_angle=0.0,
                global_polarization=(1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j),
                center=(0.0, 0.0, 0.0),
                azimuthal_phase=0.0,
                forward_only=True,
            ),
        )
    raise ValueError(f"Unsupported source name {name!r}.")


def _render_source_showcase(
    *,
    run: pcl.SimulationResult,
    label: str,
    wavelength: float,
    source_note: str,
    out_dir: Path,
    half_span: float,
    dx: float,
    real_limit: float | None,
    abs_limit: float | None,
    phase_cmap: str,
    dpi: int,
    show_progress: bool,
) -> Path:
    """Render one 3x5 source-showcase figure and return output path."""
    real_limits = None if real_limit is None else (-float(real_limit), float(real_limit))
    abs_limits = None if abs_limit is None else (0.0, float(abs_limit))
    fig, _ = pcl.io.plot_source_showcase_slices(
        run,
        field_component="initial",
        plane_values=(0.0, 0.0, 0.0),
        phase_component="Ex",
        phase_cmap=str(phase_cmap),
        axis_0_min=-float(half_span),
        axis_0_max=float(half_span),
        axis_1_min=-float(half_span),
        axis_1_max=float(half_span),
        dx=float(dx),
        real_limits=real_limits,
        abs_limits=abs_limits,
        rowwise_percentile=99.5,
        show_progress=bool(show_progress),
        center_pixel_policy="none",
    )
    fig.suptitle(
        (
            f"Source showcase ({label})  lambda={wavelength:.1f} nm  "
            f"| window=[-{half_span:.0f}, {half_span:.0f}] nm, dx={dx:.0f} nm  | {source_note}"
        ),
        fontsize=12,
    )
    out_path = out_dir / f"sources_demo_{label}.png"
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    return out_path


def _source_note(source: pcl.core.Source) -> str:
    """Return concise source-specific parameters for figure subtitles."""
    if isinstance(source, pcl.GaussianBeam):
        f = tuple(float(v) for v in source.focal_point)
        return f"Gaussian: beam_width={float(source.beam_width):.0f} nm, focal_point={f}"
    if isinstance(source, pcl.BesselBeam):
        c = tuple(float(v) for v in source.center)
        return (
            "Bessel: "
            f"m={int(source.order_m)}, cone_angle={float(source.cone_angle):.4f} rad, center={c}"
        )
    if isinstance(source, pcl.CartesianPolarizedBesselBeam):
        c = tuple(float(v) for v in source.center)
        gp = tuple(complex(v) for v in source.global_polarization)
        return (
            "Bessel (global Cartesian polarization): "
            f"m={int(source.order_m)}, cone_angle={float(source.cone_angle):.4f} rad, "
            f"global_polarization={gp}, center={c}"
        )
    if isinstance(source, pcl.LaguerreGaussianBeam):
        f = tuple(float(v) for v in source.focal_point)
        return (
            "Laguerre-Gaussian: "
            f"p={int(source.radial_order_p)}, l={int(source.azimuthal_order_l)}, "
            f"beam_width={float(source.beam_width):.0f} nm, focal_point={f}"
        )
    if isinstance(source, pcl.FocusedLaguerreGaussianBeam):
        f = tuple(float(v) for v in source.focal_point)
        return (
            "Focused Laguerre-Gaussian: "
            f"p={int(source.radial_order_p)}, l={int(source.azimuthal_order_l)}, "
            f"beam_width={float(source.beam_width):.0f} nm, "
            f"focal_length={float(source.focal_length):.0f} nm, "
            f"NA={float(source.numerical_aperture):.3f}, focal_point={f}"
        )
    if isinstance(source, pcl.CartesianPolarizedFocusedLaguerreGaussianBeam):
        f = tuple(float(v) for v in source.focal_point)
        gp = tuple(complex(v) for v in source.global_polarization)
        return (
            "Focused Laguerre-Gaussian (global Cartesian polarization): "
            f"p={int(source.radial_order_p)}, l={int(source.azimuthal_order_l)}, "
            f"beam_width={float(source.beam_width):.0f} nm, "
            f"focal_length={float(source.focal_length):.0f} nm, "
            f"NA={float(source.numerical_aperture):.3f}, "
            f"global_polarization={gp}, focal_point={f}"
        )
    if isinstance(source, pcl.PlaneWave):
        f = tuple(float(v) for v in source.focal_point)
        return f"Plane wave: focal_point={f}"
    return f"{type(source).__name__}"


def _propagation_and_polarization_note(source: pcl.core.Source) -> str:
    """Return propagation-axis and polarization metadata for figure subtitles."""
    if hasattr(source, "polar_angle"):
        beta = float(source.polar_angle)
        direction = "+z" if float(np.cos(beta)) >= 0.0 else "-z"
        prop = f"propagation={direction} (polar_angle={beta:.3f} rad)"
    else:
        prop = "propagation=n/a"

    if hasattr(source, "polarization"):
        pol = source.polarization
        pol_txt = f"polarization={pol!r}"
    elif hasattr(source, "global_polarization"):
        pol = source.global_polarization
        pol_txt = f"global_polarization={pol!r}"
    else:
        pol_txt = "polarization=n/a"
    return f"{prop} | {pol_txt}"


def main() -> None:
    wavelength = 550.0
    n_medium = 1.0 + 0.0j
    lmax = 3
    # The polar grid includes both endpoints, whereas the azimuthal grid is
    # periodic and excludes 2*pi.
    n_polar, n_azimuth = 181, 180
    half_span, dx = 5000.0, 50.0
    beam_width = 1000.0
    laguerre_p, laguerre_l = 0, 1
    focused_focal_length = 1000.0
    cone_angle = _derived_bessel_cone_angle(wavelength=wavelength, gaussian_beam_width=beam_width)
    focused_na = _derived_focused_na(
        wavelength=wavelength,
        gaussian_beam_width=beam_width,
        n_medium=n_medium,
    )
    source_names: tuple[SourceName, ...] = (
        "plane_wave",
        "gaussian",
        "laguerre_gaussian",
        "focused_laguerre_gaussian",
        "focused_laguerre_gaussian_cartesian",
        "bessel",
        "bessel_cartesian",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Wavelength: {wavelength:.1f} nm | medium_n: {n_medium.real:.3f}")

    for src_name in source_names:
        bessel_orders = (0, 1) if src_name in {"bessel", "bessel_cartesian"} else (0,)
        for bessel_order in bessel_orders:
            label, source = _build_source(
                src_name,
                wavelength=wavelength,
                n_medium=n_medium,
                gaussian_beam_width=beam_width,
                laguerre_radial_order=laguerre_p,
                laguerre_azimuthal_order=laguerre_l,
                focused_focal_length=focused_focal_length,
                focused_numerical_aperture=focused_na,
                bessel_cone_angle=cone_angle,
                bessel_order=int(bessel_order),
            )
            print(f"Running showcase for {label}...")
            sim = _make_no_particle_simulation(
                wavelength=wavelength,
                n_medium=n_medium,
                lmax=lmax,
                n_polar=n_polar,
                n_azimuth=n_azimuth,
                verbose=True,
            )
            run = sim.run(source, include_farfield=False)
            source_note = f"{_source_note(source)} | {_propagation_and_polarization_note(source)}"
            out_path = _render_source_showcase(
                run=run,
                label=label,
                wavelength=wavelength,
                source_note=source_note,
                out_dir=OUTPUT_DIR,
                half_span=half_span,
                dx=dx,
                real_limit=None,
                abs_limit=None,
                phase_cmap="twilight_shifted",
                dpi=220,
                show_progress=True,
            )
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
