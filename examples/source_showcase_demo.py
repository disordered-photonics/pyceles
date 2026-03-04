from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np

import pyceles as pcl

SourceName = Literal[
    "plane_wave",
    "gaussian",
    "laguerre_gaussian",
    "focused_laguerre_gaussian",
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
    source: pcl.core.Source,
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
        source=source,
        polar_angles=pcl.core.uniform_polar_grid(int(n_polar)),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(int(n_azimuth)),
        solver_method="auto",
        compute_dtype="complex64",
        accum_dtype="complex128",
        verbose=bool(verbose),
    )
    return pcl.Simulation(
        cfg,
        positions=np.zeros((0, 3), dtype=float),
        radii=np.zeros((0,), dtype=float),
        n_particle=np.zeros((0,), dtype=np.complex128),
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
        channel="mixed",
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
    out_path = out_dir / f"source_showcase_{label}.png"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate no-particle initial-field source showcases (3x5 panels) for "
            "normal-incidence z-axis-aligned sources."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=[
            "plane_wave",
            "gaussian",
            "laguerre_gaussian",
            "focused_laguerre_gaussian",
            "bessel",
            "bessel_cartesian",
        ],
        default=[
            "plane_wave",
            "gaussian",
            "laguerre_gaussian",
            "focused_laguerre_gaussian",
            "bessel",
            "bessel_cartesian",
        ],
        help="Sources to render.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/source_showcase"))
    parser.add_argument("--wavelength", type=float, default=550.0, help="Wavelength in nm.")
    parser.add_argument("--n-medium", type=float, default=1.0, help="Background refractive index.")
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--n-polar", type=int, default=361, help="Source polar grid samples.")
    parser.add_argument("--n-azimuth", type=int, default=181, help="Source azimuth grid samples.")
    parser.add_argument(
        "--half-span",
        type=float,
        default=5000.0,
        help="Half-window size (nm) for all showcase slices.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=50.0,
        help="Slice sampling step in nm (coarser step to cover wider windows).",
    )
    parser.add_argument(
        "--gaussian-beam-width",
        type=float,
        default=1000.0,
        help="Gaussian beam width parameter in nm.",
    )
    parser.add_argument(
        "--bessel-cone-angle",
        type=float,
        default=np.nan,
        help=(
            "Bessel cone angle in radians. Default derives from Gaussian beam width "
            "to keep comparable convergence behavior."
        ),
    )
    parser.add_argument(
        "--laguerre-p",
        type=int,
        default=0,
        help="Laguerre-Gaussian radial index p for both collimated/focused LG sources.",
    )
    parser.add_argument(
        "--laguerre-l",
        type=int,
        default=1,
        help="Laguerre-Gaussian azimuthal index l for both collimated/focused LG sources.",
    )
    parser.add_argument(
        "--focused-focal-length",
        type=float,
        default=1000.0,
        help="Focused LG focal length in nm.",
    )
    parser.add_argument(
        "--focused-na",
        type=float,
        default=np.nan,
        help=(
            "Focused LG numerical aperture (must be < n_medium). "
            "Default derives from the Gaussian beam width to match divergence."
        ),
    )
    parser.add_argument(
        "--real-limit",
        type=float,
        default=np.nan,
        help="Optional fixed |Re(E*)| color limit. Default uses row-wise robust auto-scaling.",
    )
    parser.add_argument(
        "--abs-limit",
        type=float,
        default=np.nan,
        help="Optional fixed |E| color limit. Default uses row-wise robust auto-scaling.",
    )
    parser.add_argument("--phase-cmap", type=str, default="twilight_shifted")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.dx) <= 0.0:
        raise ValueError("`dx` must be > 0.")
    if float(args.half_span) <= 0.0:
        raise ValueError("`half_span` must be > 0.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavelength = float(args.wavelength)
    n_medium = complex(float(args.n_medium), 0.0)
    beam_width = float(args.gaussian_beam_width)
    cone_angle = (
        _derived_bessel_cone_angle(wavelength=wavelength, gaussian_beam_width=beam_width)
        if not np.isfinite(float(args.bessel_cone_angle))
        else float(args.bessel_cone_angle)
    )
    real_limit = float(args.real_limit) if np.isfinite(float(args.real_limit)) else None
    abs_limit = float(args.abs_limit) if np.isfinite(float(args.abs_limit)) else None

    focused_na = (
        _derived_focused_na(
            wavelength=wavelength,
            gaussian_beam_width=beam_width,
            n_medium=n_medium,
        )
        if not np.isfinite(float(args.focused_na))
        else float(args.focused_na)
    )

    if focused_na >= float(np.real(n_medium)):
        raise ValueError(
            "`focused-na` must be smaller than n_medium. "
            f"Got focused_na={focused_na!r}, n_medium={float(np.real(n_medium))!r}."
        )

    if not args.quiet:
        print(f"Output directory: {out_dir}")
        print(f"Wavelength: {wavelength:.1f} nm | medium_n: {n_medium.real:.3f}")
        print(f"Shared showcase window: [-{args.half_span:.1f}, {args.half_span:.1f}] nm")
        print(f"Shared showcase dx: {args.dx:.1f} nm")
        print(f"Gaussian beam width: {beam_width:.1f} nm")
        print(f"Laguerre mode indices: p={int(args.laguerre_p)}, l={int(args.laguerre_l)}")
        print(
            "Focused LG: "
            f"focal_length={float(args.focused_focal_length):.1f} nm, "
            f"NA={focused_na:.3f}"
        )
        print(f"Bessel cone angle: {cone_angle:.4f} rad")
        if real_limit is None:
            print("Re(E*) scaling: row-wise robust auto")
        else:
            print(f"Re(E*) scaling: fixed +/-{real_limit:.3g}")
        if abs_limit is None:
            print("|E| scaling: row-wise robust auto")
        else:
            print(f"|E| scaling: fixed [0, {abs_limit:.3g}]")

    for src_name_raw in args.sources:
        if src_name_raw not in {
            "plane_wave",
            "gaussian",
            "laguerre_gaussian",
            "focused_laguerre_gaussian",
            "bessel",
            "bessel_cartesian",
        }:
            raise ValueError(f"Unsupported source {src_name_raw!r}.")
        src_name = cast(SourceName, src_name_raw)

        bessel_orders = (0, 1) if src_name in {"bessel", "bessel_cartesian"} else (0,)
        for bessel_order in bessel_orders:
            label, source = _build_source(
                src_name,
                wavelength=wavelength,
                n_medium=n_medium,
                gaussian_beam_width=beam_width,
                laguerre_radial_order=int(args.laguerre_p),
                laguerre_azimuthal_order=int(args.laguerre_l),
                focused_focal_length=float(args.focused_focal_length),
                focused_numerical_aperture=focused_na,
                bessel_cone_angle=cone_angle,
                bessel_order=int(bessel_order),
            )
            if not args.quiet:
                print(f"Running showcase for {label}...")
            sim = _make_no_particle_simulation(
                source,
                wavelength=wavelength,
                n_medium=n_medium,
                lmax=int(args.lmax),
                n_polar=int(args.n_polar),
                n_azimuth=int(args.n_azimuth),
                verbose=(not args.quiet),
            )
            run = sim.run(include_farfield=False)
            source_note = f"{_source_note(source)} | {_propagation_and_polarization_note(source)}"
            out_path = _render_source_showcase(
                run=run,
                label=label,
                wavelength=wavelength,
                source_note=source_note,
                out_dir=out_dir,
                half_span=float(args.half_span),
                dx=float(args.dx),
                real_limit=real_limit,
                abs_limit=abs_limit,
                phase_cmap=str(args.phase_cmap),
                dpi=int(args.dpi),
                show_progress=(not args.quiet),
            )
            if not args.quiet:
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
