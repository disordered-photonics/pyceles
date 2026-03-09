from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

import pyceles as pcl
from pyceles.core.fields import project_source_to_svwf
from pyceles.core.indexing import n_modes
from pyceles.core.matvec import (
    estimate_translation_cache_bytes,
    make_prepared_A_and_rhs,
    prepare_matvec,
)
from pyceles.linear.preconditioner import make_grid_block_preconditioner
from pyceles.linear.solvers import solve_linear_system
from pyceles.postprocessing.farfield import compute_far_field_patterns


def _find_repo_root() -> Path:
    here = Path.cwd().resolve()
    if (here / "examples" / "sphere_parameters.txt").exists():
        return here
    if (here.parent / "examples" / "sphere_parameters.txt").exists():
        return here.parent
    raise FileNotFoundError("Could not find examples/sphere_parameters.txt")


def _load_geometry(path: Path, n_particles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    n = int(n_particles)
    if n < 1 or n > data.shape[0]:
        raise ValueError(f"n_particles must be in [1, {data.shape[0]}], got {n_particles}.")
    radii = data[:n, 0].astype(float)
    positions = data[:n, 1:4].astype(float)
    n_particle = (data[:n, 4] + 1j * data[:n, 5]).astype(np.complex128)
    return positions, radii, n_particle


def _top_stats(prof: cProfile.Profile, limit: int) -> list[dict[str, Any]]:
    stats = pstats.Stats(prof)
    rows: list[dict[str, Any]] = []
    for func, payload in cast(Any, stats).stats.items():
        cc, nc, tt, ct, _ = payload
        rows.append(
            {
                "file": str(func[0]),
                "line": int(func[1]),
                "function": str(func[2]),
                "primitive_calls": int(cc),
                "total_calls": int(nc),
                "tottime_s": float(tt),
                "cumtime_s": float(ct),
            }
        )
    rows.sort(key=lambda x: x["cumtime_s"], reverse=True)
    return rows[: int(limit)]


def _profile_phase(
    *,
    phase: str,
    out_dir: Path,
    top_n: int,
    fn,
    **kwargs,
) -> tuple[Any, dict[str, Any]]:
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    result = prof.runcall(fn, **kwargs)
    elapsed = time.perf_counter() - t0

    prof_path = out_dir / f"{phase}.prof"
    txt_path = out_dir / f"{phase}.txt"
    prof.dump_stats(str(prof_path))

    with txt_path.open("w", encoding="utf-8") as f:
        st = pstats.Stats(prof, stream=f)
        st.sort_stats("cumulative")
        st.print_stats(int(top_n))

    summary = {
        "phase": phase,
        "wall_time_s": float(elapsed),
        "prof_file": str(prof_path),
        "text_report": str(txt_path),
        "top_functions": _top_stats(prof, top_n),
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile pyceles solver/far-field/near-field phases."
    )
    parser.add_argument("--n-particles", type=int, default=500)
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--wavelength", type=float, default=550.0)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--n-beta", type=int, default=3601)
    parser.add_argument("--n-alpha", type=int, default=180)
    parser.add_argument("--dx", type=float, default=80.0, help="Near-field grid spacing.")
    parser.add_argument(
        "--source-model",
        choices=("planewave", "gaussian"),
        default="gaussian",
        help="Incident source model used for projection, far field, and near-field initial component.",
    )
    parser.add_argument("--polarization", choices=("TE", "TM"), default="TE")
    parser.add_argument("--polar-angle", type=float, default=0.0)
    parser.add_argument("--azimuthal-angle", type=float, default=0.0)
    parser.add_argument("--beam-width", type=float, default=2000.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="gmres")
    parser.add_argument("--solver-rtol", type=float, default=1e-4)
    parser.add_argument("--solver-restart", type=int, default=100)
    parser.add_argument("--solver-maxiter", type=int, default=1000)
    parser.add_argument(
        "--radial-lut-dr",
        type=float,
        default=0.0,
        help="Radial LUT spacing in length units; 0 enables auto delta(kr)=1e-2.",
    )
    parser.add_argument(
        "--force-general-initial-field",
        action="store_true",
        help="Force the general alpha-beta initial-field integration during near-field evaluation.",
    )
    parser.add_argument(
        "--compute-dtype", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--accum-dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument(
        "--cache-mode",
        choices=("off", "on", "both"),
        default="both",
        help="Profile solver with translation block cache off/on or both.",
    )
    parser.add_argument(
        "--preconditioner-mode",
        choices=("none", "grid_block", "both"),
        default="none",
        help="Profile solver with no preconditioner, grid-block preconditioner, or both.",
    )
    parser.add_argument(
        "--preconditioner-subdivisions",
        type=int,
        default=2,
        help="Grid subdivisions per axis for built-in grid_block preconditioner.",
    )
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/profiling"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    source_model = cast(Literal["planewave", "gaussian"], args.source_model)
    polarization = cast(Literal["TE", "TM"], args.polarization)
    compute_dtype_name = cast(Literal["complex64", "complex128"], args.compute_dtype)
    accum_dtype_name = cast(Literal["complex64", "complex128"], args.accum_dtype)

    root = _find_repo_root()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    positions, radii, n_particle = _load_geometry(
        root / "examples" / "sphere_parameters.txt", args.n_particles
    )

    source: pcl.PlaneWave | pcl.GaussianBeam
    if source_model == "planewave":
        source = pcl.PlaneWave(
            wavelength=float(args.wavelength),
            medium_n=float(args.n_medium) + 0j,
            polarization=polarization,
            polar_angle=float(args.polar_angle),
            azimuthal_angle=float(args.azimuthal_angle),
            amplitude=float(args.amplitude),
        )
    else:
        source = pcl.GaussianBeam(
            wavelength=float(args.wavelength),
            medium_n=float(args.n_medium) + 0j,
            polarization=polarization,
            polar_angle=float(args.polar_angle),
            azimuthal_angle=float(args.azimuthal_angle),
            amplitude=float(args.amplitude),
            beam_width=float(args.beam_width),
            focal_point=(0.0, 0.0, 0.0),
        )
    cfg = pcl.SimulationConfig(
        wavelength=float(args.wavelength),
        n_medium=float(args.n_medium) + 0j,
        lmax=int(args.lmax),
        source=source,
        polar_angles=pcl.core.uniform_polar_grid(int(args.n_beta)),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(int(args.n_alpha)),
        radial_lut_dr=float(args.radial_lut_dr),
        solver_method=cast(
            Literal["auto", "gmres", "bicgstab", "lgmres", "gcrotmk", "direct"],
            args.solver,
        ),
        solver_rtol=float(args.solver_rtol),
        solver_restart=int(args.solver_restart),
        solver_maxiter=int(args.solver_maxiter),
        compute_dtype=compute_dtype_name,
        accum_dtype=accum_dtype_name,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f"Preparing problem: N={positions.shape[0]}, lmax={cfg.lmax}")

    n_spheres = positions.shape[0]
    n_modes_l = n_modes(cfg.lmax)
    k = 2.0 * np.pi / float(cfg.wavelength) * float(np.real(cfg.n_medium))
    k0 = 2.0 * np.pi / float(cfg.wavelength)
    source_polar_angles, source_azimuthal_angles = cfg.source_angular_grids()
    farfield_polar_angles, farfield_azimuthal_angles = cfg.farfield_angular_grids()

    b = project_source_to_svwf(
        positions,
        cfg.lmax,
        source,
        polar_angles=source_polar_angles,
        azimuthal_angles=source_azimuthal_angles,
        dtype=np.dtype(cfg.compute_dtype),
    ).astype(np.dtype(cfg.accum_dtype), copy=False)
    rhs_input = b.reshape(n_spheres * n_modes_l)

    cache_flags: list[bool]
    if args.cache_mode == "off":
        cache_flags = [False]
    elif args.cache_mode == "on":
        cache_flags = [True]
    else:
        cache_flags = [False, True]

    preconditioner_kinds: list[Literal["none", "grid_block"]]
    if args.preconditioner_mode == "none":
        preconditioner_kinds = ["none"]
    elif args.preconditioner_mode == "grid_block":
        preconditioner_kinds = ["grid_block"]
    else:
        preconditioner_kinds = ["none", "grid_block"]

    cache_bytes_est = estimate_translation_cache_bytes(
        n_spheres,
        cfg.lmax,
        dtype=np.dtype(cfg.compute_dtype),
    )
    if not args.quiet and any(cache_flags):
        print(
            f"Estimated full W_ij cache payload: {cache_bytes_est / 1024**3:.2f} GiB (raw arrays only)"
        )

    solver_runs: list[dict[str, Any]] = []
    primary_solver_res = None
    primary_rhs = None

    for cache_on in cache_flags:
        prepared = prepare_matvec(
            lmax=cfg.lmax,
            k=k,
            particles=pcl.core.spheres_from_arrays(
                positions=positions,
                radii=radii,
                refractive_indices=n_particle,
            ),
            n_medium=cfg.n_medium,
            radial_lut_dr=cfg.radial_lut_dr,
            cache_translation_blocks=cache_on,
            operator_dtype=np.dtype(cfg.compute_dtype),
        )
        A_mv, rhs = make_prepared_A_and_rhs(prepared, rhs_input)
        n_unknowns = int(rhs.shape[0])
        solver_name = str(cfg.solver_method).lower()
        will_use_direct = solver_name == "direct" or (
            solver_name == "auto" and n_unknowns <= int(cfg.solver_direct_max_n)
        )
        for precond_kind in preconditioner_kinds:
            preconditioner = None
            preconditioner_build_s: float | None = None
            preconditioner_effective: str = precond_kind
            if precond_kind == "grid_block":
                if will_use_direct:
                    # Direct solves ignore preconditioners; record this explicitly.
                    preconditioner_effective = "ignored_for_direct"
                else:
                    t_pc0 = time.perf_counter()
                    preconditioner = make_grid_block_preconditioner(
                        prepared,
                        subdivisions=int(args.preconditioner_subdivisions),
                        cubic_bbox=True,
                        max_block_unknowns=None,
                        show_progress=not args.quiet,
                    )
                    preconditioner_build_s = float(time.perf_counter() - t_pc0)

            phase_name = f"solver_cache_{'on' if cache_on else 'off'}_pc_{preconditioner_effective}"
            if not args.quiet:
                print(f"Profiling phase: {phase_name}")
            solver_res, solver_summary = _profile_phase(
                phase=phase_name,
                out_dir=out_dir,
                top_n=args.top_n,
                fn=solve_linear_system,
                A_mv=A_mv,
                b=rhs,
                method=cfg.solver_method,
                A_dense=None,
                preconditioner=preconditioner,
                rtol=cfg.solver_rtol,
                atol=0.0,
                restart=cfg.solver_restart,
                maxiter=cfg.solver_maxiter,
                direct_max_n=cfg.solver_direct_max_n,
                dtype=np.dtype(cfg.compute_dtype),
                show_progress=not args.quiet,
            )
            solver_runs.append(
                {
                    "cache_translation_blocks": bool(cache_on),
                    "preconditioner": str(preconditioner_effective),
                    "preconditioner_build_s": preconditioner_build_s,
                    "solver_result": {
                        "method": str(solver_res.method),
                        "iterations": int(solver_res.iterations),
                        "info": int(solver_res.info),
                        "relative_residual": float(solver_res.relative_residual),
                        "residual_norm": float(solver_res.residual_norm),
                    },
                    "summary": solver_summary,
                }
            )
            if primary_solver_res is None:
                primary_solver_res = solver_res
                primary_rhs = rhs

    if primary_solver_res is None or primary_rhs is None:
        raise RuntimeError("No solver profile run was executed.")

    coeffs = primary_solver_res.x.reshape(n_spheres, n_modes_l)

    if not args.quiet:
        print("Profiling phase: farfield")
    farfield, farfield_summary = _profile_phase(
        phase="farfield",
        out_dir=out_dir,
        top_n=args.top_n,
        fn=compute_far_field_patterns,
        positions=positions,
        coeffs=coeffs,
        k=k,
        lmax=cfg.lmax,
        polar_angles=farfield_polar_angles,
        azimuthal_angles=farfield_azimuthal_angles,
        source=source,
        dtype=np.dtype(cfg.compute_dtype),
        show_progress=not args.quiet,
    )

    run = pcl.SimulationResult(
        config=cfg,
        particles=tuple(
            pcl.core.spheres_from_arrays(
                positions=positions,
                radii=radii,
                refractive_indices=n_particle,
            )
        ),
        k=k,
        k0=k0,
        coeffs=coeffs,
        rhs=rhs.reshape(n_spheres, n_modes_l),
        initial_coeffs=b,
        initial_coeffs_basis=None,
        coeffs_basis=None,
        solver_result=primary_solver_res,
        solver_result_basis=None,
        farfield=farfield,
        farfield_basis=None,
        power=None,
        power_basis=None,
        cross_sections=None,
        cross_sections_basis=None,
        unpolarized=None,
        decomposition_forward=None,
        decomposition_backward=None,
        decomposition_forward_basis=None,
        decomposition_backward_basis=None,
        polarization_jones=source.jones_coefficients(),
    )

    if not args.quiet:
        print("Profiling phase: nearfield")
    near_field, nearfield_summary = _profile_phase(
        phase="nearfield",
        out_dir=out_dir,
        top_n=args.top_n,
        fn=pcl.compute_near_field_slice,
        run=run,
        axis_0_min=-4000.0,
        axis_0_max=4000.0,
        axis_1_min=-3000.0,
        axis_1_max=5000.0,
        dx=float(args.dx),
        plane="y",
        plane_value=0.0,
        show_progress=not args.quiet,
        force_general_initial_field=bool(args.force_general_initial_field),
        center_pixel_policy="interpolate",
    )

    phases: list[dict[str, Any]] = [sr["summary"] for sr in solver_runs]
    phases.extend([farfield_summary, nearfield_summary])
    summary = {
        "config": {
            "n_particles": int(args.n_particles),
            "lmax": int(args.lmax),
            "wavelength": float(args.wavelength),
            "n_medium": float(args.n_medium),
            "n_beta": int(args.n_beta),
            "n_alpha": int(args.n_alpha),
            "dx": float(args.dx),
            "source_model": source_model,
            "polarization": polarization,
            "polar_angle": float(args.polar_angle),
            "azimuthal_angle": float(args.azimuthal_angle),
            "beam_width": float(args.beam_width),
            "amplitude": float(args.amplitude),
            "solver": str(args.solver),
            "solver_rtol": float(args.solver_rtol),
            "solver_restart": int(args.solver_restart),
            "solver_maxiter": int(args.solver_maxiter),
            "radial_lut_dr": float(args.radial_lut_dr),
            "compute_dtype": compute_dtype_name,
            "accum_dtype": accum_dtype_name,
            "cache_mode": str(args.cache_mode),
            "preconditioner_mode": str(args.preconditioner_mode),
            "preconditioner_subdivisions": int(args.preconditioner_subdivisions),
        },
        "estimated_translation_cache_bytes": int(cache_bytes_est),
        "solver_runs": solver_runs,
        "nearfield_grid_shape": [int(near_field.axis_0.shape[0]), int(near_field.axis_0.shape[1])],
        "phases": phases,
    }

    summary_path = out_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote profiling reports to {out_dir}")
    print(f"Machine-readable summary: {summary_path}")
    print("Phase wall times [s]:")
    for ph in phases:
        print(f"  - {ph['phase']}: {ph['wall_time_s']:.3f}")


if __name__ == "__main__":
    main()
