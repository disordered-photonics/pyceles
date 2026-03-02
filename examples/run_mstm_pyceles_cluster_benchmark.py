"""Benchmark pyceles against MSTM v4.0 for tilted Gaussian-beam clusters.

This is the single maintained pyceles-vs-MSTM comparison script. It can run
MSTM and pyceles for the same configuration and produces:
- efficiency comparisons (Qext, Qabs, Qsca, up/down hemispheres),
- near-field component maps (Re Ex/Ey/Ez and Re Hx/Hy/Hz) for TE/TM,
- S11 hemisphere maps from MSTM scattering_map_model=1 and pyceles overlays,
- S11 incident-plane semilogy curves from MSTM scattering_map_model=0 and pyceles overlays,
- RMSE/rel-RMSE statistics.

Conventions used here:
- fixed channel mapping: pyceles te -> MSTM par, pyceles tm -> MSTM perp,
- fixed near-field phase/sign convention: MSTM par channel multiplied by -1,
- fixed far-field scales (no fitting):
  scattering_map_model=1 uses 100*pi^2, scattering_map_model=0 uses 100*pi^3.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import pyceles as pcl
from pyceles.io import (
    far_field_intensity,
    plot_farfield_hemispheres,
    plot_field_component,
    plot_spheres,
)

SolverMethod = Literal["auto", "gmres", "bicgstab", "lgmres", "gcrotmk", "direct"]
ComplexDType = Literal["complex64", "complex128"]
# Fixed channel mapping used throughout benchmark diagnostics.
MSTM_POLARIZATION_MAPPING: dict[str, str] = {"te": "par", "tm": "perp"}
MSTM_NEARFIELD_PHASE_CORRECTION: dict[str, complex] = {"par": (-1.0 + 0.0j), "perp": (1.0 + 0.0j)}
S11_SCALE_MODEL1 = float(100.0 * np.pi * np.pi)
S11_SCALE_MODEL0 = float(100.0 * np.pi * np.pi * np.pi)


@dataclass(frozen=True)
class NearFieldGridConfig:
    # Physical geometry units (same units as sphere parameter file, here nm).
    # MSTM expects near-field borders in its internal (post-scaling) coordinate system.
    # We convert these physical values via x_dimless = k0 * x_physical when writing input.
    minimum_border: tuple[float, float, float] = (-4000.0, 0.0, -3000.0)
    maximum_border: tuple[float, float, float] = (4000.0, 0.0, 5000.0)
    step_size: float = 40.0


@dataclass(frozen=True)
class BenchmarkConfig:
    mstm_exe: Path | None = None
    sphere_parameters: Path = Path("examples/sphere_parameters.txt")
    workdir: Path = Path("outputs/mstm")
    outdir: Path = Path("outputs/mstm_diagnostics")
    n_particles: int = 500
    wavelength: float = 550.0
    beam_width: float = 1700.0
    polar_angle: float = 0.43
    azimuthal_angle: float = 0.37
    lmax: int = 3
    n_beta: int = 1801
    n_alpha: int = 720
    py_solver_method: SolverMethod = "gmres"
    py_solver_rtol: float = 1.0e-5
    py_solver_maxiter: int = 1500
    py_solver_restart: int = 100
    py_compute_dtype: ComplexDType = "complex128"
    py_accum_dtype: ComplexDType = "complex128"
    mstm_mie_epsilon: float = -3.0
    mstm_solution_epsilon: float = 1.0e-5
    mstm_max_iterations: int = 10000
    mstm_scattering_map_dimension: int = 121
    # MSTM v4 behavior: near-field model 1=total field, 2=scattered field.
    mstm_near_field_model: int = 1
    near_field_grid: NearFieldGridConfig = NearFieldGridConfig()
    output_prefix: str = "mstm_pyceles_tilted_benchmark"
    run_mstm: bool = True


@dataclass(frozen=True)
class PlotMeta:
    plane: str
    plane_value: float
    axis_h: np.ndarray
    axis_v: np.ndarray
    axis_h_label: str
    axis_v_label: str


def _fmt_d(x: float) -> str:
    return f"{float(x):.10e}".replace("e", "d", 1)


def _float_d(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "E"))


_NUMERIC_TOKEN_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?")


def _extract_float_tokens(line: str) -> list[float]:
    return [_float_d(tok) for tok in _NUMERIC_TOKEN_RE.findall(line)]


def _load_geometry(path: Path, n_particles: int) -> np.ndarray:
    # Geometry is benchmarked as-is from sphere_parameters.txt.
    # No z shifting/clearance manipulation is applied here.
    data = np.loadtxt(path)
    return np.array(data[:n_particles], copy=True)


def _write_geometry_snapshot(path: Path, sphere_data: np.ndarray) -> None:
    header = "radius x y z n_real n_imag"
    np.savetxt(path, sphere_data, fmt="%.12g", header=header, comments="")


def _write_mstm_input(
    cfg: BenchmarkConfig,
    inp_path: Path,
    output_filename: str,
    nearfield_output_filename: str,
    sphere_data: np.ndarray,
    *,
    scattering_map_model: int = 1,
    scattering_map_dimension: int | None = None,
    scattering_map_increment_deg: float | None = None,
    calculate_near_field: bool = True,
) -> None:
    k0 = 2.0 * np.pi / float(cfg.wavelength)
    beam_const = 1.0 / (k0 * float(cfg.beam_width))
    alpha_deg = float(np.degrees(cfg.azimuthal_angle))
    beta_deg = float(np.degrees(cfg.polar_angle))
    nf = cfg.near_field_grid
    nf_min_dimless = tuple(float(v) * float(k0) for v in nf.minimum_border)
    nf_max_dimless = tuple(float(v) * float(k0) for v in nf.maximum_border)
    nf_step_dimless = float(nf.step_size) * float(k0)
    map_model = int(scattering_map_model)
    if map_model == 1:
        # MSTM map-model 1 uses a square sampling lattice in (kx, ky) per
        # hemisphere, clipped to the unit disk. `scattering_map_dimension` is
        # the lattice side count (not the total number of polar-angle bins).
        map_dimension = int(
            cfg.mstm_scattering_map_dimension
            if scattering_map_dimension is None
            else scattering_map_dimension
        )
        if map_dimension < 3:
            raise ValueError("scattering_map_dimension must be >= 3 for map model 1.")
    elif map_model == 0:
        if scattering_map_increment_deg is None:
            if int(cfg.n_beta) > 1:
                map_increment = 180.0 / float(int(cfg.n_beta) - 1)
            else:
                map_increment = 1.0
        else:
            map_increment = float(scattering_map_increment_deg)
        if map_increment <= 0.0:
            raise ValueError("scattering_map_increment_deg must be > 0 for map model 0.")
    else:
        raise ValueError("scattering_map_model must be 0 or 1.")

    lines: list[str] = [
        "output_file",
        output_filename,
        "append_output_file",
        "f",
        "print_sphere_data",
        "t",
        "number_spheres",
        str(int(cfg.n_particles)),
        "number_plane_boundaries",
        "0",
        "sphere_data",
    ]
    for row in sphere_data:
        radius, x, y, z, n_real, n_imag = [float(v) for v in row]
        lines.append(
            f"{_fmt_d(x)},{_fmt_d(y)},{_fmt_d(z)},{_fmt_d(radius)},({_fmt_d(n_real)},{_fmt_d(n_imag)})"
        )

    lines += [
        "end_of_sphere_data",
        "length_scale_factor",
        _fmt_d(k0),
        "ref_index_scale_factor",
        "(1.0d0,0.0d0)",
        "random_orientation",
        "f",
        "incident_alpha_deg",
        _fmt_d(alpha_deg),
        "incident_beta_deg",
        _fmt_d(beta_deg),
        "gaussian_beam_constant",
        _fmt_d(beam_const),
        "gaussian_beam_focal_point",
        "0.0d0,0.0d0,0.0d0",
        "max_iterations",
        str(int(cfg.mstm_max_iterations)),
        "solution_epsilon",
        _fmt_d(cfg.mstm_solution_epsilon),
        "mie_epsilon",
        _fmt_d(cfg.mstm_mie_epsilon),
        "calculate_up_down_scattering",
        "t",
        "calculate_scattering_matrix",
        "t",
        "scattering_map_model",
        str(map_model),
        # Keep raw fixed-orientation S11 outputs (normalize_s11=false).
        "normalize_s11",
        "f",
        # Keep scattering-map angles in the target/lab frame (not incident-aligned frame).
        # This matches pyceles far-field grids directly (beta from +z axis) without
        # requiring an additional rotation layer in postprocessing.
        "incident_frame",
        "f",
    ]
    if map_model == 1:
        lines += [
            "scattering_map_dimension",
            str(int(map_dimension)),
        ]
    else:
        lines += [
            "scattering_map_increment",
            _fmt_d(map_increment),
        ]

    if bool(calculate_near_field):
        lines += [
            "calculate_near_field",
            "t",
            "near_field_calculation_model",
            str(int(cfg.mstm_near_field_model)),
            "near_field_output_file",
            nearfield_output_filename,
            "near_field_minimum_border",
            f"{_fmt_d(nf_min_dimless[0])},{_fmt_d(nf_min_dimless[1])},{_fmt_d(nf_min_dimless[2])}",
            "near_field_maximum_border",
            f"{_fmt_d(nf_max_dimless[0])},{_fmt_d(nf_max_dimless[1])},{_fmt_d(nf_max_dimless[2])}",
            "near_field_step_size",
            _fmt_d(nf_step_dimless),
        ]
    else:
        lines += ["calculate_near_field", "f"]
    lines += ["end_of_options"]
    inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_block(output_text: str, pattern: str) -> list[float]:
    m = re.search(pattern, output_text, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Could not find output block matching pattern: {pattern!r}")
    return [_float_d(tok) for tok in m.group(1).split()]


def _parse_mstm_scattering_map(text: str) -> dict[str, dict[str, np.ndarray]]:
    lines = text.splitlines()

    def _parse_section(start_label: str, stop_label: str | None) -> list[list[float]]:
        start = None
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith(start_label.lower()):
                start = i
                break
        if start is None:
            return []

        i = start + 1
        while i < len(lines):
            if "kx" in lines[i].lower() and "ky" in lines[i].lower():
                i += 1
                break
            i += 1

        rows: list[list[float]] = []
        while i < len(lines):
            ln = lines[i].strip()
            if not ln:
                i += 1
                continue
            if stop_label is not None and ln.lower().startswith(stop_label.lower()):
                break
            toks = _extract_float_tokens(ln)
            if len(toks) >= 18:
                rows.append(toks[:18])
                i += 1
                continue
            break
        return rows

    bwd_rows = _parse_section("backward hemisphere scattering", "forward hemisphere scattering")
    fwd_rows = _parse_section("forward hemisphere scattering", None)

    def _to_arrays(rows: list[list[float]]) -> dict[str, np.ndarray]:
        if not rows:
            return {
                "kx": np.zeros((0,), dtype=float),
                "ky": np.zeros((0,), dtype=float),
                "s11": np.zeros((0,), dtype=float),
            }
        arr = np.asarray(rows, dtype=float)
        return {
            "kx": arr[:, 0],
            "ky": arr[:, 1],
            "s11": arr[:, 2],
        }

    return {
        "backward": _to_arrays(bwd_rows),
        "forward": _to_arrays(fwd_rows),
    }


def _parse_mstm_scattering_curve_incident_plane(text: str) -> dict[str, np.ndarray]:
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("scattering matrix in incident plane"):
            start = i
            break
    if start is None:
        return {
            "theta_deg": np.zeros((0,), dtype=float),
            "s11": np.zeros((0,), dtype=float),
            "matrix": np.zeros((0, 16), dtype=float),
        }

    i = start + 1
    while i < len(lines):
        ln = lines[i].strip().lower()
        if ln.startswith("theta") and "11" in ln:
            i += 1
            break
        i += 1

    rows: list[list[float]] = []
    while i < len(lines):
        toks = _extract_float_tokens(lines[i])
        if len(toks) >= 17:
            rows.append(toks[:17])
            i += 1
            continue
        break

    if not rows:
        return {
            "theta_deg": np.zeros((0,), dtype=float),
            "s11": np.zeros((0,), dtype=float),
            "matrix": np.zeros((0, 16), dtype=float),
        }

    arr = np.asarray(rows, dtype=float)
    return {
        "theta_deg": arr[:, 0],
        "s11": arr[:, 1],
        "matrix": arr[:, 1:17],
    }


def _parse_mstm_output(output_path: Path) -> dict[str, Any]:
    text = output_path.read_text(encoding="utf-8", errors="ignore")

    acs_vals = _parse_block(
        text,
        r"volume cluster radius, area mean sphere radius, circumscribing radius, cross section radius\s*\n([^\n]+)\n",
    )
    if len(acs_vals) < 4:
        raise RuntimeError("Unexpected cross section radius block shape.")
    acs = float(acs_vals[3])

    q_tot = _parse_block(
        text,
        r"total extinction, absorption, scattering efficiencies \(unpol, par, perp incidence\)\s*\n([^\n]+)\n",
    )
    if len(q_tot) < 9:
        raise RuntimeError("Unexpected total efficiencies block shape.")

    q_hemi = _parse_block(
        text,
        r"down and up hemispherical scattering efficiencies \(unpol, par, perp\)\s*\n([^\n]+)\n",
    )
    if len(q_hemi) < 6:
        raise RuntimeError("Unexpected hemispherical scattering block shape.")

    solver_in = _parse_block(
        text,
        r"max_iterations,solution_epsilon,\s*mie_epsilon\s*\n([^\n]+)\n",
    )
    if len(solver_in) < 3:
        raise RuntimeError("Unexpected solver-input block shape.")

    mie_block = _parse_block(
        text,
        r"maximum Mie order, number of equations:\s*\n([^\n]+)\n",
    )
    if len(mie_block) < 2:
        raise RuntimeError("Unexpected maximum-Mie-order block shape.")

    solver_out = _parse_block(
        text,
        r"number iterations, error, solution time\s*\n([^\n]+)\n",
    )
    if len(solver_out) < 3:
        raise RuntimeError("Unexpected solver-result block shape.")

    qext_u, qabs_u, qsca_u = q_tot[0:3]
    qext_p, qabs_p, qsca_p = q_tot[3:6]
    qext_s, qabs_s, qsca_s = q_tot[6:9]

    qsca_down_u, qsca_up_u = q_hemi[0:2]
    qsca_down_p, qsca_up_p = q_hemi[2:4]
    qsca_down_s, qsca_up_s = q_hemi[4:6]

    scattering_map = _parse_mstm_scattering_map(text)
    scattering_curve_incident = _parse_mstm_scattering_curve_incident_plane(text)
    area = float(np.pi * acs**2)
    return {
        "cross_section_radius_dimless": acs,
        "cross_section_area_dimless": area,
        "max_iterations_input": int(round(solver_in[0])),
        "solution_epsilon_input": float(solver_in[1]),
        "mie_epsilon_input": float(solver_in[2]),
        "max_mie_order_used": int(round(mie_block[0])),
        "number_equations": int(round(mie_block[1])),
        "solver_iterations": int(round(solver_out[0])),
        "solver_residual": float(solver_out[1]),
        "solver_time_s": float(solver_out[2]),
        "Qext_unpol": float(qext_u),
        "Qabs_unpol": float(qabs_u),
        "Qsca_unpol": float(qsca_u),
        "Qext_par": float(qext_p),
        "Qabs_par": float(qabs_p),
        "Qsca_par": float(qsca_p),
        "Qext_perp": float(qext_s),
        "Qabs_perp": float(qabs_s),
        "Qsca_perp": float(qsca_s),
        "Qsca_down_unpol": float(qsca_down_u),
        "Qsca_up_unpol": float(qsca_up_u),
        "Qsca_down_par": float(qsca_down_p),
        "Qsca_up_par": float(qsca_up_p),
        "Qsca_down_perp": float(qsca_down_s),
        "Qsca_up_perp": float(qsca_up_s),
        "scattering_map": scattering_map,
        "scattering_curve_incident": scattering_curve_incident,
    }


def _parse_mstm_nearfield_output(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    while idx < len(lines) and "run number" not in lines[idx].lower():
        idx += 1
    if idx >= len(lines):
        raise RuntimeError(f"Could not find near-field run header in {path}.")
    idx += 1

    def _next_nonempty(i: int) -> int:
        while i < len(lines) and not lines[i].strip():
            i += 1
        return i

    idx = _next_nonempty(idx)
    run_number = int(round(_extract_float_tokens(lines[idx])[0]))
    idx += 1
    idx = _next_nonempty(idx)
    n_intersecting = int(round(_extract_float_tokens(lines[idx])[0]))
    idx += 1
    for _ in range(n_intersecting):
        idx = _next_nonempty(idx)
        idx += 1

    idx = _next_nonempty(idx)
    n_boundaries = int(round(_extract_float_tokens(lines[idx])[0]))
    idx += 1
    for _ in range(n_boundaries):
        idx = _next_nonempty(idx)
        idx += 1

    idx = _next_nonempty(idx)
    grid_min = np.asarray(_extract_float_tokens(lines[idx])[:3], dtype=float)
    idx += 1
    idx = _next_nonempty(idx)
    grid_max = np.asarray(_extract_float_tokens(lines[idx])[:3], dtype=float)
    idx += 1
    idx = _next_nonempty(idx)
    dims_tokens = _extract_float_tokens(lines[idx])
    if len(dims_tokens) < 3:
        raise RuntimeError("Malformed near-field grid dimensions line.")
    dims = np.asarray([int(round(v)) for v in dims_tokens[:3]], dtype=int)
    idx += 1

    npts = int(np.prod(dims))
    coords = np.zeros((npts, 3), dtype=float)
    e_pol1 = np.zeros((npts, 3), dtype=np.complex128)
    h_pol1 = np.zeros((npts, 3), dtype=np.complex128)
    e_pol2 = np.zeros((npts, 3), dtype=np.complex128)
    h_pol2 = np.zeros((npts, 3), dtype=np.complex128)

    for i in range(npts):
        idx = _next_nonempty(idx)
        toks = _extract_float_tokens(lines[idx])
        idx += 1
        if len(toks) < 27:
            raise RuntimeError(
                f"Malformed near-field row {i}: expected >=27 floats, got {len(toks)}."
            )
        vals = np.asarray(toks[:27], dtype=float)
        coords[i, :] = vals[0:3]

        def _as_complex(v: np.ndarray) -> np.ndarray:
            return v[0::2] + 1j * v[1::2]

        e_pol1[i, :] = _as_complex(vals[3:9])
        h_pol1[i, :] = _as_complex(vals[9:15])
        e_pol2[i, :] = _as_complex(vals[15:21])
        h_pol2[i, :] = _as_complex(vals[21:27])

    return {
        "run_number": run_number,
        "n_intersecting_spheres": n_intersecting,
        "n_boundaries": n_boundaries,
        "grid_min": grid_min,
        "grid_max": grid_max,
        "grid_dims": dims,
        "coords": coords,
        "E_par": e_pol1,
        "H_par": h_pol1,
        "E_perp": e_pol2,
        "H_perp": h_pol2,
    }


def _mstm_dimless_to_physical(points_dimless: np.ndarray, wavelength: float) -> np.ndarray:
    """Convert MSTM near-field coordinates (dimensionless) to physical length units."""
    k0 = 2.0 * np.pi / float(wavelength)
    return np.asarray(points_dimless, dtype=float) / float(k0)


def _pyceles_run(
    cfg: BenchmarkConfig, sphere_data: np.ndarray, nearfield_points: np.ndarray
) -> dict[str, Any]:
    positions = np.asarray(sphere_data[:, 1:4], dtype=float)
    radii = np.asarray(sphere_data[:, 0], dtype=float)
    n_particle = np.asarray(sphere_data[:, 4] + 1j * sphere_data[:, 5], dtype=np.complex128)

    source = pcl.GaussianBeam(
        wavelength=float(cfg.wavelength),
        medium_n=1.0 + 0j,
        polarization="TE",
        polar_angle=float(cfg.polar_angle),
        azimuthal_angle=float(cfg.azimuthal_angle),
        amplitude=1.0,
        beam_width=float(cfg.beam_width),
        focal_point=(0.0, 0.0, 0.0),
    )

    sim_cfg = pcl.SimulationConfig(
        wavelength=float(cfg.wavelength),
        n_medium=1.0 + 0j,
        lmax=int(cfg.lmax),
        source=source,
        polar_angles=pcl.core.uniform_polar_grid(int(cfg.n_beta)),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(int(cfg.n_alpha)),
        solver_method=cfg.py_solver_method,
        solver_rtol=float(cfg.py_solver_rtol),
        solver_maxiter=int(cfg.py_solver_maxiter),
        solver_restart=int(cfg.py_solver_restart),
        solver_direct_max_n=20_000,
        compute_dtype=cfg.py_compute_dtype,
        accum_dtype=cfg.py_accum_dtype,
        verbose=False,
    )

    sim = pcl.Simulation(sim_cfg, positions=positions, radii=radii, n_particle=n_particle)
    solved = sim.solve_sources(
        {
            "te": source.with_polarization("TE"),
            "tm": source.with_polarization("TM"),
        }
    )
    multi = sim.postprocess_sources(solved)
    run_basis = {"te": multi["te"], "tm": multi["tm"]}

    eff_basis: dict[str, dict[str, float]] = {}
    for ch in ("te", "tm"):
        run_ch = run_basis[ch]
        if (
            run_ch.power is None
            or run_ch.decomposition_forward is None
            or run_ch.decomposition_backward is None
        ):
            raise RuntimeError(
                f"Expected finite-beam power diagnostics for channel '{ch}', but they are missing."
            )
        p0 = float(run_ch.power["P_initial"])
        s_up = float(run_ch.decomposition_forward["P_scattered"] / p0)
        s_down = float(run_ch.decomposition_backward["P_scattered"] / p0)
        t_frac = float(run_ch.power["T"])
        r_frac = float(run_ch.power["R"])
        q_abs = float(1.0 - t_frac - r_frac)
        q_sca = float(s_up + s_down)
        eff_basis[ch] = {
            "Qabs": q_abs,
            "Qsca_up": s_up,
            "Qsca_down": s_down,
            "Qsca": q_sca,
            "Qext": float(q_abs + q_sca),
            "P_initial": p0,
        }

    ff_te = run_basis["te"].farfield.scattered_te
    I_te = far_field_intensity(
        run_basis["te"].farfield.scattered_te, run_basis["te"].farfield.scattered_tm
    )
    I_tm = far_field_intensity(
        run_basis["tm"].farfield.scattered_te, run_basis["tm"].farfield.scattered_tm
    )
    I_unpol = 0.5 * (I_te + I_tm)
    alpha = np.asarray(ff_te["alpha"], dtype=float)
    beta = np.asarray(ff_te["beta"], dtype=float)
    kx = np.cos(alpha)[:, None] * np.sin(beta)[None, :]
    ky = np.sin(alpha)[:, None] * np.sin(beta)[None, :]
    kz = np.cos(beta)[None, :] * np.ones((alpha.size, 1), dtype=float)

    nf_te = pcl.compute_near_field(
        run_basis["te"], points=nearfield_points, channel="mixed", show_progress=False
    )
    nf_tm = pcl.compute_near_field(
        run_basis["tm"], points=nearfield_points, channel="mixed", show_progress=False
    )
    if int(cfg.mstm_near_field_model) == 1:
        e_te = np.asarray(nf_te.E_total, dtype=np.complex128)
        h_te = np.asarray(nf_te.H_total, dtype=np.complex128)
        e_tm = np.asarray(nf_tm.E_total, dtype=np.complex128)
        h_tm = np.asarray(nf_tm.H_total, dtype=np.complex128)
        nearfield_kind = "total"
    elif int(cfg.mstm_near_field_model) == 2:
        e_te = np.asarray(nf_te.E_scattered, dtype=np.complex128)
        h_te = np.asarray(nf_te.H_scattered, dtype=np.complex128)
        e_tm = np.asarray(nf_tm.E_scattered, dtype=np.complex128)
        h_tm = np.asarray(nf_tm.H_scattered, dtype=np.complex128)
        nearfield_kind = "scattered"
    else:
        raise ValueError(
            f"Unsupported MSTM near-field model {cfg.mstm_near_field_model}; expected 1 or 2."
        )

    iter_val = np.asarray(multi.solver_result.iterations)
    rr_val = np.asarray(multi.solver_result.relative_residual)
    if iter_val.ndim == 0:
        iterations_out: int | list[int] = int(iter_val)
    else:
        iterations_out = [int(v) for v in iter_val.tolist()]
    if rr_val.ndim == 0:
        rr_out: float | list[float] = float(rr_val)
    else:
        rr_out = [float(v) for v in rr_val.tolist()]

    return {
        "solver": {
            "method": str(multi.solver_result.method),
            "iterations": iterations_out,
            "relative_residual": rr_out,
            "rhs_count": int(multi.solver_result.rhs_count),
        },
        "efficiencies_basis": eff_basis,
        "farfield_map": {
            "alpha": alpha,
            "beta": beta,
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "I_unpolarized": I_unpol,
        },
        "nearfield_basis": {
            "kind": nearfield_kind,
            "E_te": e_te,
            "H_te": h_te,
            "E_tm": e_tm,
            "H_tm": h_tm,
            "inside_te": np.asarray(nf_te.inside_mask, dtype=bool),
            "inside_tm": np.asarray(nf_tm.inside_mask, dtype=bool),
        },
    }


def _eff_vector(d: dict[str, float]) -> np.ndarray:
    return np.asarray([d["Qabs"], d["Qsca_down"], d["Qsca_up"], d["Qsca"], d["Qext"]], dtype=float)


def _eff_compare_fixed_mapping(
    py_eff: dict[str, dict[str, float]],
    mstm: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    mstm_eff = {
        "par": {
            "Qabs": float(mstm["Qabs_par"]),
            "Qsca_down": float(mstm["Qsca_down_par"]),
            "Qsca_up": float(mstm["Qsca_up_par"]),
            "Qsca": float(mstm["Qsca_par"]),
            "Qext": float(mstm["Qext_par"]),
        },
        "perp": {
            "Qabs": float(mstm["Qabs_perp"]),
            "Qsca_down": float(mstm["Qsca_down_perp"]),
            "Qsca_up": float(mstm["Qsca_up_perp"]),
            "Qsca": float(mstm["Qsca_perp"]),
            "Qext": float(mstm["Qext_perp"]),
        },
    }

    deltas: dict[str, dict[str, Any]] = {}
    all_py = []
    all_mstm = []
    for py_ch, mstm_ch in mapping.items():
        v_py = _eff_vector(py_eff[py_ch])
        v_ms = _eff_vector(mstm_eff[mstm_ch])
        dv = v_py - v_ms
        deltas[py_ch] = {
            "mstm_channel": mstm_ch,
            "dQabs": float(dv[0]),
            "dQsca_down": float(dv[1]),
            "dQsca_up": float(dv[2]),
            "dQsca": float(dv[3]),
            "dQext": float(dv[4]),
            "max_abs_delta": float(np.max(np.abs(dv))),
        }
        all_py.append(v_py)
        all_mstm.append(v_ms)
    py_cat = np.concatenate(all_py)
    ms_cat = np.concatenate(all_mstm)
    rmse = float(np.sqrt(np.mean((py_cat - ms_cat) ** 2)))
    rel_rmse = float(rmse / np.sqrt(np.mean(ms_cat**2)))
    return {
        "mapping": dict(mapping),
        "per_channel": deltas,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
    }


def _rel_rmse_complex(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=np.complex128)
    bb = np.asarray(b, dtype=np.complex128)
    if mask is not None:
        aa = aa[mask]
        bb = bb[mask]
    num = float(np.linalg.norm((aa - bb).ravel()))
    den = float(np.linalg.norm(bb.ravel()))
    if den == 0.0:
        return float("nan")
    return num / den


def _compare_nearfield(
    chosen_mapping: dict[str, str],
    py_nf: dict[str, np.ndarray],
    mstm_nf: dict[str, np.ndarray],
    outside_mask: np.ndarray,
) -> dict[str, Any]:
    mstm_by_pol = {
        # Fixed TE convention alignment:
        # MSTM "par" channel is multiplied by -1 globally.
        "par": {
            "E": MSTM_NEARFIELD_PHASE_CORRECTION["par"]
            * np.asarray(mstm_nf["E_par"], dtype=np.complex128),
            "H": MSTM_NEARFIELD_PHASE_CORRECTION["par"]
            * np.asarray(mstm_nf["H_par"], dtype=np.complex128),
        },
        "perp": {
            "E": MSTM_NEARFIELD_PHASE_CORRECTION["perp"]
            * np.asarray(mstm_nf["E_perp"], dtype=np.complex128),
            "H": MSTM_NEARFIELD_PHASE_CORRECTION["perp"]
            * np.asarray(mstm_nf["H_perp"], dtype=np.complex128),
        },
    }
    py_by_pol = {
        "te": {
            "E": np.asarray(py_nf["E_te"], dtype=np.complex128),
            "H": np.asarray(py_nf["H_te"], dtype=np.complex128),
        },
        "tm": {
            "E": np.asarray(py_nf["E_tm"], dtype=np.complex128),
            "H": np.asarray(py_nf["H_tm"], dtype=np.complex128),
        },
    }

    out: dict[str, Any] = {"per_channel": {}}
    rels_all: list[float] = []
    rels_out: list[float] = []
    for py_ch, mstm_ch in chosen_mapping.items():
        e_py = py_by_pol[py_ch]["E"]
        h_py = py_by_pol[py_ch]["H"]
        e_ms = mstm_by_pol[mstm_ch]["E"]
        h_ms = mstm_by_pol[mstm_ch]["H"]
        e_all = _rel_rmse_complex(e_py, e_ms, mask=None)
        h_all = _rel_rmse_complex(h_py, h_ms, mask=None)
        e_out = _rel_rmse_complex(e_py, e_ms, mask=outside_mask)
        h_out = _rel_rmse_complex(h_py, h_ms, mask=outside_mask)
        out["per_channel"][py_ch] = {
            "mstm_channel": mstm_ch,
            "applied_mstm_phase_real": float(np.real(MSTM_NEARFIELD_PHASE_CORRECTION[mstm_ch])),
            "applied_mstm_phase_imag": float(np.imag(MSTM_NEARFIELD_PHASE_CORRECTION[mstm_ch])),
            "E_rel_rmse_all": float(e_all),
            "H_rel_rmse_all": float(h_all),
            "E_rel_rmse_outside": float(e_out),
            "H_rel_rmse_outside": float(h_out),
        }
        rels_all.extend([e_all, h_all])
        rels_out.extend([e_out, h_out])
    out["mean_rel_rmse_all"] = float(np.nanmean(rels_all))
    out["mean_rel_rmse_outside"] = float(np.nanmean(rels_out))
    out["n_points_all"] = int(py_nf["E_te"].shape[0])
    out["n_points_outside"] = int(np.sum(outside_mask))
    return out


def _sample_py_intensity_at_kxy(
    kx_py: np.ndarray,
    ky_py: np.ndarray,
    kz_py: np.ndarray,
    i_py: np.ndarray,
    kx_targets: np.ndarray,
    ky_targets: np.ndarray,
    *,
    hemisphere: str,
) -> np.ndarray:
    kx = np.asarray(kx_py, dtype=float).ravel()
    ky = np.asarray(ky_py, dtype=float).ravel()
    kz = np.asarray(kz_py, dtype=float).ravel()
    inten = np.asarray(i_py, dtype=float).ravel()
    if hemisphere == "forward":
        mask = kz >= 0.0
    elif hemisphere == "backward":
        mask = kz <= 0.0
    else:
        raise ValueError("hemisphere must be 'forward' or 'backward'.")
    pts = np.column_stack([kx[mask], ky[mask]])
    tree = cKDTree(pts)
    _, idx = tree.query(np.column_stack([kx_targets, ky_targets]), k=1)
    return inten[mask][idx]


def _real_metrics(model_values: np.ndarray, ref_values: np.ndarray) -> dict[str, float]:
    m = np.asarray(model_values, dtype=float).reshape(-1)
    r = np.asarray(ref_values, dtype=float).reshape(-1)
    if m.size == 0 or r.size == 0 or m.size != r.size:
        return {
            "n": int(m.size),
            "rmse": float("nan"),
            "rel_rmse": float("nan"),
            "pearson_r": float("nan"),
        }
    rmse = float(np.sqrt(np.mean((m - r) ** 2)))
    den = float(np.sqrt(np.mean(r**2)))
    rel_rmse = float(rmse / den) if den > 0.0 else float("nan")
    if m.size >= 2 and np.std(m) > 0 and np.std(r) > 0:
        pearson = float(np.corrcoef(m, r)[0, 1])
    else:
        pearson = float("nan")
    return {"n": int(m.size), "rmse": rmse, "rel_rmse": rel_rmse, "pearson_r": pearson}


def _compare_farfield_s11(mstm_map: dict[str, Any], py_map: dict[str, Any]) -> dict[str, Any]:
    kx_py = np.asarray(py_map["kx"], dtype=float)
    ky_py = np.asarray(py_map["ky"], dtype=float)
    kz_py = np.asarray(py_map["kz"], dtype=float)
    i_py = np.asarray(py_map["I_unpolarized"], dtype=float)

    out: dict[str, Any] = {}
    combined_model: list[np.ndarray] = []
    combined_ref: list[np.ndarray] = []
    for hemi in ("forward", "backward"):
        kx_t = np.asarray(mstm_map[hemi]["kx"], dtype=float)
        ky_t = np.asarray(mstm_map[hemi]["ky"], dtype=float)
        s11 = np.asarray(mstm_map[hemi]["s11"], dtype=float)
        if kx_t.size == 0:
            out[hemi] = {
                "n": 0,
                "rmse": float("nan"),
                "rel_rmse": float("nan"),
                "pearson_r": float("nan"),
            }
            continue
        py_samples = _sample_py_intensity_at_kxy(
            kx_py, ky_py, kz_py, i_py, kx_t, ky_t, hemisphere=hemi
        )
        mstm_rescaled = s11 / S11_SCALE_MODEL1
        metrics = _real_metrics(py_samples, mstm_rescaled)
        out[hemi] = metrics
        combined_model.append(py_samples)
        combined_ref.append(mstm_rescaled)

    if combined_model:
        comb_m = np.concatenate(combined_model)
        comb_r = np.concatenate(combined_ref)
        out["combined"] = _real_metrics(comb_m, comb_r)
    else:
        out["combined"] = {
            "n": 0,
            "rmse": float("nan"),
            "rel_rmse": float("nan"),
            "pearson_r": float("nan"),
        }
    return out


def _grid_index_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(values)
    idx = np.searchsorted(uniq, values)
    return uniq, idx


def _fields_to_grid(
    coords: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(coords, dtype=float)
    vv = np.asarray(values, dtype=np.complex128)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if vv.ndim != 2 or vv.shape[1] != 3 or vv.shape[0] != xyz.shape[0]:
        raise ValueError("values must have shape (N, 3) with matching N")

    x_vals, ix = _grid_index_arrays(xyz[:, 0])
    y_vals, iy = _grid_index_arrays(xyz[:, 1])
    z_vals, iz = _grid_index_arrays(xyz[:, 2])

    grid = np.zeros((z_vals.size, y_vals.size, x_vals.size, 3), dtype=np.complex128)
    grid[iz, iy, ix, :] = vv
    return x_vals, y_vals, z_vals, grid


def _slice_meta_and_component(
    *,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
    grid: np.ndarray,
    comp: int,
) -> tuple[PlotMeta, np.ndarray]:
    nx = x_vals.size
    ny = y_vals.size
    nz = z_vals.size
    if ny == 1:
        return (PlotMeta("y", float(y_vals[0]), x_vals, z_vals, "x", "z"), grid[:, 0, :, comp])
    if nx == 1:
        return (PlotMeta("x", float(x_vals[0]), y_vals, z_vals, "y", "z"), grid[:, :, 0, comp])
    if nz == 1:
        return (PlotMeta("z", float(z_vals[0]), x_vals, y_vals, "x", "y"), grid[0, :, :, comp])
    raise ValueError("Near-field grid is fully 3D; plotting expects one singleton axis.")


def _plot_nearfield_component_pairs(
    *,
    py_fields: np.ndarray,
    mstm_fields: np.ndarray,
    coords_phys: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    out_path: Path,
    channel_label: str,
    field_label: str,
) -> None:
    x_vals, y_vals, z_vals, py_grid = _fields_to_grid(coords_phys, py_fields)
    _, _, _, ms_grid = _fields_to_grid(coords_phys, mstm_fields)

    comp_names = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)

    for row, cname in enumerate(comp_names):
        meta_py, py_data = _slice_meta_and_component(
            x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, grid=py_grid, comp=row
        )
        meta_ms, ms_data = _slice_meta_and_component(
            x_vals=x_vals, y_vals=y_vals, z_vals=z_vals, grid=ms_grid, comp=row
        )
        if (
            meta_py.plane != meta_ms.plane
            or abs(meta_py.plane_value - meta_ms.plane_value) > 1e-12
            or not np.array_equal(meta_py.axis_h, meta_ms.axis_h)
            or not np.array_equal(meta_py.axis_v, meta_ms.axis_v)
        ):
            raise RuntimeError("pyceles and MSTM slices are not on matching grids.")

        py_re = np.real(py_data)
        ms_re = np.real(ms_data)
        lim = float(np.max(np.abs(np.concatenate([py_re.ravel(), ms_re.ravel()]))))
        if lim == 0.0:
            lim = 1.0

        im0 = plot_field_component(
            axes[row, 0],
            meta_py.axis_h,
            meta_py.axis_v,
            py_re,
            title=f"pyceles Re({field_label}{cname})",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            axis_0_label=f"{meta_py.axis_h_label} (nm)",
            axis_1_label=f"{meta_py.axis_v_label} (nm)",
        )
        im1 = plot_field_component(
            axes[row, 1],
            meta_py.axis_h,
            meta_py.axis_v,
            ms_re,
            title=f"MSTM Re({field_label}{cname})",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            axis_0_label=f"{meta_py.axis_h_label} (nm)",
            axis_1_label=f"{meta_py.axis_v_label} (nm)",
        )
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.02)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.02)

        for col in (0, 1):
            plot_spheres(
                axes[row, col],
                positions,
                radii,
                plane=meta_py.plane,
                plane_value=meta_py.plane_value,
                alpha=0.35,
                color="k",
            )

    fig.suptitle(f"{channel_label}: {field_label} component comparison", y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _sample_scalar_at_kxy(
    *,
    source_kx: np.ndarray,
    source_ky: np.ndarray,
    source_values: np.ndarray,
    target_kx: np.ndarray,
    target_ky: np.ndarray,
) -> np.ndarray:
    src_pts = np.column_stack(
        [np.asarray(source_kx, dtype=float), np.asarray(source_ky, dtype=float)]
    )
    src_vals = np.asarray(source_values, dtype=float).reshape(-1)
    if src_pts.shape[0] == 0:
        return np.full(np.asarray(target_kx, dtype=float).shape, np.nan, dtype=float)
    tree = cKDTree(src_pts)
    _, idx = tree.query(
        np.column_stack([np.asarray(target_kx, dtype=float), np.asarray(target_ky, dtype=float)]),
        k=1,
    )
    return src_vals[idx]


def _mstm_s11_on_py_grid(py_map: dict[str, Any], mstm_map: dict[str, Any]) -> np.ndarray:
    kx_py = np.asarray(py_map["kx"], dtype=float)
    ky_py = np.asarray(py_map["ky"], dtype=float)
    kz_py = np.asarray(py_map["kz"], dtype=float)
    out = np.full(kx_py.shape, np.nan, dtype=float)

    for hemi in ("forward", "backward"):
        mask = kz_py >= 0.0 if hemi == "forward" else kz_py <= 0.0
        if not np.any(mask):
            continue
        sampled = _sample_scalar_at_kxy(
            source_kx=np.asarray(mstm_map[hemi]["kx"], dtype=float),
            source_ky=np.asarray(mstm_map[hemi]["ky"], dtype=float),
            source_values=np.asarray(mstm_map[hemi]["s11"], dtype=float),
            target_kx=kx_py[mask],
            target_ky=ky_py[mask],
        )
        out[mask] = sampled
    return np.nan_to_num(out, nan=0.0)


def _plot_s11_maps_with_pyceles_helpers(
    *,
    py_map: dict[str, Any],
    mstm_map: dict[str, Any],
    out_py_path: Path,
    out_mstm_path: Path,
) -> None:
    beta = np.asarray(py_map["beta"], dtype=float)
    alpha = np.asarray(py_map["alpha"], dtype=float)
    py_s11 = np.asarray(py_map["I_unpolarized"], dtype=float)
    mstm_on_py_rescaled = _mstm_s11_on_py_grid(py_map, mstm_map) / S11_SCALE_MODEL1

    fig_py, _ = plot_farfield_hemispheres(
        beta,
        alpha,
        py_s11,
        title=r"pyceles $S_{11}$-like map",
        independent_scales=True,
    )
    out_py_path.parent.mkdir(parents=True, exist_ok=True)
    fig_py.savefig(out_py_path, dpi=180)
    plt.close(fig_py)

    fig_ms, _ = plot_farfield_hemispheres(
        beta,
        alpha,
        mstm_on_py_rescaled,
        title=r"MSTM $S_{11}$ map (rescaled)",
        independent_scales=True,
    )
    out_mstm_path.parent.mkdir(parents=True, exist_ok=True)
    fig_ms.savefig(out_mstm_path, dpi=180)
    plt.close(fig_ms)


def _fold_theta_curve(theta_deg: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th = np.abs(np.asarray(theta_deg, dtype=float).reshape(-1))
    yy = np.asarray(y, dtype=float).reshape(-1)
    if th.size == 0 or yy.size == 0 or th.size != yy.size:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    th_round = np.round(th, 10)
    uniq = np.unique(th_round)
    out = np.zeros_like(uniq)
    for i, tv in enumerate(uniq):
        m = th_round == tv
        out[i] = float(np.mean(yy[m]))
    order = np.argsort(uniq)
    return uniq[order], out[order]


def _sample_py_s11_at_theta(
    py_map: dict[str, Any],
    theta_deg: np.ndarray,
    *,
    incident_azimuthal_angle: float,
) -> np.ndarray:
    theta = np.asarray(theta_deg, dtype=float).reshape(-1)
    beta = np.deg2rad(np.abs(theta))
    alpha0 = float(incident_azimuthal_angle)
    alpha = np.where(theta >= 0.0, alpha0, alpha0 + np.pi)
    kx_t = np.sin(beta) * np.cos(alpha)
    ky_t = np.sin(beta) * np.sin(alpha)

    out = np.zeros_like(theta, dtype=float)
    fwd = beta <= (0.5 * np.pi + 1e-14)
    if np.any(fwd):
        out[fwd] = _sample_py_intensity_at_kxy(
            np.asarray(py_map["kx"], dtype=float),
            np.asarray(py_map["ky"], dtype=float),
            np.asarray(py_map["kz"], dtype=float),
            np.asarray(py_map["I_unpolarized"], dtype=float),
            kx_t[fwd],
            ky_t[fwd],
            hemisphere="forward",
        )
    bwd = ~fwd
    if np.any(bwd):
        out[bwd] = _sample_py_intensity_at_kxy(
            np.asarray(py_map["kx"], dtype=float),
            np.asarray(py_map["ky"], dtype=float),
            np.asarray(py_map["kz"], dtype=float),
            np.asarray(py_map["I_unpolarized"], dtype=float),
            kx_t[bwd],
            ky_t[bwd],
            hemisphere="backward",
        )
    return out


def _plot_s11_curve_semilogy(
    *,
    theta_deg: np.ndarray,
    py_s11: np.ndarray,
    mstm_s11_rescaled: np.ndarray,
    out_path: Path,
) -> None:
    th_u, py_u = _fold_theta_curve(theta_deg, py_s11)
    _, ms_u = _fold_theta_curve(theta_deg, mstm_s11_rescaled)
    if th_u.size == 0:
        return
    x = np.deg2rad(th_u)
    eps = 1.0e-30
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
    ax.semilogy(x, np.maximum(py_u, eps), label="pyceles", lw=1.6)
    ax.semilogy(x, np.maximum(ms_u, eps), label="MSTM (rescaled)", lw=1.6)
    ax.set_xlim(0.0, np.pi)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$S_{11}$")
    ax.set_title(r"Incident-plane $S_{11}(\theta)$, $\theta\in[0,\pi]$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _triple(values: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected exactly 3 values, got {len(values)}.")
    return float(values[0]), float(values[1]), float(values[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pyceles-vs-MSTM benchmark with near/far-field plots."
    )
    parser.add_argument(
        "--mstm-exe",
        type=Path,
        default=None,
        help="Path to mstm executable. Required unless --parse-only-mstm is used.",
    )
    parser.add_argument("--sphere-parameters", type=Path, default=BenchmarkConfig.sphere_parameters)
    parser.add_argument("--workdir", type=Path, default=BenchmarkConfig.workdir)
    parser.add_argument("--outdir", type=Path, default=BenchmarkConfig.outdir)
    parser.add_argument("--n-particles", type=int, default=BenchmarkConfig.n_particles)
    parser.add_argument("--wavelength", type=float, default=BenchmarkConfig.wavelength)
    parser.add_argument("--beam-width", type=float, default=BenchmarkConfig.beam_width)
    parser.add_argument("--polar-angle", type=float, default=BenchmarkConfig.polar_angle)
    parser.add_argument("--azimuthal-angle", type=float, default=BenchmarkConfig.azimuthal_angle)
    parser.add_argument("--lmax", type=int, default=BenchmarkConfig.lmax)
    parser.add_argument("--n-beta", type=int, default=BenchmarkConfig.n_beta)
    parser.add_argument("--n-alpha", type=int, default=BenchmarkConfig.n_alpha)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=BenchmarkConfig.py_solver_rtol,
        help="Shared solver tolerance for pyceles and MSTM unless tool-specific overrides are set.",
    )
    parser.add_argument(
        "--py-solver-method",
        type=str,
        choices=["auto", "gmres", "bicgstab", "lgmres", "gcrotmk", "direct"],
        default=BenchmarkConfig.py_solver_method,
    )
    parser.add_argument(
        "--py-solver-rtol",
        type=float,
        default=None,
        help="Optional pyceles solver tolerance override. Defaults to --epsilon.",
    )
    parser.add_argument("--py-solver-maxiter", type=int, default=BenchmarkConfig.py_solver_maxiter)
    parser.add_argument("--py-solver-restart", type=int, default=BenchmarkConfig.py_solver_restart)
    parser.add_argument(
        "--py-compute-dtype",
        type=str,
        choices=["complex64", "complex128"],
        default=BenchmarkConfig.py_compute_dtype,
    )
    parser.add_argument(
        "--py-accum-dtype",
        type=str,
        choices=["complex64", "complex128"],
        default=BenchmarkConfig.py_accum_dtype,
    )
    parser.add_argument(
        "--mstm-mie-epsilon",
        type=float,
        default=None,
        help="MSTM mie_epsilon. Defaults to -lmax when omitted.",
    )
    parser.add_argument(
        "--mstm-solution-epsilon",
        type=float,
        default=None,
        help="Optional MSTM solution epsilon override. Defaults to --epsilon.",
    )
    parser.add_argument(
        "--mstm-max-iterations", type=int, default=BenchmarkConfig.mstm_max_iterations
    )
    parser.add_argument(
        "--mstm-scattering-map-dimension",
        type=int,
        default=BenchmarkConfig.mstm_scattering_map_dimension,
    )
    parser.add_argument(
        "--mstm-near-field-model",
        type=int,
        choices=[1, 2],
        default=BenchmarkConfig.mstm_near_field_model,
        help="MSTM near-field mode: 1=total field, 2=scattered field.",
    )
    parser.add_argument(
        "--nf-min",
        type=float,
        nargs=3,
        default=BenchmarkConfig.near_field_grid.minimum_border,
        metavar=("XMIN", "YMIN", "ZMIN"),
    )
    parser.add_argument(
        "--nf-max",
        type=float,
        nargs=3,
        default=BenchmarkConfig.near_field_grid.maximum_border,
        metavar=("XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--nf-step", type=float, default=BenchmarkConfig.near_field_grid.step_size)
    parser.add_argument("--output-prefix", type=str, default=BenchmarkConfig.output_prefix)
    parser.add_argument(
        "--parse-only-mstm",
        action="store_true",
        help="Skip launching MSTM and only parse existing MSTM output files.",
    )
    args = parser.parse_args()
    shared_epsilon = float(args.epsilon)
    py_solver_rtol = shared_epsilon if args.py_solver_rtol is None else float(args.py_solver_rtol)
    mstm_solution_epsilon = (
        shared_epsilon if args.mstm_solution_epsilon is None else float(args.mstm_solution_epsilon)
    )
    mstm_mie_epsilon = (
        -float(args.lmax) if args.mstm_mie_epsilon is None else float(args.mstm_mie_epsilon)
    )

    cfg = BenchmarkConfig(
        mstm_exe=args.mstm_exe,
        sphere_parameters=args.sphere_parameters,
        workdir=args.workdir,
        outdir=args.outdir,
        n_particles=int(args.n_particles),
        wavelength=float(args.wavelength),
        beam_width=float(args.beam_width),
        polar_angle=float(args.polar_angle),
        azimuthal_angle=float(args.azimuthal_angle),
        lmax=int(args.lmax),
        n_beta=int(args.n_beta),
        n_alpha=int(args.n_alpha),
        py_solver_method=cast(SolverMethod, args.py_solver_method),
        py_solver_rtol=py_solver_rtol,
        py_solver_maxiter=int(args.py_solver_maxiter),
        py_solver_restart=int(args.py_solver_restart),
        py_compute_dtype=cast(ComplexDType, args.py_compute_dtype),
        py_accum_dtype=cast(ComplexDType, args.py_accum_dtype),
        mstm_mie_epsilon=mstm_mie_epsilon,
        mstm_solution_epsilon=mstm_solution_epsilon,
        mstm_max_iterations=int(args.mstm_max_iterations),
        mstm_scattering_map_dimension=int(args.mstm_scattering_map_dimension),
        mstm_near_field_model=int(args.mstm_near_field_model),
        near_field_grid=NearFieldGridConfig(
            minimum_border=_triple(args.nf_min),
            maximum_border=_triple(args.nf_max),
            step_size=float(args.nf_step),
        ),
        output_prefix=str(args.output_prefix),
        run_mstm=not bool(args.parse_only_mstm),
    )
    if cfg.run_mstm and cfg.mstm_exe is None:
        parser.error("--mstm-exe is required unless --parse-only-mstm is used.")

    cfg.workdir.mkdir(parents=True, exist_ok=True)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    sphere_data = _load_geometry(cfg.sphere_parameters, cfg.n_particles)
    geometry_file = cfg.workdir / f"{cfg.output_prefix}_sphere_parameters_used.txt"
    _write_geometry_snapshot(geometry_file, sphere_data)

    inp_file = cfg.workdir / f"{cfg.output_prefix}.inp"
    out_file = cfg.workdir / f"{cfg.output_prefix}.dat"
    nf_file = cfg.workdir / f"{cfg.output_prefix}_nearfield.dat"
    inp_file_map0 = cfg.workdir / f"{cfg.output_prefix}_map0.inp"
    out_file_map0 = cfg.workdir / f"{cfg.output_prefix}_map0.dat"
    nf_file_map0 = cfg.workdir / f"{cfg.output_prefix}_map0_nearfield_unused.dat"

    # model 1: hemisphere map + near-field
    _write_mstm_input(
        cfg,
        inp_file,
        out_file.name,
        nf_file.name,
        sphere_data,
        scattering_map_model=1,
        scattering_map_dimension=int(cfg.mstm_scattering_map_dimension),
        calculate_near_field=True,
    )
    # model 0: incident-plane theta curve only
    _write_mstm_input(
        cfg,
        inp_file_map0,
        out_file_map0.name,
        nf_file_map0.name,
        sphere_data,
        scattering_map_model=0,
        scattering_map_increment_deg=None,
        calculate_near_field=False,
    )

    if cfg.run_mstm:
        if cfg.mstm_exe is None:
            raise RuntimeError("Internal error: missing mstm executable path.")
        for inpf in (inp_file, inp_file_map0):
            proc = subprocess.run(
                [str(cfg.mstm_exe), inpf.name],
                cwd=str(cfg.workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "MSTM failed.\n"
                    f"input: {inpf.name}\n"
                    f"return code: {proc.returncode}\n"
                    f"stdout tail:\n{chr(10).join(proc.stdout.splitlines()[-40:])}\n"
                    f"stderr tail:\n{chr(10).join(proc.stderr.splitlines()[-40:])}"
                )

    mstm_out = _parse_mstm_output(out_file)
    mstm_out_map0 = _parse_mstm_output(out_file_map0)
    mstm_nf = _parse_mstm_nearfield_output(nf_file)

    mstm_coords_dimless = np.asarray(mstm_nf["coords"], dtype=float)
    mstm_coords_physical = _mstm_dimless_to_physical(mstm_coords_dimless, cfg.wavelength)
    py_out = _pyceles_run(cfg, sphere_data, mstm_coords_physical)
    chosen_mapping = dict(MSTM_POLARIZATION_MAPPING)
    eff_cmp = _eff_compare_fixed_mapping(py_out["efficiencies_basis"], mstm_out, chosen_mapping)

    positions = np.asarray(sphere_data[:, 1:4], dtype=float)
    radii = np.asarray(sphere_data[:, 0], dtype=float)
    # Use pyceles inside-mask bookkeeping instead of re-implementing point-in-sphere checks.
    inside = np.asarray(py_out["nearfield_basis"]["inside_te"], dtype=bool) | np.asarray(
        py_out["nearfield_basis"]["inside_tm"], dtype=bool
    )
    outside = ~inside

    nearfield_cmp = _compare_nearfield(
        chosen_mapping=chosen_mapping,
        py_nf=py_out["nearfield_basis"],
        mstm_nf=mstm_nf,
        outside_mask=outside,
    )
    farfield_map_cmp = _compare_farfield_s11(
        mstm_map=mstm_out["scattering_map"],
        py_map=py_out["farfield_map"],
    )

    curve0 = mstm_out_map0.get("scattering_curve_incident", {})
    theta0 = np.asarray(curve0.get("theta_deg", np.zeros((0,), dtype=float)), dtype=float)
    s110 = np.asarray(curve0.get("s11", np.zeros((0,), dtype=float)), dtype=float)
    py_curve0 = _sample_py_s11_at_theta(
        py_out["farfield_map"],
        theta0,
        incident_azimuthal_angle=float(cfg.azimuthal_angle),
    )
    farfield_curve_cmp = _real_metrics(py_curve0, s110 / S11_SCALE_MODEL0)

    # Build phase-aligned MSTM near-field channels for side-by-side maps.
    mstm_by_pol = {
        "par": {
            "E": MSTM_NEARFIELD_PHASE_CORRECTION["par"]
            * np.asarray(mstm_nf["E_par"], dtype=np.complex128),
            "H": MSTM_NEARFIELD_PHASE_CORRECTION["par"]
            * np.asarray(mstm_nf["H_par"], dtype=np.complex128),
        },
        "perp": {
            "E": MSTM_NEARFIELD_PHASE_CORRECTION["perp"]
            * np.asarray(mstm_nf["E_perp"], dtype=np.complex128),
            "H": MSTM_NEARFIELD_PHASE_CORRECTION["perp"]
            * np.asarray(mstm_nf["H_perp"], dtype=np.complex128),
        },
    }
    py_by_pol = {
        "te": {
            "E": np.asarray(py_out["nearfield_basis"]["E_te"], dtype=np.complex128),
            "H": np.asarray(py_out["nearfield_basis"]["H_te"], dtype=np.complex128),
        },
        "tm": {
            "E": np.asarray(py_out["nearfield_basis"]["E_tm"], dtype=np.complex128),
            "H": np.asarray(py_out["nearfield_basis"]["H_tm"], dtype=np.complex128),
        },
    }
    for py_ch in ("te", "tm"):
        ms_ch = chosen_mapping[py_ch]
        _plot_nearfield_component_pairs(
            py_fields=py_by_pol[py_ch]["E"],
            mstm_fields=mstm_by_pol[ms_ch]["E"],
            coords_phys=mstm_coords_physical,
            positions=positions,
            radii=radii,
            out_path=cfg.outdir / f"{cfg.output_prefix}_{py_ch}_nearfield_E_pairs.png",
            channel_label=py_ch.upper(),
            field_label="E",
        )
        _plot_nearfield_component_pairs(
            py_fields=py_by_pol[py_ch]["H"],
            mstm_fields=mstm_by_pol[ms_ch]["H"],
            coords_phys=mstm_coords_physical,
            positions=positions,
            radii=radii,
            out_path=cfg.outdir / f"{cfg.output_prefix}_{py_ch}_nearfield_H_pairs.png",
            channel_label=py_ch.upper(),
            field_label="H",
        )

    _plot_s11_maps_with_pyceles_helpers(
        py_map=py_out["farfield_map"],
        mstm_map=mstm_out["scattering_map"],
        out_py_path=cfg.outdir / f"{cfg.output_prefix}_s11_pyceles_hemispheres.png",
        out_mstm_path=cfg.outdir / f"{cfg.output_prefix}_s11_mstm_hemispheres.png",
    )
    _plot_s11_curve_semilogy(
        theta_deg=theta0,
        py_s11=py_curve0,
        mstm_s11_rescaled=s110 / S11_SCALE_MODEL0,
        out_path=cfg.outdir / f"{cfg.output_prefix}_s11_model0_semilogy.png",
    )

    mstm_summary = {
        k: v
        for k, v in mstm_out.items()
        if k not in {"scattering_map", "scattering_curve_incident"}
    }
    mstm_summary["scattering_curve_incident_scattering_map_model_0"] = {
        "n": int(theta0.size),
        "theta_deg_min": float(np.min(theta0)) if theta0.size else float("nan"),
        "theta_deg_max": float(np.max(theta0)) if theta0.size else float("nan"),
        "s11_max": float(np.max(s110)) if s110.size else float("nan"),
    }

    out = {
        "config": {
            "mstm_exe": None if cfg.mstm_exe is None else str(cfg.mstm_exe),
            "sphere_parameters": str(cfg.sphere_parameters),
            "workdir": str(cfg.workdir),
            "outdir": str(cfg.outdir),
            "n_particles": cfg.n_particles,
            "wavelength": cfg.wavelength,
            "beam_width": cfg.beam_width,
            "polar_angle_rad": cfg.polar_angle,
            "azimuthal_angle_rad": cfg.azimuthal_angle,
            "lmax": cfg.lmax,
            "n_beta": cfg.n_beta,
            "n_alpha": cfg.n_alpha,
            "epsilon": shared_epsilon,
            "py_solver_method": cfg.py_solver_method,
            "py_solver_rtol": cfg.py_solver_rtol,
            "py_solver_rtol_source": "epsilon" if args.py_solver_rtol is None else "explicit",
            "py_solver_maxiter": cfg.py_solver_maxiter,
            "py_solver_restart": cfg.py_solver_restart,
            "py_compute_dtype": cfg.py_compute_dtype,
            "py_accum_dtype": cfg.py_accum_dtype,
            "mstm_mie_epsilon": cfg.mstm_mie_epsilon,
            "mstm_solution_epsilon": cfg.mstm_solution_epsilon,
            "mstm_solution_epsilon_source": (
                "epsilon" if args.mstm_solution_epsilon is None else "explicit"
            ),
            "mstm_max_iterations": cfg.mstm_max_iterations,
            "mstm_scattering_map_dimension": cfg.mstm_scattering_map_dimension,
            "mstm_normalize_s11": False,
            "mstm_near_field_model": cfg.mstm_near_field_model,
            "near_field_grid": {
                "units": "physical (same units as sphere data file)",
                "minimum_border": list(cfg.near_field_grid.minimum_border),
                "maximum_border": list(cfg.near_field_grid.maximum_border),
                "step_size": cfg.near_field_grid.step_size,
            },
            "mstm_scattering_map_increment_deg_scattering_map_model_0": float(
                180.0 / max(int(cfg.n_beta) - 1, 1)
            ),
            "mstm_incident_frame": False,
            "output_prefix": cfg.output_prefix,
        },
        "mstm": mstm_summary,
        "mstm_nearfield": {
            "run_number": int(mstm_nf["run_number"]),
            "grid_dims": [int(v) for v in np.asarray(mstm_nf["grid_dims"]).tolist()],
            "grid_min_dimless": [float(v) for v in np.asarray(mstm_nf["grid_min"]).tolist()],
            "grid_max_dimless": [float(v) for v in np.asarray(mstm_nf["grid_max"]).tolist()],
            "coords_units": "dimensionless in MSTM output; converted to physical before pyceles comparison",
            "n_points": int(np.asarray(mstm_nf["coords"]).shape[0]),
            "n_points_inside_cluster": int(np.sum(inside)),
            "n_points_outside_cluster": int(np.sum(outside)),
        },
        "pyceles": {
            "solver": py_out["solver"],
            "efficiencies_basis": py_out["efficiencies_basis"],
            "farfield_grid_shape": [
                int(np.asarray(py_out["farfield_map"]["alpha"]).size),
                int(np.asarray(py_out["farfield_map"]["beta"]).size),
            ],
        },
        "comparison": {
            "polarization_mapping": {
                "name": "te->par,tm->perp",
                "mapping": chosen_mapping,
                "note": "MSTM near-field columns are (E_par,H_par,E_perp,H_perp); benchmark uses fixed mapping te->par, tm->perp.",
            },
            "nearfield_phase_correction_on_mstm": {
                "par_real": float(np.real(MSTM_NEARFIELD_PHASE_CORRECTION["par"])),
                "par_imag": float(np.imag(MSTM_NEARFIELD_PHASE_CORRECTION["par"])),
                "perp_real": float(np.real(MSTM_NEARFIELD_PHASE_CORRECTION["perp"])),
                "perp_imag": float(np.imag(MSTM_NEARFIELD_PHASE_CORRECTION["perp"])),
            },
            "farfield_scales": {
                "scattering_map_model_1_scale": S11_SCALE_MODEL1,
                "scattering_map_model_0_scale": S11_SCALE_MODEL0,
                "note": "scattering_map_model=0 has an extra pi factor relative to scattering_map_model=1 for normalize_s11=false in MSTM v4.0 print path.",
            },
            "efficiency_fixed_mapping": eff_cmp,
            "nearfield_field_kind": py_out["nearfield_basis"]["kind"],
            "nearfield_field": nearfield_cmp,
            "farfield_frame_note": "Compared in target/lab frame (MSTM incident_frame=f).",
            "farfield_s11_map_scattering_map_model_1_fixed_scale": farfield_map_cmp,
            "farfield_s11_curve_scattering_map_model_0_fixed_scale": farfield_curve_cmp,
        },
        "plot_outputs": {
            "te_E": str(cfg.outdir / f"{cfg.output_prefix}_te_nearfield_E_pairs.png"),
            "te_H": str(cfg.outdir / f"{cfg.output_prefix}_te_nearfield_H_pairs.png"),
            "tm_E": str(cfg.outdir / f"{cfg.output_prefix}_tm_nearfield_E_pairs.png"),
            "tm_H": str(cfg.outdir / f"{cfg.output_prefix}_tm_nearfield_H_pairs.png"),
            "s11_pyceles": str(cfg.outdir / f"{cfg.output_prefix}_s11_pyceles_hemispheres.png"),
            "s11_mstm": str(cfg.outdir / f"{cfg.output_prefix}_s11_mstm_hemispheres.png"),
            "s11_scattering_map_model_0_semilogy": str(
                cfg.outdir / f"{cfg.output_prefix}_s11_model0_semilogy.png"
            ),
        },
    }
    summary_path = cfg.outdir / f"{cfg.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
