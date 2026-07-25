"""Run an MSTM-first periodic benchmark harness for the 15-sphere homogeneous unit cell.

This example prepares a reproducible MSTM run for the 15-sphere periodic benchmark and
extracts the reference observables that MSTM can already provide for periodic systems:

- unit-cell reflectance / absorptance / transmittance (R/T/A),
- periodic scattering matrices at reciprocal-lattice directions,
- optional near-field slice data on user-controlled regular grids.

The implementation follows the MSTM v4 manual and examples, especially:
- periodic systems report unit-cell R/T/A and scattering at reciprocal-lattice
  directions,
- near-field calculations are supported for systems with periodicity,

Output layout:
- workdir/<case>/mstm_periodic_main.inp / .dat
- workdir/<case>/mstm_periodic_nf_xy.inp / .dat / _nf.dat  (optional)
- workdir/<case>/mstm_periodic_nf_xz.inp / .dat / _nf.dat  (optional)
- outdir/<case>/summary.json
- outdir/<case>/scattering_orders.npz
- outdir/<case>/nearfield_xy.npz / nearfield_xz.npz (optional)

The summary JSON is intentionally compact and points to the NPZ payloads for the
large arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class SphereRow:
    center: tuple[float, float, float]
    radius: float
    refractive_index: complex


@dataclass(frozen=True)
class NearFieldSliceConfig:
    name: str
    minimum_border: tuple[float, float, float]
    maximum_border: tuple[float, float, float]
    step_size: float
    model: int = 1
    store_surface_vector: bool = True
    expansion_spacing: float = 5.0
    expansion_order: int = 10


@dataclass(frozen=True)
class PeriodicBenchmarkCase:
    name: str
    vacuum_wavelength: float
    medium_refractive_index: complex
    incidence_polar_deg: float
    incidence_azimuth_deg: float
    lattice_ax: float
    lattice_ay: float
    lmax: int
    spheres: tuple[SphereRow, ...]
    nearfield_xy: NearFieldSliceConfig | None = None
    nearfield_xz: NearFieldSliceConfig | None = None


@dataclass(frozen=True)
class RunConfig:
    mstm_exe: Path | None
    workdir: Path
    outdir: Path
    parse_only_mstm: bool = False
    enable_nearfield_xy: bool = True
    enable_nearfield_xz: bool = True
    solution_epsilon: float = 1.0e-9
    mie_epsilon: float | None = None
    max_iterations: int = 10000
    run_pyceles: bool = True
    pyceles_nearfield_bmax: float | None = None
    pyceles_field_evanescent_decay: float = 8.0
    pyceles_periodic_method: Literal["ewald", "rayleigh"] = "ewald"


_NUMERIC_TOKEN_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?")


def _fifteen_spheres_homogeneous_case() -> PeriodicBenchmarkCase:
    sphere_rows = (
        ((500.0, 500.0, 320.0), 100.0),
        ((300.0, 200.0, 380.0), 110.0),
        ((700.0, 750.0, 220.0), 90.0),
        ((200.0, 800.0, 700.0), 120.0),
        ((800.0, 300.0, 650.0), 80.0),
        ((100.0, 100.0, 1100.0), 100.0),
        ((900.0, 900.0, 1000.0), 110.0),
        ((500.0, 100.0, 1300.0), 90.0),
        ((150.0, 450.0, 1200.0), 120.0),
        ((850.0, 600.0, 1500.0), 80.0),
        ((450.0, 700.0, 1800.0), 100.0),
        ((250.0, 300.0, 1900.0), 110.0),
        ((750.0, 150.0, 2100.0), 90.0),
        ((900.0, 500.0, 2300.0), 120.0),
        ((500.0, 900.0, 2400.0), 80.0),
    )
    spheres = tuple(
        SphereRow(center=position, radius=radius, refractive_index=2.0 + 0.0j)
        for position, radius in sphere_rows
    )
    # Two practical first slices:
    # - xy plane above the top particle to compare transmitted-field structure,
    # - xz plane through the cell center for a cross-sectional field view.
    nearfield_xy = NearFieldSliceConfig(
        name="xy",
        minimum_border=(0.0, 0.0, 3000.0),
        maximum_border=(1001.0, 1001.0, 3000.0),
        step_size=10.0,
    )
    nearfield_xz = NearFieldSliceConfig(
        name="xz",
        minimum_border=(0.0, 500.5, -200.0),
        maximum_border=(1001.0, 500.5, 3200.0),
        step_size=10.0,
    )
    return PeriodicBenchmarkCase(
        name="fifteen_spheres_homogeneous",
        vacuum_wavelength=500.0,
        medium_refractive_index=1.0 + 0.0j,
        incidence_polar_deg=30.0,
        incidence_azimuth_deg=270.0,
        lattice_ax=1001.0,
        lattice_ay=1001.0,
        lmax=3,
        spheres=spheres,
        nearfield_xy=nearfield_xy,
        nearfield_xz=nearfield_xz,
    )


CASES: dict[str, PeriodicBenchmarkCase] = {
    "fifteen_spheres_homogeneous": _fifteen_spheres_homogeneous_case(),
}


def _fmt_d(value: float) -> str:
    return f"{float(value):.10e}".replace("e", "d", 1)


def _float_d(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "E"))


def _extract_float_tokens(line: str) -> list[float]:
    return [_float_d(tok) for tok in _NUMERIC_TOKEN_RE.findall(line)]


def _complex_payload(value: complex) -> dict[str, float]:
    z = complex(value)
    return {"real": float(z.real), "imag": float(z.imag)}


def _write_geometry_snapshot(path: Path, case: PeriodicBenchmarkCase) -> None:
    rows = []
    for sphere in case.spheres:
        x, y, z = sphere.center
        rows.append(
            [
                x,
                y,
                z,
                sphere.radius,
                sphere.refractive_index.real,
                sphere.refractive_index.imag,
            ]
        )
    header = "x y z radius n_real n_imag"
    np.savetxt(path, np.asarray(rows, dtype=float), fmt="%.12g", header=header, comments="")


def _write_mstm_input(
    case: PeriodicBenchmarkCase,
    cfg: RunConfig,
    inp_path: Path,
    output_filename: str,
    *,
    calculate_scattering_matrix: bool,
    nearfield: NearFieldSliceConfig | None,
    nearfield_output_filename: str | None = None,
) -> None:
    if nearfield is not None and nearfield_output_filename is None:
        raise ValueError("nearfield_output_filename is required when nearfield is enabled.")

    k0 = 2.0 * np.pi / float(case.vacuum_wavelength)
    mie_eps = -float(case.lmax) if cfg.mie_epsilon is None else float(cfg.mie_epsilon)

    host_n = float(np.real(case.medium_refractive_index))
    if host_n <= 0.0:
        raise ValueError(
            f"Host refractive index must be positive real. Got {case.medium_refractive_index!r}."
        )

    lines: list[str] = [
        "output_file",
        output_filename,
        "append_output_file",
        "f",
        "print_sphere_data",
        "t",
        "number_spheres",
        str(len(case.spheres)),
        "sphere_data",
    ]
    for sphere in case.spheres:
        x, y, z = sphere.center
        # MSTM periodic inputs combine a global refractive-index scale factor with
        # per-sphere relative refractive indices.
        n_rel = complex(sphere.refractive_index) / complex(host_n, 0.0)
        nr = float(np.real(n_rel))
        ni = float(np.imag(n_rel))
        lines.append(
            f"{_fmt_d(x)},{_fmt_d(y)},{_fmt_d(z)},{_fmt_d(sphere.radius)},({_fmt_d(nr)},{_fmt_d(ni)})"
        )
    lines += [
        "end_of_sphere_data",
        # Homogeneous host, no interfaces.
        "number_plane_boundaries",
        "0",
        # Periodic scattering in MSTM uses layer_ref_index(...) as host RI for
        # normalization/propagation; keep it aligned with ref_index_scale_factor.
        "layer_ref_index",
        f"({_fmt_d(host_n)},0.0d0)",
        # Positions, radii, and cell widths are specified in physical units and
        # converted internally by MSTM through this scale factor.
        "length_scale_factor",
        _fmt_d(k0),
        # Sphere refractive indices are written relative to host and then scaled
        # back to absolute values through this factor.
        "ref_index_scale_factor",
        f"({_fmt_d(host_n)},0.0d0)",
        "periodic_lattice",
        "t",
        "cell_width",
        f"{_fmt_d(case.lattice_ax)},{_fmt_d(case.lattice_ay)}",
        # Periodic lattice model is plane-wave only in MSTM.
        "incident_beta_deg",
        _fmt_d(case.incidence_polar_deg),
        "incident_alpha_deg",
        _fmt_d(case.incidence_azimuth_deg),
        "max_iterations",
        str(int(cfg.max_iterations)),
        "solution_epsilon",
        _fmt_d(cfg.solution_epsilon),
        # Existing pyceles-vs-MSTM scripts use mie_epsilon=-lmax as the practical default.
        "mie_epsilon",
        _fmt_d(mie_eps),
        "calculate_scattering_matrix",
        "t" if calculate_scattering_matrix else "f",
        # Keep scattering data in the target/lab frame.
        "incident_frame",
        "f",
    ]

    if nearfield is None:
        lines += ["calculate_near_field", "f"]
    else:
        k0 = 2.0 * np.pi / float(case.vacuum_wavelength)
        nf_min = tuple(float(v) * k0 for v in nearfield.minimum_border)
        nf_max = tuple(float(v) * k0 for v in nearfield.maximum_border)
        nf_step = float(nearfield.step_size) * k0
        lines += [
            "calculate_near_field",
            "t",
            "near_field_calculation_model",
            str(int(nearfield.model)),
            "store_surface_vector",
            "t" if nearfield.store_surface_vector else "f",
            "near_field_expansion_spacing",
            _fmt_d(nearfield.expansion_spacing),
            "near_field_expansion_order",
            str(int(nearfield.expansion_order)),
            "near_field_output_file",
            str(nearfield_output_filename),
            # Per manual, near-field borders are NOT rescaled by length_scale_factor,
            # so we convert from physical units to MSTM dimensionless coordinates here.
            "near_field_minimum_border",
            f"{_fmt_d(nf_min[0])},{_fmt_d(nf_min[1])},{_fmt_d(nf_min[2])}",
            "near_field_maximum_border",
            f"{_fmt_d(nf_max[0])},{_fmt_d(nf_max[1])},{_fmt_d(nf_max[2])}",
            "near_field_step_size",
            _fmt_d(nf_step),
        ]

    lines += ["end_of_options"]
    inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_mstm(exe: Path, inp_path: Path, workdir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exe), inp_path.name],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )


def _parse_block(output_text: str, pattern: str) -> list[float]:
    match = re.search(pattern, output_text, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Could not find output block matching pattern: {pattern!r}")
    raw = match.group(1)
    tokens = _extract_float_tokens(raw)
    if tokens:
        return tokens
    return [_float_d(tok) for tok in raw.split()]


def _parse_periodic_scattering(text: str) -> dict[str, dict[str, np.ndarray]]:
    lines = text.splitlines()

    def _parse_section(start_label: str, stop_label: str | None) -> dict[str, np.ndarray]:
        start = None
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith(start_label.lower()):
                start = i
                break
        if start is None:
            return {
                "kx": np.zeros((0,), dtype=float),
                "ky": np.zeros((0,), dtype=float),
                "matrix": np.zeros((0, 16), dtype=float),
            }

        i = start + 1
        while i < len(lines):
            ln = lines[i].strip().lower()
            if ln.startswith("kx") and "ky" in ln:
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

        if not rows:
            return {
                "kx": np.zeros((0,), dtype=float),
                "ky": np.zeros((0,), dtype=float),
                "matrix": np.zeros((0, 16), dtype=float),
            }
        arr = np.asarray(rows, dtype=float)
        return {
            "kx": arr[:, 0],
            "ky": arr[:, 1],
            "matrix": arr[:, 2:18],
        }

    return {
        "backward": _parse_section(
            "backward hemisphere scattering", "forward hemisphere scattering"
        ),
        "forward": _parse_section("forward hemisphere scattering", None),
    }


def _parse_mstm_output(output_path: Path) -> dict[str, Any]:
    text = output_path.read_text(encoding="utf-8", errors="ignore")
    warnings = [ln.strip() for ln in text.splitlines() if ln.strip().lower().startswith("warning:")]

    result: dict[str, Any] = {"warnings": warnings}
    try:
        cell_block = _parse_block(
            text,
            r"periodic lattice cell width, incident lateral vector\s*\n([^\n]+)\n",
        )
        if len(cell_block) >= 4:
            result["cell_width_dimless"] = [float(cell_block[0]), float(cell_block[1])]
            result["incident_lateral_vector_dimless"] = [float(cell_block[2]), float(cell_block[3])]
    except Exception:
        pass

    try:
        iter_block = _parse_block(
            text,
            r"number iterations, error, solution time\s*\n([^\n]+)\n",
        )
        if len(iter_block) >= 3:
            result["iterations"] = round(iter_block[0])
            result["solution_error"] = float(iter_block[1])
            result["solution_time_seconds"] = float(iter_block[2])
    except Exception:
        pass

    try:
        mie_block = _parse_block(
            text,
            r"maximum Mie order, number of equations:\s*\n([^\n]+)\n",
        )
        if len(mie_block) >= 2:
            result["max_mie_order"] = round(mie_block[0])
            result["number_of_equations"] = round(mie_block[1])
    except Exception:
        pass

    try:
        rta = _parse_block(
            text,
            r"unit cell reflectance, absorptance, transmittance \(unpol, par, perp\)\s*\n([^\n]+)\n",
        )
        if len(rta) >= 9:
            result["rta"] = {
                "unpolarized": {
                    "reflectance": float(rta[0]),
                    "absorptance": float(rta[1]),
                    "transmittance": float(rta[2]),
                },
                "parallel": {
                    "reflectance": float(rta[3]),
                    "absorptance": float(rta[4]),
                    "transmittance": float(rta[5]),
                },
                "perpendicular": {
                    "reflectance": float(rta[6]),
                    "absorptance": float(rta[7]),
                    "transmittance": float(rta[8]),
                },
            }
    except Exception:
        pass

    result["periodic_scattering"] = _parse_periodic_scattering(text)
    return result


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
    run_number = round(_extract_float_tokens(lines[idx])[0])
    idx += 1
    idx = _next_nonempty(idx)
    n_intersecting = round(_extract_float_tokens(lines[idx])[0])
    idx += 1
    intersecting_rows: list[list[float]] = []
    for _ in range(n_intersecting):
        idx = _next_nonempty(idx)
        vals = _extract_float_tokens(lines[idx])
        intersecting_rows.append(vals[:4])
        idx += 1

    idx = _next_nonempty(idx)
    n_boundaries = round(_extract_float_tokens(lines[idx])[0])
    idx += 1
    boundaries: list[float] = []
    for _ in range(n_boundaries):
        idx = _next_nonempty(idx)
        vals = _extract_float_tokens(lines[idx])
        if vals:
            boundaries.append(float(vals[0]))
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
    dims = np.asarray([round(v) for v in dims_tokens[:3]], dtype=int)
    idx += 1

    npts = int(np.prod(dims))
    coords = np.zeros((npts, 3), dtype=float)
    e_par = np.zeros((npts, 3), dtype=np.complex128)
    h_par = np.zeros((npts, 3), dtype=np.complex128)
    e_perp = np.zeros((npts, 3), dtype=np.complex128)
    h_perp = np.zeros((npts, 3), dtype=np.complex128)

    def _as_complex(values: np.ndarray) -> np.ndarray:
        return values[0::2] + 1j * values[1::2]

    for i in range(npts):
        idx = _next_nonempty(idx)
        toks = _extract_float_tokens(lines[idx])
        idx += 1
        if len(toks) < 27:
            raise RuntimeError(
                f"Malformed near-field row {i}: expected >=27 floats, got {len(toks)}."
            )
        vals_arr = np.asarray(toks[:27], dtype=float)
        coords[i, :] = vals_arr[0:3]
        e_par[i, :] = _as_complex(vals_arr[3:9])
        h_par[i, :] = _as_complex(vals_arr[9:15])
        e_perp[i, :] = _as_complex(vals_arr[15:21])
        h_perp[i, :] = _as_complex(vals_arr[21:27])

    return {
        "run_number": run_number,
        "intersecting_spheres": np.asarray(intersecting_rows, dtype=float)
        if intersecting_rows
        else np.zeros((0, 4), dtype=float),
        "intersecting_boundaries_z": np.asarray(boundaries, dtype=float),
        "grid_min_dimless": grid_min,
        "grid_max_dimless": grid_max,
        "grid_dims": dims,
        "coords_dimless": coords,
        "E_par": e_par,
        "H_par": h_par,
        "E_perp": e_perp,
        "H_perp": h_perp,
    }


def _mstm_dimless_to_physical(points_dimless: np.ndarray, wavelength: float) -> np.ndarray:
    k0 = 2.0 * np.pi / float(wavelength)
    return np.asarray(points_dimless, dtype=float) / float(k0)


def _save_periodic_scattering_npz(path: Path, scattering: dict[str, dict[str, np.ndarray]]) -> None:
    np.savez_compressed(
        path,
        backward_kx=scattering["backward"]["kx"],
        backward_ky=scattering["backward"]["ky"],
        backward_matrix=scattering["backward"]["matrix"],
        forward_kx=scattering["forward"]["kx"],
        forward_ky=scattering["forward"]["ky"],
        forward_matrix=scattering["forward"]["matrix"],
    )


def _save_nearfield_npz(path: Path, nearfield: dict[str, Any], wavelength: float) -> None:
    coords_physical = _mstm_dimless_to_physical(nearfield["coords_dimless"], wavelength)
    np.savez_compressed(
        path,
        coords_dimless=nearfield["coords_dimless"],
        coords_physical=coords_physical,
        grid_dims=nearfield["grid_dims"],
        grid_min_dimless=nearfield["grid_min_dimless"],
        grid_max_dimless=nearfield["grid_max_dimless"],
        E_par=nearfield["E_par"],
        H_par=nearfield["H_par"],
        E_perp=nearfield["E_perp"],
        H_perp=nearfield["H_perp"],
        intersecting_spheres=nearfield["intersecting_spheres"],
        intersecting_boundaries_z=nearfield["intersecting_boundaries_z"],
    )


def _grid_payload(
    slice_cfg: NearFieldSliceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step = float(slice_cfg.step_size)

    def _axis_samples(vmin: float, vmax: float) -> np.ndarray:
        if np.isclose(vmin, vmax, rtol=0.0, atol=1e-12):
            return np.asarray([vmin], dtype=float)
        # MSTM includes an endpoint when it falls exactly on the regular grid,
        # but it does not add a final point beyond a non-grid-aligned maximum.
        values = np.arange(vmin, vmax + 0.5 * step, step, dtype=float)
        if values.size == 0:
            return np.asarray([vmin], dtype=float)
        return values

    x = _axis_samples(float(slice_cfg.minimum_border[0]), float(slice_cfg.maximum_border[0]))
    y = _axis_samples(float(slice_cfg.minimum_border[1]), float(slice_cfg.maximum_border[1]))
    z = _axis_samples(float(slice_cfg.minimum_border[2]), float(slice_cfg.maximum_border[2]))
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.column_stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)))
    dims = np.array([x.size, y.size, z.size], dtype=np.int64)
    return (
        np.asarray(coords, dtype=float),
        dims,
        np.asarray(slice_cfg.minimum_border, dtype=float),
        np.asarray(slice_cfg.maximum_border, dtype=float),
    )


def _count_nonfinite_complex(array: np.ndarray) -> int:
    values = np.asarray(array, dtype=np.complex128)
    valid = np.isfinite(values.real) & np.isfinite(values.imag)
    return int(np.count_nonzero(~valid))


def _intersecting_spheres(
    spheres: tuple[SphereRow, ...],
    slice_cfg: NearFieldSliceConfig,
) -> np.ndarray:
    min_b = np.asarray(slice_cfg.minimum_border, dtype=float)
    max_b = np.asarray(slice_cfg.maximum_border, dtype=float)
    fixed_axes = np.where(np.isclose(min_b, max_b, rtol=0.0, atol=1e-12))[0]
    rows: list[list[float]] = []
    for sphere in spheres:
        center = np.asarray(sphere.center, dtype=float)
        radius = float(sphere.radius)
        if fixed_axes.size == 0:
            continue
        intersects = True
        for axis in fixed_axes:
            intersects = intersects and (abs(float(center[axis]) - float(min_b[axis])) <= radius)
        if intersects:
            rows.append([float(center[0]), float(center[1]), float(center[2]), radius])
    if not rows:
        return np.zeros((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def _axis_from_coords(coords: np.ndarray, axis: int) -> np.ndarray:
    return np.unique(np.round(np.asarray(coords[:, axis], dtype=float), 10))


def _complex_similarity_metrics(model: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    a = np.asarray(model, dtype=np.complex128).reshape(-1)
    b = np.asarray(reference, dtype=np.complex128).reshape(-1)
    valid = np.isfinite(a.real) & np.isfinite(a.imag) & np.isfinite(b.real) & np.isfinite(b.imag)
    a = a[valid]
    b = b[valid]
    if a.size == 0:
        return {"n_valid": 0}
    den = np.vdot(a, a)
    scale = np.vdot(a, b) / den if abs(den) > 0.0 else 0.0 + 0.0j
    norm_b = float(np.linalg.norm(b))
    denom_b = norm_b if norm_b > 1e-30 else 1e-30
    rel = float(np.linalg.norm(a - b) / denom_b)
    rel_scaled = float(np.linalg.norm(scale * a - b) / denom_b)
    ia = np.abs(a) ** 2
    ib = np.abs(b) ** 2
    den_i = float(np.dot(ia, ia))
    scale_i = float(np.dot(ia, ib) / den_i) if den_i > 0.0 else 0.0
    norm_ib = float(np.linalg.norm(ib))
    denom_ib = norm_ib if norm_ib > 1e-30 else 1e-30
    rel_i = float(np.linalg.norm(ia - ib) / denom_ib)
    rel_i_scaled = float(np.linalg.norm(scale_i * ia - ib) / denom_ib)
    corr = float(np.corrcoef(ia, ib)[0, 1]) if ia.size > 1 else 1.0
    return {
        "n_valid": int(a.size),
        "rel_l2": rel,
        "rel_l2_best_complex_scale": rel_scaled,
        "best_complex_scale": {"real": float(np.real(scale)), "imag": float(np.imag(scale))},
        "intensity_rel_l2": rel_i,
        "intensity_rel_l2_best_scalar": rel_i_scaled,
        "best_intensity_scale": scale_i,
        "intensity_corrcoef": corr,
    }


def _vector_intensity_plane(values: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    # MSTM near-field rows are emitted in Fortran-style traversal order.
    # Pyceles and MSTM arrays are evaluated on the same reference-point order,
    # so we preserve that ordering when regridding for visualization.
    tensor = np.asarray(values, dtype=np.complex128).reshape(*dims, 3, order="F")
    intensity = np.sum(np.abs(tensor) ** 2, axis=-1)
    nx, ny, nz = dims
    if nz == 1:
        return np.asarray(intensity[:, :, 0], dtype=float)
    if ny == 1:
        return np.asarray(intensity[:, 0, :], dtype=float)
    if nx == 1:
        return np.asarray(intensity[0, :, :], dtype=float)
    raise ValueError(f"Expected planar slice with one singleton axis, got dims={dims!r}.")


def _plane_axes_from_reference(ref_npz: Any) -> tuple[np.ndarray, np.ndarray, str, str]:
    coords = np.asarray(ref_npz["coords_physical"], dtype=float)
    dims_arr = np.asarray(ref_npz["grid_dims"], dtype=np.int64).reshape(3)
    dims: tuple[int, int, int] = (int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2]))
    x = _axis_from_coords(coords, 0)
    y = _axis_from_coords(coords, 1)
    z = _axis_from_coords(coords, 2)
    nx, ny, nz = dims
    if nz == 1:
        return x, y, "x", "y"
    if ny == 1:
        return x, z, "x", "z"
    if nx == 1:
        return y, z, "y", "z"
    raise ValueError(f"Expected planar slice with one singleton axis, got dims={dims!r}.")


def _plot_bipanel_intensity(
    *,
    axis_h: np.ndarray,
    axis_v: np.ndarray,
    mstm_values: np.ndarray,
    pyceles_values: np.ndarray,
    out_path: Path,
    title: str,
    axis_h_label: str,
    axis_v_label: str,
) -> None:
    import matplotlib.pyplot as plt

    extent = [
        float(np.min(axis_h)),
        float(np.max(axis_h)),
        float(np.min(axis_v)),
        float(np.max(axis_v)),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)
    panels = [
        ("MSTM", mstm_values),
        ("pyceles", pyceles_values),
    ]
    for ax, (label, data) in zip(axs, panels, strict=True):
        im = ax.imshow(
            np.asarray(data, dtype=float).T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
        )
        ax.set_title(label)
        ax.set_xlabel(axis_h_label)
        ax.set_ylabel(axis_v_label)
        colorbar = fig.colorbar(im, ax=ax)
        colorbar.set_label(r"$|F|^2$")
    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _jsonable_case(case: PeriodicBenchmarkCase) -> dict[str, Any]:
    return {
        "name": case.name,
        "vacuum_wavelength": float(case.vacuum_wavelength),
        "medium_refractive_index": _complex_payload(case.medium_refractive_index),
        "incidence_polar_deg": float(case.incidence_polar_deg),
        "incidence_azimuth_deg": float(case.incidence_azimuth_deg),
        "lattice": {"ax": float(case.lattice_ax), "ay": float(case.lattice_ay)},
        "lmax": int(case.lmax),
        "spheres": [
            {
                "center": [float(v) for v in sphere.center],
                "radius": float(sphere.radius),
                "refractive_index": _complex_payload(sphere.refractive_index),
            }
            for sphere in case.spheres
        ],
    }


def _build_summary(
    case: PeriodicBenchmarkCase,
    cfg: RunConfig,
    main_output: dict[str, Any],
    scattering_npz: Path,
    nearfield_artifacts: dict[str, dict[str, Any]],
    pyceles_output: dict[str, Any] | None,
    workdir_case: Path,
    outdir_case: Path,
) -> dict[str, Any]:
    periodic_scattering = main_output.get("periodic_scattering", {})
    summary: dict[str, Any] = {
        "case": _jsonable_case(case),
        "mstm": {
            "executable": None if cfg.mstm_exe is None else str(cfg.mstm_exe),
            "workdir": str(workdir_case),
            "outdir": str(outdir_case),
            "solution_epsilon": float(cfg.solution_epsilon),
            "mie_epsilon": float(-case.lmax if cfg.mie_epsilon is None else cfg.mie_epsilon),
            "main": {
                key: value for key, value in main_output.items() if key != "periodic_scattering"
            },
            "periodic_scattering": {
                "npz": str(scattering_npz),
                "backward_count": int(
                    periodic_scattering.get("backward", {}).get("kx", np.zeros((0,))).size
                ),
                "forward_count": int(
                    periodic_scattering.get("forward", {}).get("kx", np.zeros((0,))).size
                ),
                "matrix_layout": "Rows are reciprocal directions. Columns are S11,S21,S31,S41,S12,...,S44 in MSTM output order.",
            },
            "nearfield": nearfield_artifacts,
            "notes": [
                "Periodic systems in MSTM report unit-cell reflectance/absorptance/transmittance instead of finite-cluster efficiencies.",
                "Periodic scattering matrices are written only at reciprocal-lattice directions.",
                "Near-field borders are specified in dimensionless coordinates in MSTM output; NPZ stores both dimensionless and physical coordinates.",
            ],
        },
    }
    if pyceles_output is not None:
        summary["pyceles"] = pyceles_output
        mstm_rta = main_output.get("rta")
        if isinstance(mstm_rta, dict):
            try:
                te = pyceles_output["channels"]["te"]
                tm = pyceles_output["channels"]["tm"]
                d_par = {
                    "R": float(tm["R"] - float(mstm_rta["parallel"]["reflectance"])),
                    "T": float(tm["T"] - float(mstm_rta["parallel"]["transmittance"])),
                    "A": float(tm["A"] - float(mstm_rta["parallel"]["absorptance"])),
                }
                d_perp = {
                    "R": float(te["R"] - float(mstm_rta["perpendicular"]["reflectance"])),
                    "T": float(te["T"] - float(mstm_rta["perpendicular"]["transmittance"])),
                    "A": float(te["A"] - float(mstm_rta["perpendicular"]["absorptance"])),
                }
                summary["comparison"] = {
                    "mapping_convention": {
                        "parallel": "pyceles_tm",
                        "perpendicular": "pyceles_te",
                    },
                    "delta_parallel": d_par,
                    "delta_perpendicular": d_perp,
                    "delta_unpolarized_avg": {
                        "R": float(
                            pyceles_output["unpolarized"]["R"]
                            - float(mstm_rta["unpolarized"]["reflectance"])
                        ),
                        "T": float(
                            pyceles_output["unpolarized"]["T"]
                            - float(mstm_rta["unpolarized"]["transmittance"])
                        ),
                        "A": float(
                            pyceles_output["unpolarized"]["A"]
                            - float(mstm_rta["unpolarized"]["absorptance"])
                        ),
                    },
                }
            except Exception:
                pass
    return summary


def _save_pyceles_periodic_npz(path: Path, periodic: Any) -> None:
    np.savez_compressed(
        path,
        lattice_a1=np.asarray(periodic.lattice_a1, dtype=float),
        lattice_a2=np.asarray(periodic.lattice_a2, dtype=float),
        unit_cell_area=np.asarray(float(periodic.unit_cell_area), dtype=float),
        incident_k_parallel=np.asarray(periodic.incident_k_parallel, dtype=float),
        order_mn=np.asarray(periodic.order_mn, dtype=np.int32),
        order_k_parallel=np.asarray(periodic.order_k_parallel, dtype=float),
        order_kz=np.asarray(periodic.order_kz, dtype=np.complex128),
        order_propagating=np.asarray(periodic.order_propagating, dtype=bool),
        reflected_amplitudes=np.asarray(periodic.reflected_amplitudes, dtype=np.complex128),
        transmitted_amplitudes=np.asarray(periodic.transmitted_amplitudes, dtype=np.complex128),
        reflected_power_per_order=np.asarray(periodic.reflected_power_per_order, dtype=float),
        transmitted_power_per_order=np.asarray(periodic.transmitted_power_per_order, dtype=float),
        incident_power_per_area=np.asarray(float(periodic.incident_power_per_area), dtype=float),
        reflectance=np.asarray(float(periodic.reflectance), dtype=float),
        transmittance=np.asarray(float(periodic.transmittance), dtype=float),
        absorptance=np.asarray(float(periodic.absorptance), dtype=float),
    )


def _pyceles_nearfield_reference_distance(case: PeriodicBenchmarkCase) -> float:
    """Return a conservative exterior distance for near-field order selection."""
    radii = [float(sphere.radius) for sphere in case.spheres]
    if not radii:
        return 1.0
    return max(2.0 * max(radii), 1.0e-12)


def _field_bmax_for_evanescent_decay(*, k: float, distance: float, decay: float) -> float:
    """Return a reciprocal cutoff from a target evanescent decay length."""
    kf = float(k)
    d = max(float(distance), 1.0e-12)
    q = max(float(decay), 0.0) / d
    return float(math.sqrt(kf * kf + q * q))


def _resolve_pyceles_nearfield_bmax(
    case: PeriodicBenchmarkCase,
    cfg: RunConfig,
) -> tuple[float | None, dict[str, Any] | None]:
    """Resolve the pyceles periodic near-field reciprocal output radius."""
    if cfg.pyceles_nearfield_bmax is not None:
        value = float(cfg.pyceles_nearfield_bmax)
        return value, {"source": "explicit", "value": value}
    distance = _pyceles_nearfield_reference_distance(case)
    k = 2.0 * math.pi * float(np.real(case.medium_refractive_index)) / float(case.vacuum_wavelength)
    decay = float(cfg.pyceles_field_evanescent_decay)
    value = _field_bmax_for_evanescent_decay(k=k, distance=distance, decay=decay)
    return value, {
        "source": "auto_evanescent_decay",
        "value": float(value),
        "k": float(k),
        "value_over_k": float(value / k) if k > 0.0 else None,
        "characteristic_distance": float(distance),
        "evanescent_decay": float(decay),
    }


def _run_pyceles_case(
    case: PeriodicBenchmarkCase,
    outdir_case: Path,
    *,
    cfg: RunConfig,
    nearfield_bmax: float | None,
    nearfield_bmax_policy: dict[str, Any] | None = None,
    nearfield_reference: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    import pyceles as pcl

    beta = math.radians(float(case.incidence_polar_deg))
    alpha = math.radians(float(case.incidence_azimuth_deg))
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(ax=float(case.lattice_ax), ay=float(case.lattice_ay)),
        options=pcl.PeriodicOptions(method=cfg.pyceles_periodic_method),
    )
    particles = [
        pcl.Sphere(
            position=(float(sphere.center[0]), float(sphere.center[1]), float(sphere.center[2])),
            radius=float(sphere.radius),
            refractive_index=complex(sphere.refractive_index),
        )
        for sphere in case.spheres
    ]

    channels: dict[str, dict[str, Any]] = {}
    runs: dict[str, Any] = {}
    for pol in ("TE", "TM"):
        source = pcl.PlaneWave(
            wavelength=float(case.vacuum_wavelength),
            medium_n=complex(case.medium_refractive_index),
            polarization=pol,
            polar_angle=beta,
            azimuthal_angle=alpha,
            amplitude=1.0,
        )
        config = pcl.SimulationConfig(
            wavelength=float(case.vacuum_wavelength),
            n_medium=complex(case.medium_refractive_index),
            lmax=int(case.lmax),
            source=source,
            periodic=periodic,
            solver_method="direct",
            verbose=False,
        )
        run = pcl.Simulation(config, particles=particles).run(include_farfield=False)
        if run.periodic is None:
            raise RuntimeError("Periodic pyceles run did not populate `SimulationResult.periodic`.")
        periodic_payload = run.periodic
        npz_path = outdir_case / f"pyceles_orders_{pol.lower()}.npz"
        _save_pyceles_periodic_npz(npz_path, periodic_payload)
        key = pol.lower()
        channels[key] = {
            "npz": str(npz_path),
            "R": float(periodic_payload.reflectance),
            "T": float(periodic_payload.transmittance),
            "A": float(periodic_payload.absorptance),
            "order_count": int(np.asarray(periodic_payload.order_mn).shape[0]),
            "propagating_count": int(
                np.count_nonzero(np.asarray(periodic_payload.order_propagating))
            ),
        }
        runs[key] = run

    nearfield_artifacts: dict[str, dict[str, Any]] = {}
    slice_jobs: list[NearFieldSliceConfig] = []
    if case.nearfield_xy is not None:
        slice_jobs.append(case.nearfield_xy)
    if case.nearfield_xz is not None:
        slice_jobs.append(case.nearfield_xz)

    k0 = 2.0 * np.pi / float(case.vacuum_wavelength)
    run_te = runs["te"]
    run_tm = runs["tm"]
    for slice_cfg in slice_jobs:
        ref_meta = None if nearfield_reference is None else nearfield_reference.get(slice_cfg.name)
        if ref_meta is None:
            coords, dims, min_b, max_b = _grid_payload(slice_cfg)
        else:
            coords = np.asarray(ref_meta["coords_physical"], dtype=float)
            dims_arr = np.asarray(ref_meta["grid_dims"], dtype=np.int64).reshape(3)
            dims = np.asarray(
                (int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2])), dtype=np.int64
            )
            min_dimless = np.asarray(ref_meta["grid_min_dimless"], dtype=float).reshape(3)
            max_dimless = np.asarray(ref_meta["grid_max_dimless"], dtype=float).reshape(3)
            min_b = np.asarray([float(v) / float(k0) for v in min_dimless], dtype=float)
            max_b = np.asarray([float(v) / float(k0) for v in max_dimless], dtype=float)
        try:
            nf_te = pcl.compute_periodic_near_field(
                run_te,
                points=coords,
                channel="mixed",
                field_bmax=nearfield_bmax,
                show_progress=True,
            )
            nf_tm = pcl.compute_periodic_near_field(
                run_tm,
                points=coords,
                channel="mixed",
                field_bmax=nearfield_bmax,
                show_progress=True,
            )
        except NotImplementedError as exc:
            nearfield_artifacts[slice_cfg.name] = {
                "status": "not_implemented",
                "reason": str(exc),
                "n_points": int(coords.shape[0]),
                "grid_dims": np.asarray(dims, dtype=np.int64).tolist(),
                "field_bmax": None if nearfield_bmax is None else float(nearfield_bmax),
            }
            continue
        e_par = np.asarray(nf_tm.E_total, dtype=np.complex128).reshape(-1, 3)
        h_par = np.asarray(nf_tm.H_total, dtype=np.complex128).reshape(-1, 3)
        e_perp = np.asarray(nf_te.E_total, dtype=np.complex128).reshape(-1, 3)
        h_perp = np.asarray(nf_te.H_total, dtype=np.complex128).reshape(-1, 3)
        npz_path = outdir_case / f"pyceles_nearfield_{slice_cfg.name}.npz"
        np.savez_compressed(
            npz_path,
            coords_physical=np.asarray(coords, dtype=float),
            coords_dimless=np.asarray(coords, dtype=float) * float(k0),
            grid_dims=np.asarray(dims, dtype=np.int64),
            grid_min_dimless=np.asarray(min_b, dtype=float) * float(k0),
            grid_max_dimless=np.asarray(max_b, dtype=float) * float(k0),
            E_par=e_par,
            H_par=h_par,
            E_perp=e_perp,
            H_perp=h_perp,
            intersecting_spheres=_intersecting_spheres(case.spheres, slice_cfg),
            intersecting_boundaries_z=np.zeros((0,), dtype=float),
        )
        nearfield_artifacts[slice_cfg.name] = {
            "npz": str(npz_path),
            "n_points": int(coords.shape[0]),
            "grid_dims": np.asarray(dims, dtype=np.int64).tolist(),
            "mapping_convention": {"parallel": "tm", "perpendicular": "te"},
            "field_kind": "total (periodic near-field evaluator)",
            "field_bmax": None if nearfield_bmax is None else float(nearfield_bmax),
            "periodic_consistency": "Automatic exterior/interior periodic evaluator dispatch",
            "grid_source": "mstm_reference" if ref_meta is not None else "pyceles_slice_config",
            "nonfinite_values": {
                "E_par": _count_nonfinite_complex(e_par),
                "H_par": _count_nonfinite_complex(h_par),
                "E_perp": _count_nonfinite_complex(e_perp),
                "H_perp": _count_nonfinite_complex(h_perp),
            },
        }

    return {
        "channels": channels,
        "unpolarized": {
            "R": float(0.5 * (channels["te"]["R"] + channels["tm"]["R"])),
            "T": float(0.5 * (channels["te"]["T"] + channels["tm"]["T"])),
            "A": float(0.5 * (channels["te"]["A"] + channels["tm"]["A"])),
        },
        "nearfield": nearfield_artifacts,
        "nearfield_bmax_policy": nearfield_bmax_policy,
        "periodic_method": cfg.pyceles_periodic_method,
        "mapping_convention": {"parallel": "tm", "perpendicular": "te"},
    }


def run_case(case: PeriodicBenchmarkCase, cfg: RunConfig) -> dict[str, Any]:
    workdir_case = cfg.workdir / case.name
    outdir_case = cfg.outdir / case.name
    workdir_case.mkdir(parents=True, exist_ok=True)
    outdir_case.mkdir(parents=True, exist_ok=True)

    _write_geometry_snapshot(outdir_case / "geometry_snapshot.txt", case)

    main_inp = workdir_case / "mstm_periodic_main.inp"
    main_out = workdir_case / "mstm_periodic_main.dat"
    _write_mstm_input(
        case,
        cfg,
        main_inp,
        main_out.name,
        calculate_scattering_matrix=True,
        nearfield=None,
    )

    nf_jobs: list[tuple[str, NearFieldSliceConfig]] = []
    if cfg.enable_nearfield_xy and case.nearfield_xy is not None:
        nf_jobs.append(("xy", case.nearfield_xy))
    if cfg.enable_nearfield_xz and case.nearfield_xz is not None:
        nf_jobs.append(("xz", case.nearfield_xz))

    if not cfg.parse_only_mstm:
        if cfg.mstm_exe is None:
            raise ValueError("mstm_exe is required unless parse_only_mstm is true.")
        proc = _run_mstm(cfg.mstm_exe, main_inp, workdir_case)
        if proc.returncode != 0:
            raise RuntimeError(
                f"MSTM main run failed.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        for tag, nf_cfg in nf_jobs:
            nf_inp = workdir_case / f"mstm_periodic_nf_{tag}.inp"
            nf_out = workdir_case / f"mstm_periodic_nf_{tag}.dat"
            nf_data = workdir_case / f"mstm_periodic_nf_{tag}_field.dat"
            _write_mstm_input(
                case,
                cfg,
                nf_inp,
                nf_out.name,
                calculate_scattering_matrix=False,
                nearfield=nf_cfg,
                nearfield_output_filename=nf_data.name,
            )
            proc_nf = _run_mstm(cfg.mstm_exe, nf_inp, workdir_case)
            if proc_nf.returncode != 0:
                raise RuntimeError(
                    f"MSTM near-field run ({tag}) failed.\n"
                    f"stdout:\n{proc_nf.stdout}\n\n"
                    f"stderr:\n{proc_nf.stderr}"
                )

    main_output = _parse_mstm_output(main_out)
    scattering_npz = outdir_case / "scattering_orders.npz"
    _save_periodic_scattering_npz(scattering_npz, main_output["periodic_scattering"])

    nearfield_artifacts: dict[str, dict[str, Any]] = {}
    nearfield_reference: dict[str, dict[str, Any]] = {}
    for tag, _ in nf_jobs:
        nf_data = workdir_case / f"mstm_periodic_nf_{tag}_field.dat"
        nf_out = workdir_case / f"mstm_periodic_nf_{tag}.dat"
        nf_parsed = _parse_mstm_nearfield_output(nf_data)
        nf_npz = outdir_case / f"nearfield_{tag}.npz"
        _save_nearfield_npz(nf_npz, nf_parsed, case.vacuum_wavelength)
        nf_meta = _parse_mstm_output(nf_out)
        nearfield_reference[tag] = {
            "coords_physical": _mstm_dimless_to_physical(
                nf_parsed["coords_dimless"], case.vacuum_wavelength
            ),
            "grid_dims": np.asarray(nf_parsed["grid_dims"], dtype=np.int64).tolist(),
            "grid_min_dimless": np.asarray(nf_parsed["grid_min_dimless"], dtype=float).tolist(),
            "grid_max_dimless": np.asarray(nf_parsed["grid_max_dimless"], dtype=float).tolist(),
        }
        nearfield_artifacts[tag] = {
            "npz": str(nf_npz),
            "run_number": int(nf_parsed["run_number"]),
            "grid_dims": [int(v) for v in np.asarray(nf_parsed["grid_dims"], dtype=int)],
            "n_points": int(np.prod(np.asarray(nf_parsed["grid_dims"], dtype=int))),
            "main_output_excerpt": {
                key: value for key, value in nf_meta.items() if key != "periodic_scattering"
            },
        }

    pyceles_output = None
    if cfg.run_pyceles:
        pyceles_nearfield_bmax, pyceles_nearfield_bmax_policy = _resolve_pyceles_nearfield_bmax(
            case, cfg
        )
        pyceles_output = _run_pyceles_case(
            case,
            outdir_case,
            cfg=cfg,
            nearfield_bmax=pyceles_nearfield_bmax,
            nearfield_bmax_policy=pyceles_nearfield_bmax_policy,
            nearfield_reference=nearfield_reference,
        )

    summary = _build_summary(
        case,
        cfg,
        main_output,
        scattering_npz,
        nearfield_artifacts,
        pyceles_output,
        workdir_case,
        outdir_case,
    )
    summary_path = outdir_case / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _compare_and_plot_bipanel_nearfield(
    *,
    outdir_case: Path,
    generate_plots: bool,
) -> Path:
    summary_path = outdir_case / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mstm_nf = summary.get("mstm", {}).get("nearfield", {})
    py_nf = summary.get("pyceles", {}).get("nearfield", {})
    if not isinstance(mstm_nf, dict) or not isinstance(py_nf, dict):
        raise RuntimeError("Missing MSTM/pyceles nearfield payload in benchmark summary.")

    report: dict[str, Any] = {
        "case": str(summary.get("case", {}).get("name", "")),
        "reference": "mstm",
        "notes": [
            "Pyceles nearfield payload uses automatic periodic exterior/interior dispatch.",
            "Pyceles nearfield maps are evaluated on the exact MSTM coordinates.",
        ],
    }
    plot_dir = outdir_case / "bipanel_maps"

    for tag in ("xy", "xz"):
        m_meta = mstm_nf.get(tag)
        p_meta = py_nf.get(tag)
        if not isinstance(m_meta, dict) or not isinstance(p_meta, dict):
            continue
        if "npz" not in m_meta:
            continue
        if "npz" not in p_meta:
            report[tag] = {
                "status": str(p_meta.get("status", "missing")),
                "reason": str(p_meta.get("reason", "")),
            }
            continue
        m_path = Path(str(m_meta["npz"]))
        p_path = Path(str(p_meta["npz"]))
        m = np.load(m_path)
        p = np.load(p_path)
        ref_points = np.asarray(m["coords_physical"], dtype=float)
        dims_arr = np.asarray(m["grid_dims"], dtype=np.int64).reshape(3)
        dims: tuple[int, int, int] = (int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2]))
        axis_h, axis_v, axis_h_label, axis_v_label = _plane_axes_from_reference(m)
        keys = ("E_par", "E_perp", "H_par", "H_perp")

        block: dict[str, Any] = {
            "mstm_points": int(ref_points.shape[0]),
            "pyceles_points": int(np.asarray(p["coords_physical"]).shape[0]),
            "pyceles_vs_mstm": {},
            "plots": {},
        }
        for key in keys:
            m_vec = np.asarray(m[key], dtype=np.complex128)
            p_points = np.asarray(p["coords_physical"], dtype=float)
            if p_points.shape != ref_points.shape or not np.allclose(
                p_points, ref_points, rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(
                    f"Pyceles near-field grid for {tag} does not match the MSTM reference grid; "
                    "benchmark comparison requires identical coordinates."
                )
            p_dims_arr = np.asarray(p["grid_dims"], dtype=np.int64).reshape(3)
            p_dims = (int(p_dims_arr[0]), int(p_dims_arr[1]), int(p_dims_arr[2]))
            if p_dims != dims:
                raise RuntimeError(
                    f"Pyceles near-field grid dims for {tag} do not match the MSTM reference dims: {p_dims} vs {dims}."
                )
            p_vec = np.asarray(p[key], dtype=np.complex128)

            block["pyceles_vs_mstm"][key] = _complex_similarity_metrics(m_vec, p_vec)

            m_int = _vector_intensity_plane(m_vec, dims)
            p_int = _vector_intensity_plane(p_vec, dims)
            if generate_plots:
                out_path = plot_dir / f"{tag}_{key}_intensity_bipanel.png"
                _plot_bipanel_intensity(
                    axis_h=axis_h,
                    axis_v=axis_v,
                    mstm_values=m_int,
                    pyceles_values=p_int,
                    out_path=out_path,
                    title=f"{tag.upper()} {key} intensity",
                    axis_h_label=axis_h_label,
                    axis_v_label=axis_v_label,
                )
                block["plots"][key] = str(out_path)
        report[tag] = block

    report_path = outdir_case / "map_level_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MSTM periodic 15-sphere benchmark and extract R/T/A, reciprocal-order scattering, and near-field slices."
    )
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        default="fifteen_spheres_homogeneous",
        help="Benchmark case to run.",
    )
    parser.add_argument(
        "--mstm-exe",
        type=Path,
        default=None,
        help="Path to the MSTM executable. Required unless --parse-only-mstm is used.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("outputs/mstm_periodic"),
        help="Directory where input/output files for raw MSTM runs will be stored.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/mstm_periodic_benchmark"),
        help="Directory where parsed benchmark artifacts will be stored.",
    )
    parser.add_argument(
        "--solution-epsilon",
        type=float,
        default=1.0e-8,
        help="MSTM solution_epsilon.",
    )
    parser.add_argument(
        "--mie-epsilon",
        type=float,
        default=None,
        help="Optional MSTM mie_epsilon override. Defaults to -lmax.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="MSTM max_iterations.",
    )
    parser.add_argument(
        "--medium-n-real",
        type=float,
        default=None,
        help="Optional homogeneous host refractive index override (real, positive).",
    )
    parser.add_argument(
        "--parse-only-mstm",
        action="store_true",
        help="Skip launching MSTM and only parse existing files in --workdir.",
    )
    parser.add_argument(
        "--skip-nearfield-xy",
        action="store_true",
        help="Do not run or parse the default xy near-field slice.",
    )
    parser.add_argument(
        "--skip-nearfield-xz",
        action="store_true",
        help="Do not run or parse the default xz near-field slice.",
    )
    parser.add_argument(
        "--skip-pyceles",
        action="store_true",
        help="Skip the pyceles periodic run and write MSTM-only artifacts.",
    )
    parser.add_argument(
        "--pyceles-nearfield-bmax",
        type=float,
        default=None,
        help=(
            "Optional pyceles periodic near-field reciprocal-radius cutoff. "
            "When omitted, the benchmark derives one from --pyceles-field-evanescent-decay."
        ),
    )
    parser.add_argument(
        "--pyceles-field-evanescent-decay",
        type=float,
        default=8.0,
        help=(
            "Target evanescent decay used to derive pyceles near-field bmax "
            "when --pyceles-nearfield-bmax is omitted."
        ),
    )
    parser.add_argument(
        "--pyceles-periodic-method",
        choices=("ewald", "rayleigh"),
        default="ewald",
        help="Periodic coupling method used by the pyceles comparison run.",
    )
    parser.add_argument(
        "--plot-map-panels",
        action="store_true",
        help="Generate MSTM-vs-pyceles nearfield intensity panel plots.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.parse_only_mstm and args.mstm_exe is None:
        parser.error("--mstm-exe is required unless --parse-only-mstm is used.")
    if args.pyceles_nearfield_bmax is not None and float(args.pyceles_nearfield_bmax) <= 0.0:
        parser.error("--pyceles-nearfield-bmax must be > 0 when provided.")
    if float(args.pyceles_field_evanescent_decay) < 0.0:
        parser.error("--pyceles-field-evanescent-decay must be >= 0.")

    case = CASES[args.case]
    if args.medium_n_real is not None:
        n_host = float(args.medium_n_real)
        if n_host <= 0.0:
            parser.error("--medium-n-real must be > 0.")
        case = replace(case, medium_refractive_index=complex(n_host, 0.0))
    cfg = RunConfig(
        mstm_exe=args.mstm_exe,
        workdir=args.workdir,
        outdir=args.outdir,
        parse_only_mstm=bool(args.parse_only_mstm),
        enable_nearfield_xy=not bool(args.skip_nearfield_xy),
        enable_nearfield_xz=not bool(args.skip_nearfield_xz),
        solution_epsilon=float(args.solution_epsilon),
        mie_epsilon=None if args.mie_epsilon is None else float(args.mie_epsilon),
        max_iterations=int(args.max_iterations),
        run_pyceles=not bool(args.skip_pyceles),
        pyceles_nearfield_bmax=(
            None if args.pyceles_nearfield_bmax is None else float(args.pyceles_nearfield_bmax)
        ),
        pyceles_field_evanescent_decay=float(args.pyceles_field_evanescent_decay),
        pyceles_periodic_method=args.pyceles_periodic_method,
    )
    summary = run_case(case, cfg)
    map_report_path: Path | None = None
    if bool(args.plot_map_panels) and not bool(args.skip_pyceles):
        map_report_path = _compare_and_plot_bipanel_nearfield(
            outdir_case=cfg.outdir / case.name,
            generate_plots=True,
        )
    payload = {
        "summary_json": str(Path(summary["mstm"]["outdir"]) / "summary.json"),
        "rta": summary["mstm"].get("main", {}).get("rta"),
        "scattering_counts": {
            "backward": summary["mstm"]["periodic_scattering"]["backward_count"],
            "forward": summary["mstm"]["periodic_scattering"]["forward_count"],
        },
        "nearfield": summary["mstm"]["nearfield"],
    }
    if "pyceles" in summary:
        payload["pyceles"] = summary["pyceles"]
    if "comparison" in summary:
        payload["comparison"] = summary["comparison"]
    if map_report_path is not None:
        payload["map_parity_report_json"] = str(map_report_path)
        payload["map_parity_bipanel_dir"] = str(cfg.outdir / case.name / "bipanel_maps")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
