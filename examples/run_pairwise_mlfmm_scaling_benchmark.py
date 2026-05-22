"""Generate solve-time scaling data for pairwise and MLFMM coupling.

The script generates a random sphere cloud for each requested particle count,
runs the selected coupling backends on that same geometry, writes resumable
JSON records, and can emit a quick log-log timing plot. Cold CuPy runs can
include one-time kernel and library initialization overhead, so the per-phase
timings in the JSON are the preferred source for later analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from tqdm.auto import tqdm

import pyceles as pcl
from pyceles.core.indexing import n_modes
from pyceles.core.operators import MLFMMOptions

OperatorBackend = Literal["numpy", "cupy"]
CouplingBackend = Literal["pairwise", "mlfmm"]
SolverMethod = Literal["gmres", "fgmres", "bicgstab", "lgmres", "gcrotmk"]


@dataclass(frozen=True)
class GeneratedGeometry:
    positions: np.ndarray
    radii: np.ndarray
    refractive_indices: np.ndarray
    side_length: float
    target_volume_fraction: float
    actual_volume_fraction: float
    placement_attempts: int
    placement_time_s: float


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "pyceles").exists():
            return parent
    raise FileNotFoundError("Could not locate the pyceles repository root.")


def _parse_float_list(value: str, *, name: str) -> tuple[float, ...]:
    try:
        out = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated float list.") from exc
    if not out:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value.")
    if not all(np.isfinite(out)):
        raise argparse.ArgumentTypeError(f"{name} must contain only finite values.")
    return out


def _parse_coupling_list(value: str) -> tuple[CouplingBackend, ...]:
    allowed = {"pairwise", "mlfmm"}
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("couplings must contain at least one value.")
    bad = sorted(set(items).difference(allowed))
    if bad:
        raise argparse.ArgumentTypeError(f"Unsupported coupling backend(s): {bad}.")
    return tuple(cast(CouplingBackend, item) for item in items)


def _parse_power_range(value: str) -> tuple[int, int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("power range must have the form START,STOP.")
    try:
        start, stop = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("power range entries must be integers.") from exc
    if start < 0 or stop < start:
        raise argparse.ArgumentTypeError("power range must satisfy 0 <= START <= STOP.")
    return start, stop


def _n_values_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.n_values:
        try:
            values = tuple(
                int(item.strip()) for item in str(args.n_values).split(",") if item.strip()
            )
        except ValueError as exc:
            raise ValueError("--n-values must be a comma-separated integer list.") from exc
        if not values or any(v < 1 for v in values):
            raise ValueError("--n-values must contain positive integers.")
        return tuple(sorted(dict.fromkeys(values)))
    start, stop = cast(tuple[int, int], args.powers_of_two_range)
    return tuple(2**p for p in range(start, stop + 1))


def _cell_key(point: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    idx = np.floor(point / float(cell_size)).astype(np.int64)
    return (int(idx[0]), int(idx[1]), int(idx[2]))


def _palette_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        str(value): int(count)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def _generate_geometry(
    *,
    n_particles: int,
    radii_palette: tuple[float, ...],
    real_n_palette: tuple[float, ...],
    imag_n_palette: tuple[float, ...],
    target_volume_fraction: float,
    seed: int,
    max_attempts_per_particle: int,
    show_progress: bool,
) -> GeneratedGeometry:
    """Generate a deterministic finite cube of non-overlapping spheres."""
    n = int(n_particles)
    if n < 1:
        raise ValueError("n_particles must be positive.")
    if not (0.0 < float(target_volume_fraction) < 0.4):
        raise ValueError("target_volume_fraction must lie in (0, 0.4) for this RSA generator.")
    if any(r <= 0.0 for r in radii_palette):
        raise ValueError("all radii must be positive.")
    if int(max_attempts_per_particle) < 1:
        raise ValueError("max_attempts_per_particle must be positive.")

    rng = np.random.default_rng(int(seed) + 1_000_003 * n)
    radii = np.asarray(rng.choice(np.asarray(radii_palette, dtype=float), size=n), dtype=float)
    real_n = np.asarray(rng.choice(np.asarray(real_n_palette, dtype=float), size=n), dtype=float)
    imag_n = np.asarray(rng.choice(np.asarray(imag_n_palette, dtype=float), size=n), dtype=float)
    refractive_indices = real_n.astype(np.complex128) + 1j * imag_n.astype(np.complex128)

    particle_volume = float(np.sum((4.0 * np.pi / 3.0) * radii**3))
    side = float((particle_volume / float(target_volume_fraction)) ** (1.0 / 3.0))
    r_max = float(np.max(radii))
    if side <= 2.0 * r_max:
        raise ValueError(
            "Generated cube is too small to contain the largest sphere. "
            "Reduce target volume fraction or radii."
        )

    order = np.argsort(-radii, kind="stable")
    positions = np.empty((n, 3), dtype=float)
    cell_size = 2.0 * r_max
    cells: dict[tuple[int, int, int], list[int]] = {}
    offsets = tuple(
        (dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3)
    )
    attempts = 0
    t0 = time.perf_counter()

    iterator = order
    progress = None
    if show_progress:
        progress = tqdm(total=n, desc=f"RSA place N={n}", unit="sphere")
    try:
        for orig_idx in iterator:
            radius = float(radii[int(orig_idx)])
            placed = False
            for _ in range(int(max_attempts_per_particle)):
                attempts += 1
                candidate = rng.uniform(radius, side - radius, size=3)
                key = _cell_key(candidate, cell_size)
                ok = True
                for ox, oy, oz in offsets:
                    neighbor_key = (key[0] + ox, key[1] + oy, key[2] + oz)
                    for other_idx in cells.get(neighbor_key, ()):
                        cutoff = radius + float(radii[other_idx])
                        delta = candidate - positions[other_idx]
                        if float(np.dot(delta, delta)) < cutoff * cutoff:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    positions[int(orig_idx)] = candidate
                    cells.setdefault(key, []).append(int(orig_idx))
                    placed = True
                    if progress is not None:
                        progress.update(1)
                    break
            if not placed:
                raise RuntimeError(
                    "RSA placement failed before all particles were inserted. "
                    f"N={n}, placed={sum(len(v) for v in cells.values())}, "
                    f"radius={radius:g}, target_volume_fraction={target_volume_fraction:g}. "
                    "Try a lower volume fraction or a larger attempt budget."
                )
    finally:
        if progress is not None:
            progress.close()

    positions -= 0.5 * side
    placement_time = time.perf_counter() - t0
    actual_phi = particle_volume / side**3
    return GeneratedGeometry(
        positions=np.ascontiguousarray(positions),
        radii=np.ascontiguousarray(radii),
        refractive_indices=np.ascontiguousarray(refractive_indices),
        side_length=side,
        target_volume_fraction=float(target_volume_fraction),
        actual_volume_fraction=float(actual_phi),
        placement_attempts=int(attempts),
        placement_time_s=float(placement_time),
    )


def _geometry_summary(geometry: GeneratedGeometry) -> dict[str, Any]:
    return {
        "side_length": float(geometry.side_length),
        "target_volume_fraction": float(geometry.target_volume_fraction),
        "actual_volume_fraction": float(geometry.actual_volume_fraction),
        "placement_attempts": int(geometry.placement_attempts),
        "placement_time_s": float(geometry.placement_time_s),
        "radius_counts": _palette_counts(geometry.radii),
        "refractive_index_counts": _palette_counts(geometry.refractive_indices),
        "bounds_min": [float(v) for v in np.min(geometry.positions, axis=0).tolist()],
        "bounds_max": [float(v) for v in np.max(geometry.positions, axis=0).tolist()],
    }


def _make_source(args: argparse.Namespace) -> pcl.PlaneWave:
    return pcl.PlaneWave(
        wavelength=float(args.wavelength),
        medium_n=float(args.n_medium) + 0j,
        polarization=cast(Literal["TE", "TM"], args.polarization),
        polar_angle=float(args.polar_angle),
        azimuthal_angle=float(args.azimuthal_angle),
        amplitude=1.0,
    )


def _mlfmm_options(args: argparse.Namespace) -> MLFMMOptions:
    return MLFMMOptions(
        max_leaf_particles=int(args.mlfmm_max_leaf_particles),
        max_depth=int(args.mlfmm_max_depth),
        leaf_size_radius_factor=float(args.mlfmm_leaf_size_radius_factor),
        accuracy_level=int(args.mlfmm_accuracy_level),
        order_additive=int(args.mlfmm_order_additive),
    )


def _benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "wavelength": float(args.wavelength),
        "n_medium": float(args.n_medium),
        "lmax": int(args.lmax),
        "polarization": str(args.polarization),
        "polar_angle": float(args.polar_angle),
        "azimuthal_angle": float(args.azimuthal_angle),
        "operator_backend": str(args.operator_backend),
        "solver": str(args.solver),
        "solver_rtol": float(args.solver_rtol),
        "solver_restart": int(args.solver_restart),
        "solver_maxiter": int(args.solver_maxiter),
        "solver_compute_final_residual": not bool(args.skip_final_residual_check),
        "compute_dtype": str(args.compute_dtype),
        "accum_dtype": str(args.accum_dtype),
        "radial_lut_dr": float(args.radial_lut_dr),
        "target_volume_fraction": float(args.target_volume_fraction),
        "radii": [float(v) for v in cast(tuple[float, ...], args.radii)],
        "real_n": [float(v) for v in cast(tuple[float, ...], args.real_n)],
        "imag_n": [float(v) for v in cast(tuple[float, ...], args.imag_n)],
        "seed": int(args.seed),
        "max_attempts_per_particle": int(args.max_attempts_per_particle),
        "mlfmm_options": {
            "max_leaf_particles": int(args.mlfmm_max_leaf_particles),
            "max_depth": int(args.mlfmm_max_depth),
            "leaf_size_radius_factor": float(args.mlfmm_leaf_size_radius_factor),
            "accuracy_level": int(args.mlfmm_accuracy_level),
            "order_additive": int(args.mlfmm_order_additive),
        },
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _backend_metadata(backend: OperatorBackend) -> dict[str, Any]:
    metadata: dict[str, Any] = {"numpy_version": np.__version__}
    if backend != "cupy":
        return metadata
    try:
        import cupy as cp

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props.get("name", b"unknown")
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        metadata.update(
            {
                "cupy_version": cp.__version__,
                "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "cuda_device_id": int(device.id),
                "cuda_device_name": str(name),
                "cuda_device_total_memory": int(props.get("totalGlobalMem", 0)),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic metadata only.
        metadata["cupy_metadata_error"] = repr(exc)
    return metadata


def _run_file(
    out_dir: Path, n_particles: int, coupling: CouplingBackend, backend: OperatorBackend
) -> Path:
    return out_dir / f"N_{int(n_particles):08d}_{backend}_{coupling}.json"


def _existing_payload(path: Path, expected_fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(
            f"Existing result has a different configuration fingerprint: {path}. "
            "Use a different --out-dir or --force."
        )
    if payload.get("status") != "completed":
        return None
    return cast(dict[str, Any], payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_one(
    *,
    args: argparse.Namespace,
    geometry: GeneratedGeometry,
    coupling: CouplingBackend,
    fingerprint: str,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    particles = pcl.core.spheres_from_arrays(
        positions=geometry.positions,
        radii=geometry.radii,
        refractive_indices=geometry.refractive_indices,
    )
    source = _make_source(args)
    cfg = pcl.SimulationConfig(
        wavelength=float(args.wavelength),
        n_medium=float(args.n_medium) + 0j,
        lmax=int(args.lmax),
        source=source,
        solver_method=cast(SolverMethod, args.solver),
        solver_rtol=float(args.solver_rtol),
        solver_restart=int(args.solver_restart),
        solver_maxiter=int(args.solver_maxiter),
        solver_direct_max_n=1,
        solver_compute_final_residual=not bool(args.skip_final_residual_check),
        operator_backend=cast(OperatorBackend, args.operator_backend),
        coupling_backend=coupling,
        mlfmm_options=_mlfmm_options(args) if coupling == "mlfmm" else None,
        postprocessing_backend="numpy",
        compute_dtype=cast(Literal["complex64", "complex128"], args.compute_dtype),
        accum_dtype=cast(Literal["complex64", "complex128"], args.accum_dtype),
        radial_lut_dr=float(args.radial_lut_dr),
        check_circumscribing_sphere_overlap=bool(args.check_geometry),
        verbose=not bool(args.quiet),
    )
    sim = pcl.Simulation(cfg, particles=particles)
    start = time.perf_counter()
    run = sim.run(include_farfield=False)
    wall_time = time.perf_counter() - start
    solver = run.solver_result
    phase_timings = {}
    metadata = solver.block_metadata or {}
    raw_timings = metadata.get("simulation_phase_timings_s")
    if isinstance(raw_timings, dict):
        phase_timings = {str(key): float(value) for key, value in raw_timings.items()}
    payload = {
        "status": "completed",
        "fingerprint": fingerprint,
        "config": run_config,
        "n_particles": int(geometry.radii.size),
        "unknowns": int(geometry.radii.size) * n_modes(int(args.lmax)),
        "operator_backend": str(args.operator_backend),
        "coupling_backend": coupling,
        "solve_wall_time_s": float(wall_time),
        "solver_result": {
            "method": str(solver.method),
            "iterations": int(np.asarray(solver.iterations)),
            "info": int(np.asarray(solver.info)),
            "relative_residual": float(np.asarray(solver.relative_residual)),
            "residual_norm": float(np.asarray(solver.residual_norm)),
        },
        "phase_timings_s": phase_timings,
        "geometry": _geometry_summary(geometry),
    }
    return payload


def _write_summary(
    *,
    out_dir: Path,
    status: str,
    args: argparse.Namespace,
    benchmark_config: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    summary = {
        "status": status,
        "script": Path(__file__).name,
        "command": list(sys.argv),
        "created_at_unix_s": float(time.time()),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "backend": _backend_metadata(cast(OperatorBackend, args.operator_backend)),
        "pyceles_version": str(getattr(pcl, "__version__", "unknown")),
        "n_values": [int(v) for v in _n_values_from_args(args)],
        "couplings": list(cast(tuple[CouplingBackend, ...], args.couplings)),
        "config": benchmark_config,
        "results": rows,
    }
    _atomic_write_json(out_dir / "summary.json", summary)


def _plot_results(out_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {"pairwise": "tab:blue", "mlfmm": "tab:orange"}
    for coupling in ("pairwise", "mlfmm"):
        group = [row for row in completed if row.get("coupling_backend") == coupling]
        if not group:
            continue
        group.sort(key=lambda row: int(row["n_particles"]))
        x = np.asarray([int(row["n_particles"]) for row in group], dtype=float)
        total_time = np.asarray([float(row["solve_wall_time_s"]) for row in group], dtype=float)
        krylov_time = np.asarray(
            [
                float(
                    cast(dict[str, Any], row.get("phase_timings_s", {})).get(
                        "linear_solve_s", row["solve_wall_time_s"]
                    )
                )
                for row in group
            ],
            dtype=float,
        )
        color = colors[coupling]
        ax.loglog(
            x,
            krylov_time,
            marker="o",
            color=color,
            label=f"{coupling} Krylov",
        )
        ax.loglog(
            x,
            total_time,
            marker="s",
            linestyle="--",
            color=color,
            alpha=0.8,
            label=f"{coupling} total",
        )
        if coupling == "mlfmm":
            prep_time = np.asarray(
                [
                    float(
                        cast(dict[str, Any], row.get("phase_timings_s", {})).get(
                            "prepare_operator_s", np.nan
                        )
                    )
                    for row in group
                ],
                dtype=float,
            )
            finite = np.isfinite(prep_time) & (prep_time > 0)
            if np.any(finite):
                ax.loglog(
                    x[finite],
                    prep_time[finite],
                    marker="^",
                    linestyle=":",
                    color=color,
                    alpha=0.9,
                    label="MLFMM prep (host + device)",
                )
                upper = np.maximum(krylov_time, total_time)
                lower = np.minimum(krylov_time, total_time)
                ax.fill_between(x, lower, upper, color=color, alpha=0.10, linewidth=0)
    ax.set_xlabel("Particles N")
    ax.set_ylabel("Time [s]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "pairwise_mlfmm_scaling.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run solve-only pairwise-vs-MLFMM scaling sweeps for random sphere clouds."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/pairwise_mlfmm_scaling"))
    parser.add_argument("--powers-of-two-range", type=_parse_power_range, default=(10, 13))
    parser.add_argument("--n-values", type=str, default="")
    parser.add_argument("--couplings", type=_parse_coupling_list, default=("pairwise", "mlfmm"))
    parser.add_argument("--operator-backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--wavelength", type=float, default=550.0)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--polarization", choices=("TE", "TM"), default="TE")
    parser.add_argument("--polar-angle", type=float, default=0.0)
    parser.add_argument("--azimuthal-angle", type=float, default=0.0)
    parser.add_argument(
        "--solver", choices=("gmres", "fgmres", "bicgstab", "lgmres", "gcrotmk"), default="bicgstab"
    )
    parser.add_argument("--solver-rtol", type=float, default=1e-4)
    parser.add_argument("--solver-restart", type=int, default=50)
    parser.add_argument("--solver-maxiter", type=int, default=1000)
    parser.add_argument("--skip-final-residual-check", action="store_true")
    parser.add_argument("--compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--accum-dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--radial-lut-dr", type=float, default=1.0)
    parser.add_argument("--target-volume-fraction", type=float, default=0.10)
    parser.add_argument(
        "--radii",
        type=lambda value: _parse_float_list(value, name="radii"),
        default=(90.0, 100.0, 110.0),
    )
    parser.add_argument(
        "--real-n",
        type=lambda value: _parse_float_list(value, name="real-n"),
        default=(1.6, 1.7, 1.8),
    )
    parser.add_argument(
        "--imag-n",
        type=lambda value: _parse_float_list(value, name="imag-n"),
        default=(0.0, 0.01, 0.02),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-attempts-per-particle", type=int, default=2000)
    parser.add_argument("--mlfmm-max-leaf-particles", type=int, default=8)
    parser.add_argument("--mlfmm-max-depth", type=int, default=12)
    parser.add_argument("--mlfmm-leaf-size-radius-factor", type=float, default=4.0)
    parser.add_argument("--mlfmm-accuracy-level", type=int, default=3)
    parser.add_argument("--mlfmm-order-additive", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-geometry", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_config = _benchmark_config(args)
    rows: list[dict[str, Any]] = []
    status = "completed"

    try:
        for n_particles in _n_values_from_args(args):
            pending: list[tuple[CouplingBackend, str, dict[str, Any], Path]] = []
            for coupling in cast(tuple[CouplingBackend, ...], args.couplings):
                run_config = {
                    **benchmark_config,
                    "n_particles": int(n_particles),
                    "coupling_backend": coupling,
                }
                fingerprint = _fingerprint(run_config)
                path = _run_file(
                    out_dir,
                    int(n_particles),
                    coupling,
                    cast(OperatorBackend, args.operator_backend),
                )
                existing = None if bool(args.force) else _existing_payload(path, fingerprint)
                if existing is not None:
                    if not bool(args.quiet):
                        print(f"Skipping existing result: {path}")
                    rows.append(existing)
                    continue
                pending.append((coupling, fingerprint, run_config, path))
            if not pending:
                _write_summary(
                    out_dir=out_dir,
                    status=status,
                    args=args,
                    benchmark_config=benchmark_config,
                    rows=rows,
                )
                continue

            geometry = _generate_geometry(
                n_particles=int(n_particles),
                radii_palette=cast(tuple[float, ...], args.radii),
                real_n_palette=cast(tuple[float, ...], args.real_n),
                imag_n_palette=cast(tuple[float, ...], args.imag_n),
                target_volume_fraction=float(args.target_volume_fraction),
                seed=int(args.seed),
                max_attempts_per_particle=int(args.max_attempts_per_particle),
                show_progress=not bool(args.quiet),
            )
            for coupling, fingerprint, run_config, path in pending:
                if not bool(args.quiet):
                    print(
                        f"Running N={n_particles} coupling={coupling} "
                        f"backend={args.operator_backend}"
                    )
                payload = _run_one(
                    args=args,
                    geometry=geometry,
                    coupling=coupling,
                    fingerprint=fingerprint,
                    run_config=run_config,
                )
                _atomic_write_json(path, payload)
                rows.append(payload)
                _write_summary(
                    out_dir=out_dir,
                    status=status,
                    args=args,
                    benchmark_config=benchmark_config,
                    rows=rows,
                )
    except KeyboardInterrupt:
        status = "interrupted"
        _write_summary(
            out_dir=out_dir,
            status=status,
            args=args,
            benchmark_config=benchmark_config,
            rows=rows,
        )
        print(f"Interrupted. Wrote partial summary to {out_dir / 'summary.json'}")
        return

    if bool(args.plot):
        plot_path = _plot_results(out_dir, rows)
        if plot_path is not None:
            print(f"Wrote scaling plot: {plot_path}")
    _write_summary(
        out_dir=out_dir,
        status=status,
        args=args,
        benchmark_config=benchmark_config,
        rows=rows,
    )
    print(f"Wrote scaling summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
