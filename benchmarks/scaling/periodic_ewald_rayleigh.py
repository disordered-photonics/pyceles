"""Generate periodic Ewald/Rayleigh solve-time scaling data.

The ``lateral`` family keeps occupied height fixed while increasing the square
period; the ``vertical`` family keeps the square period fixed while increasing
height.  Each requested cell is generated deterministically, preflighted for
Rayleigh/Wood clearance, and written as a resumable JSON record.  The timed
quantity is complete wall time to the first converged solution, including
operator preparation and the Krylov solve; phase timings and iteration counts
remain available for interpretation.

CuPy is the default production backend and uses complex64 compute with
complex128 accumulation.  NumPy remains available only when explicitly
requested for small reference probes; no CPU timing is extrapolated from GPU
results.  Cached Ewald is the practical dense reference while it fits in
device memory.  Pass ``--no-cache-translation-blocks`` for the bounded-memory
matrix-free Ewald diagnostic.  Rayleigh uses its compact reciprocal plan and
sparse exact-near data.  GMRES is the standard solver; GCRO-DR and LSQR are
retained as explicit expert diagnostics.  Every recorded solve performs its
terminal physical residual check; the benchmark no longer exposes the old
proxy-only stopping switch.

The defaults are the current production ladder: lateral growth at fixed
``H/lambda=20`` and vertical growth through ``N=16384`` at fixed
``P/lambda=4.75``.  Override the lists and geometry controls explicitly for
smaller probes.  The complete preflight runs before any timed solve and
``--anomaly-policy error`` is the default.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from pyceles.core.periodic import plane_wave_k_parallel, rayleigh_report

Backend = Literal["numpy", "cupy"]
PeriodicMethod = Literal["ewald", "rayleigh"]
GrowthFamily = Literal["lateral", "vertical"]
SolverMethod = Literal["gmres", "gcro", "lsqr"]


@dataclass(frozen=True)
class PeriodicGeometry:
    family: GrowthFamily
    positions: np.ndarray
    radii: np.ndarray
    refractive_indices: np.ndarray
    ax: float
    ay: float
    height: float
    actual_volume_fraction: float
    placement_attempts: int
    placement_time_s: float


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src" / "pyceles").exists():
            return parent
    raise FileNotFoundError("Could not locate the pyceles repository root.")


def _parse_names(value: str, *, allowed: set[str], name: str) -> tuple[str, ...]:
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value.")
    bad = sorted(set(items).difference(allowed))
    if bad:
        raise argparse.ArgumentTypeError(f"Unsupported {name}: {bad}.")
    return items


def _parse_float_list(value: str, *, name: str) -> tuple[float, ...]:
    try:
        items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated float list.") from exc
    if not items or not all(np.isfinite(items)):
        raise argparse.ArgumentTypeError(f"{name} must contain finite values.")
    return items


def _parse_int_list(value: str, *, name: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list.") from exc
    if not items or any(item < 1 for item in items):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers.")
    return tuple(sorted(dict.fromkeys(items)))


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _palette_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {
        str(value): int(count)
        for value, count in zip(unique.tolist(), counts.tolist(), strict=True)
    }


def _sync(backend: Backend) -> None:
    if backend == "cupy":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


def _free_backend_memory(backend: Backend) -> None:
    gc.collect()
    if backend == "cupy":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _is_memory_exhaustion(exc: BaseException) -> bool:
    """Return whether an exception identifies a backend memory exhaustion."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in {"MemoryError", "OutOfMemoryError"}:
            return True
        current = current.__cause__ or current.__context__
    return False


def _sample_particle_properties(
    *,
    n_particles: int,
    radii_palette: tuple[float, ...],
    real_n_palette: tuple[float, ...],
    imag_n_palette: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    # Match the finite scaling benchmark's deterministic palette sampling.
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(n_particles))
    radii = np.asarray(
        rng.choice(np.asarray(radii_palette, dtype=float), size=int(n_particles)), dtype=float
    )
    real_n = np.asarray(
        rng.choice(np.asarray(real_n_palette, dtype=float), size=int(n_particles)), dtype=float
    )
    imag_n = np.asarray(
        rng.choice(np.asarray(imag_n_palette, dtype=float), size=int(n_particles)), dtype=float
    )
    refractive_indices = real_n.astype(np.complex128) + 1j * imag_n.astype(np.complex128)
    particle_volume = float(np.sum((4.0 * np.pi / 3.0) * radii**3))
    return np.ascontiguousarray(radii), np.ascontiguousarray(refractive_indices), particle_volume


def _box_dimensions(
    *,
    family: GrowthFamily,
    particle_volume: float,
    target_volume_fraction: float,
    fixed_length: float,
) -> tuple[float, float, float]:
    phi = float(target_volume_fraction)
    if family == "lateral":
        height = float(fixed_length)
        side = math.sqrt(float(particle_volume) / (phi * height))
    else:
        side = float(fixed_length)
        height = float(particle_volume) / (phi * side**2)
    return float(side), float(side), float(height)


def _place_periodic_particles(
    *,
    radii: np.ndarray,
    ax: float,
    ay: float,
    height: float,
    seed: int,
    family: GrowthFamily,
    max_attempts_per_particle: int,
    show_progress: bool,
) -> tuple[np.ndarray, int, float]:
    """Periodic RSA using minimum-image x/y overlap checks."""
    radii = np.asarray(radii, dtype=float).reshape(-1)
    n_particles = int(radii.size)
    rmax = float(np.max(radii))
    if min(float(ax), float(ay), float(height)) <= 2.0 * rmax:
        raise ValueError(
            "Generated periodic box is too small for the largest sphere. "
            "Increase N/reference size or reduce the radius palette."
        )

    cell_size = 2.0 * rmax
    nx = max(1, math.floor(float(ax) / cell_size))
    ny = max(1, math.floor(float(ay) / cell_size))
    nz = max(1, math.floor(float(height) / cell_size))
    cells: dict[tuple[int, int, int], list[int]] = {}
    positions = np.empty((n_particles, 3), dtype=float)
    order = np.argsort(-radii, kind="stable")
    family_offset = 0 if family == "lateral" else 918_273_645
    rng = np.random.default_rng(int(seed) + 7_919 * n_particles + family_offset)
    attempts = 0
    started = time.perf_counter()

    def cell_key(point: np.ndarray) -> tuple[int, int, int]:
        ix = math.floor(((point[0] + 0.5 * ax) / ax) * nx) % nx
        iy = math.floor(((point[1] + 0.5 * ay) / ay) * ny) % ny
        iz = min(
            nz - 1,
            max(0, math.floor(((point[2] + 0.5 * height) / height) * nz)),
        )
        return ix, iy, iz

    progress = (
        tqdm(total=n_particles, desc=f"Periodic RSA {family} N={n_particles}", unit="sphere")
        if show_progress
        else None
    )
    try:
        for original_index in order:
            idx = int(original_index)
            radius = float(radii[idx])
            zlo = -0.5 * height + radius
            zhi = +0.5 * height - radius
            placed = False
            for _ in range(int(max_attempts_per_particle)):
                attempts += 1
                candidate = np.array(
                    [
                        rng.uniform(-0.5 * ax, 0.5 * ax),
                        rng.uniform(-0.5 * ay, 0.5 * ay),
                        rng.uniform(zlo, zhi),
                    ],
                    dtype=float,
                )
                key = cell_key(candidate)
                neighbor_keys = {
                    ((key[0] + dx) % nx, (key[1] + dy) % ny, key[2] + dz)
                    for dx in (-1, 0, 1)
                    for dy in (-1, 0, 1)
                    for dz in (-1, 0, 1)
                    if 0 <= key[2] + dz < nz
                }
                ok = True
                for neighbor_key in neighbor_keys:
                    for other_idx in cells.get(neighbor_key, ()):
                        delta = candidate - positions[other_idx]
                        delta[0] -= ax * np.rint(delta[0] / ax)
                        delta[1] -= ay * np.rint(delta[1] / ay)
                        cutoff = radius + float(radii[other_idx])
                        if float(np.dot(delta, delta)) < cutoff * cutoff:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    positions[idx] = candidate
                    cells.setdefault(key, []).append(idx)
                    placed = True
                    if progress is not None:
                        progress.update(1)
                    break
            if not placed:
                raise RuntimeError(
                    f"Periodic RSA failed for family={family}, N={n_particles}; "
                    f"placed={sum(len(v) for v in cells.values())}."
                )
    finally:
        if progress is not None:
            progress.close()

    return np.ascontiguousarray(positions), attempts, time.perf_counter() - started


def _family_fixed_length(
    *,
    family: GrowthFamily,
    args: argparse.Namespace,
) -> float:
    ratio = (
        args.lateral_height_over_wavelength
        if family == "lateral"
        else args.vertical_period_over_wavelength
    )
    return float(ratio) * float(args.wavelength)


def _generate_geometry(
    *,
    family: GrowthFamily,
    n_particles: int,
    args: argparse.Namespace,
) -> PeriodicGeometry:
    radii, refractive_indices, particle_volume = _sample_particle_properties(
        n_particles=n_particles,
        radii_palette=cast(tuple[float, ...], args.radii),
        real_n_palette=cast(tuple[float, ...], args.real_n),
        imag_n_palette=cast(tuple[float, ...], args.imag_n),
        seed=int(args.seed),
    )
    fixed_length = _family_fixed_length(family=family, args=args)
    ax, ay, height = _box_dimensions(
        family=family,
        particle_volume=particle_volume,
        target_volume_fraction=float(args.target_volume_fraction),
        fixed_length=fixed_length,
    )
    positions, attempts, placement_time = _place_periodic_particles(
        radii=radii,
        ax=ax,
        ay=ay,
        height=height,
        seed=int(args.seed),
        family=family,
        max_attempts_per_particle=int(args.max_attempts_per_particle),
        show_progress=not bool(args.quiet),
    )
    return PeriodicGeometry(
        family=family,
        positions=positions,
        radii=radii,
        refractive_indices=refractive_indices,
        ax=ax,
        ay=ay,
        height=height,
        actual_volume_fraction=float(particle_volume / (ax * ay * height)),
        placement_attempts=int(attempts),
        placement_time_s=float(placement_time),
    )


def _geometry_summary(geometry: PeriodicGeometry, wavelength: float) -> dict[str, Any]:
    return {
        "family": geometry.family,
        "ax_nm": float(geometry.ax),
        "ay_nm": float(geometry.ay),
        "height_nm": float(geometry.height),
        "ax_over_wavelength": float(geometry.ax / wavelength),
        "height_over_wavelength": float(geometry.height / wavelength),
        "family_fixed_length_nm": float(
            geometry.height if geometry.family == "lateral" else geometry.ax
        ),
        "family_fixed_length_over_wavelength": float(
            (geometry.height if geometry.family == "lateral" else geometry.ax) / wavelength
        ),
        "actual_volume_fraction": float(geometry.actual_volume_fraction),
        "placement_attempts": int(geometry.placement_attempts),
        "placement_time_s": float(geometry.placement_time_s),
        "radius_counts": _palette_counts(geometry.radii),
        "refractive_index_counts": _palette_counts(geometry.refractive_indices),
    }


def _make_source(args: argparse.Namespace) -> pcl.PlaneWave:
    return pcl.PlaneWave(
        wavelength=float(args.wavelength),
        medium_n=complex(float(args.n_medium), 0.0),
        polarization=cast(Literal["TE", "TM"], args.polarization),
        polar_angle=float(args.polar_angle),
        azimuthal_angle=float(args.azimuthal_angle),
        amplitude=1.0,
    )


def _dtype_policy(args: argparse.Namespace, backend: Backend) -> tuple[str, str]:
    if backend == "cupy":
        return str(args.cupy_compute_dtype), str(args.cupy_accum_dtype)
    return str(args.numpy_compute_dtype), str(args.numpy_accum_dtype)


def _safety_summary_for_box(args: argparse.Namespace, *, ax: float, ay: float) -> dict[str, Any]:
    source = _make_source(args)
    lattice = pcl.RectangularLattice2D(ax=float(ax), ay=float(ay))
    k = 2.0 * math.pi * float(args.n_medium) / float(args.wavelength)
    report = rayleigh_report(
        lattice,
        k=k,
        k_parallel=plane_wave_k_parallel(source),
        include_specular=False,
    )
    return {
        "warning_level": str(report.warning_level),
        "min_clearance": float(report.min_clearance),
        "message": str(report.message),
        "nearest_orders": [[int(order.m), int(order.n)] for order in report.nearest_orders],
    }


def _safety_summary(args: argparse.Namespace, geometry: PeriodicGeometry) -> dict[str, Any]:
    return _safety_summary_for_box(args, ax=geometry.ax, ay=geometry.ay)


def _periodic_spec(
    args: argparse.Namespace, geometry: PeriodicGeometry, method: PeriodicMethod
) -> pcl.PeriodicSpec:
    return pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(ax=geometry.ax, ay=geometry.ay),
        options=pcl.PeriodicOptions(
            method=method,
            shell_tolerance=float(args.shell_tolerance),
            max_shells=int(args.max_shells),
            rayleigh_z_cut=(None if args.rayleigh_z_cut is None else float(args.rayleigh_z_cut)),
            rayleigh_reciprocal_shells=(
                None
                if args.rayleigh_reciprocal_shells is None
                else int(args.rayleigh_reciprocal_shells)
            ),
        ),
    )


def _operator_diagnostics(sim: pcl.Simulation, method: PeriodicMethod) -> dict[str, Any]:
    prepared = getattr(sim, "_prepared_operator_cache", None)
    coupling = None if prepared is None else getattr(prepared, "coupling", None)
    if coupling is None:
        return {}
    payload: dict[str, Any] = {}
    eta_method = getattr(coupling, "_ewald_eta", None)
    if callable(eta_method):
        payload["ewald_eta"] = float(eta_method())
    shell_method = getattr(coupling, "_shell_counts", None)
    if callable(shell_method):
        try:
            real_shells, reciprocal_shells = shell_method()
            payload["resolved_ewald_shells"] = [int(real_shells), int(reciprocal_shells)]
        except Exception:
            pass
    if method == "rayleigh":
        plan_method = getattr(coupling, "_rayleigh_plan", None)
        if callable(plan_method):
            plan = plan_method()
            payload["rayleigh_z_cut_nm"] = float(plan.z_cut)
            payload["rayleigh_half_width"] = int(plan.half_width)
            payload["rayleigh_reciprocal_modes"] = int(plan.n_modes_reciprocal)
        destinations = getattr(coupling, "_near_destinations", None)
        if destinations is not None:
            payload["rayleigh_near_directed_pairs"] = int(np.asarray(destinations).size)
        near_plan = getattr(coupling, "_near_cache_memory_plan", None)
        if near_plan is not None:
            # The current CuPy policy is deliberately device-resident only;
            # the memory plan no longer carries the removed host-spill enum.
            payload["rayleigh_near_cache_residency"] = "device"
            payload["rayleigh_near_cache_structural_bytes"] = int(
                near_plan.estimate.structural_bytes
            )
            payload["rayleigh_near_cache_required_device_bytes"] = int(
                near_plan.required_device_bytes
            )
    return payload


def _run_config(
    args: argparse.Namespace,
    *,
    geometry: PeriodicGeometry,
    backend: Backend,
    method: PeriodicMethod,
) -> dict[str, Any]:
    compute_dtype, accum_dtype = _dtype_policy(args, backend)
    cache_translation_blocks = bool(args.cache_translation_blocks and method == "ewald")
    return {
        "family": geometry.family,
        "n_particles": int(geometry.radii.size),
        "ax": float(geometry.ax),
        "ay": float(geometry.ay),
        "height": float(geometry.height),
        "backend": backend,
        "periodic_method": method,
        "compute_dtype": compute_dtype,
        "accum_dtype": accum_dtype,
        "wavelength": float(args.wavelength),
        "n_medium": float(args.n_medium),
        "lmax": int(args.lmax),
        "polarization": str(args.polarization),
        "polar_angle": float(args.polar_angle),
        "azimuthal_angle": float(args.azimuthal_angle),
        "solver": str(args.solver),
        "solver_rtol": float(args.solver_rtol),
        "solver_restart": int(args.solver_restart),
        "solver_maxiter": int(args.solver_maxiter),
        "solver_compute_final_residual": True,
        "solver_gcro_recycle_dim": int(args.solver_gcro_recycle_dim),
        "cache_translation_blocks": cache_translation_blocks,
        "shell_tolerance": float(args.shell_tolerance),
        "max_shells": int(args.max_shells),
        "rayleigh_z_cut": args.rayleigh_z_cut,
        "rayleigh_reciprocal_shells": args.rayleigh_reciprocal_shells,
        "seed": int(args.seed),
        # Include every geometry-generation input in the resumable fingerprint.
        # The box dimensions alone are not sufficient: different palettes or
        # geometry inputs can happen to produce the same floating-point box.
        "target_volume_fraction": float(args.target_volume_fraction),
        "lateral_height_over_wavelength": args.lateral_height_over_wavelength,
        "vertical_period_over_wavelength": args.vertical_period_over_wavelength,
        "radii_palette": list(cast(tuple[float, ...], args.radii)),
        "real_n_palette": list(cast(tuple[float, ...], args.real_n)),
        "imag_n_palette": list(cast(tuple[float, ...], args.imag_n)),
        "max_attempts_per_particle": int(args.max_attempts_per_particle),
        "anomaly_policy": str(args.anomaly_policy),
    }


def _run_one(
    args: argparse.Namespace,
    *,
    geometry: PeriodicGeometry,
    backend: Backend,
    method: PeriodicMethod,
) -> dict[str, Any]:
    compute_dtype, accum_dtype = _dtype_policy(args, backend)
    cache_translation_blocks = bool(args.cache_translation_blocks and method == "ewald")
    particles = pcl.core.spheres_from_arrays(
        positions=geometry.positions,
        radii=geometry.radii,
        refractive_indices=geometry.refractive_indices,
    )
    config = pcl.SimulationConfig(
        wavelength=float(args.wavelength),
        n_medium=complex(float(args.n_medium), 0.0),
        lmax=int(args.lmax),
        periodic=_periodic_spec(args, geometry, method),
        solver_method=cast(SolverMethod, args.solver),
        solver_rtol=float(args.solver_rtol),
        solver_restart=int(args.solver_restart),
        solver_gcro_recycle_dim=int(args.solver_gcro_recycle_dim),
        solver_maxiter=int(args.solver_maxiter),
        solver_compute_final_residual=True,
        operator_backend=backend,
        coupling_backend="pairwise",
        postprocessing_backend="inherit",
        compute_dtype=cast(Literal["complex64", "complex128"], compute_dtype),
        accum_dtype=cast(Literal["complex64", "complex128"], accum_dtype),
        cache_translation_blocks=cache_translation_blocks,
        verbose=not bool(args.quiet),
    )
    source = _make_source(args)

    _free_backend_memory(backend)
    _sync(backend)
    simulation = pcl.Simulation(config, particles=particles)
    started = time.perf_counter()
    run = simulation.run(source, include_farfield=False)
    _sync(backend)
    elapsed = time.perf_counter() - started

    solver = run.solver_result
    metadata = solver.block_metadata or {}
    raw_timings = metadata.get("simulation_phase_timings_s")
    phase_timings = (
        {str(key): float(value) for key, value in raw_timings.items()}
        if isinstance(raw_timings, dict)
        else {}
    )
    preparation_keys = {
        "source_projection_s",
        "prepare_operator_s",
        "rhs_Tb_s",
        "periodic_ewald_preparation_s",
        "periodic_rayleigh_preparation_s",
        "periodic_w_block_generation_s",
    }
    preparation_timings = {
        key: value for key, value in phase_timings.items() if key in preparation_keys
    }
    action_counts: dict[str, int] = {
        key: int(metadata[key])
        for key in ("operator_applications", "adjoint_applications")
        if key in metadata
    }
    payload = {
        "status": "completed",
        "family": geometry.family,
        "n_particles": int(geometry.radii.size),
        "unknowns": int(geometry.radii.size) * n_modes(int(args.lmax)),
        "backend": backend,
        "periodic_method": method,
        "compute_dtype": compute_dtype,
        "accum_dtype": accum_dtype,
        "solve_wall_time_s": float(elapsed),
        "solver_result": {
            "method": str(solver.method),
            "iterations": int(np.asarray(solver.iterations)),
            "info": int(np.asarray(solver.info)),
            "relative_residual": float(np.asarray(solver.relative_residual)),
            "residual_norm": float(np.asarray(solver.residual_norm)),
            **action_counts,
        },
        "phase_timings_s": phase_timings,
        "preparation_timings_s": preparation_timings,
        "preparation_wall_time_s": float(sum(preparation_timings.values())),
        "cache_translation_blocks": cache_translation_blocks,
        "operator_diagnostics": _operator_diagnostics(simulation, method),
        "geometry": _geometry_summary(geometry, float(args.wavelength)),
        "rayleigh_safety": _safety_summary(args, geometry),
    }
    del run, simulation, particles
    _free_backend_memory(backend)
    return payload


def _run_file(
    out_dir: Path,
    *,
    family: GrowthFamily,
    n_particles: int,
    backend: Backend,
    method: PeriodicMethod,
) -> Path:
    return out_dir / f"{family}_N_{n_particles:08d}_{backend}_{method}.json"


def _existing_payload(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Existing result has a different configuration fingerprint: {path}. "
            "Use another output directory, delete the file, or pass --force."
        )
    if payload.get("status") != "completed":
        return None
    return cast(dict[str, Any], payload)


def _write_summary(
    *,
    out_dir: Path,
    status: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> None:
    payload = {
        "status": status,
        "script": Path(__file__).name,
        "command": list(sys.argv),
        "created_at_unix_s": float(time.time()),
        "platform": {"python": sys.version, "platform": platform.platform()},
        "pyceles_version": str(getattr(pcl, "__version__", "unknown")),
        "families": list(args.families),
        "methods": list(args.methods),
        "backends": list(args.backends),
        "n_values": {
            "lateral": list(args.n_values_lateral),
            "vertical": list(args.n_values_vertical),
        },
        "target_volume_fraction": float(args.target_volume_fraction),
        "lateral_height_over_wavelength": args.lateral_height_over_wavelength,
        "vertical_period_over_wavelength": args.vertical_period_over_wavelength,
        "anomaly_policy": str(args.anomaly_policy),
        "results": rows,
    }
    _atomic_write_json(out_dir / "summary.json", payload)


def _preflight_sweep(
    args: argparse.Namespace,
) -> dict[tuple[GrowthFamily, int], dict[str, Any]]:
    """Check every requested cell before starting any timed solve.

    The safety report depends only on the deterministic sampled volume and the
    resulting lateral period.  Particle placement is therefore intentionally
    not repeated here: the actual RSA geometry is generated only after the
    complete anomaly preflight succeeds.
    """
    plans: dict[tuple[GrowthFamily, int], dict[str, Any]] = {}
    failures: list[str] = []
    for family in cast(tuple[GrowthFamily, ...], args.families):
        n_values = (
            cast(tuple[int, ...], args.n_values_lateral)
            if family == "lateral"
            else cast(tuple[int, ...], args.n_values_vertical)
        )
        for n_particles in n_values:
            radii, _, particle_volume = _sample_particle_properties(
                n_particles=int(n_particles),
                radii_palette=cast(tuple[float, ...], args.radii),
                real_n_palette=cast(tuple[float, ...], args.real_n),
                imag_n_palette=cast(tuple[float, ...], args.imag_n),
                seed=int(args.seed),
            )
            fixed_length = _family_fixed_length(family=family, args=args)
            ax, ay, height = _box_dimensions(
                family=family,
                particle_volume=particle_volume,
                target_volume_fraction=float(args.target_volume_fraction),
                fixed_length=fixed_length,
            )
            if min(ax, ay, height) <= 2.0 * float(np.max(radii)):
                raise RuntimeError(
                    "Periodic scaling preflight found an infeasible cell before any "
                    f"simulation: {family} N={n_particles} is too small for the "
                    "largest sphere."
                )
            safety = _safety_summary_for_box(args, ax=ax, ay=ay)
            plans[(family, int(n_particles))] = {
                "ax": float(ax),
                "ay": float(ay),
                "height": float(height),
                "safety": safety,
            }
            if not bool(args.quiet):
                print(
                    f"Preflight {family} N={n_particles}: "
                    f"L/lambda={ax / float(args.wavelength):.3f}, "
                    f"H/lambda={height / float(args.wavelength):.3f}, "
                    f"clearance={safety['min_clearance']:.3g} ({safety['warning_level']})",
                    flush=True,
                )
            if safety["warning_level"] != "ok":
                failures.append(f"{family} N={n_particles}: {safety['message']!s}")

    if failures and args.anomaly_policy == "error":
        details = "\n  ".join(failures)
        raise RuntimeError(
            "Periodic scaling preflight found Rayleigh/Wood safety warnings; "
            "no simulations were started.\n  " + details
        )
    return plans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ewald_rayleigh_scaling"))
    parser.add_argument(
        "--families",
        type=lambda value: _parse_names(value, allowed={"lateral", "vertical"}, name="families"),
        default=("lateral", "vertical"),
    )
    parser.add_argument(
        "--methods",
        type=lambda value: _parse_names(value, allowed={"ewald", "rayleigh"}, name="methods"),
        default=("ewald", "rayleigh"),
    )
    parser.add_argument(
        "--backends",
        type=lambda value: _parse_names(value, allowed={"numpy", "cupy"}, name="backends"),
        default=("cupy",),
    )
    parser.add_argument(
        "--n-values-lateral",
        type=lambda value: _parse_int_list(value, name="n-values-lateral"),
        default=(114, 228, 347, 496, 889, 1126, 1744, 2153, 3344, 4318, 7307, 10977),
    )
    parser.add_argument(
        "--n-values-vertical",
        type=lambda value: _parse_int_list(value, name="n-values-vertical"),
        default=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
    )

    # Match finite_pairwise_mlfmm.py unless stated otherwise.
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--wavelength", type=float, default=550.0)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--polarization", choices=("TE", "TM"), default="TE")
    parser.add_argument("--polar-angle", type=float, default=0.0)
    parser.add_argument("--azimuthal-angle", type=float, default=0.0)
    parser.add_argument(
        "--solver",
        choices=("gmres", "gcro", "lsqr"),
        default="gmres",
        help=(
            "Periodic solver. BiCGSTAB and LGMRES are intentionally not offered "
            "here because this benchmark targets robust periodic solves. GCRO "
            "is CuPy-only forward GCRO-DR; LSQR uses the exact periodic adjoint "
            "and is useful for capped cold-start probes."
        ),
    )
    parser.add_argument("--solver-rtol", type=float, default=1.0e-4)
    parser.add_argument(
        "--solver-restart",
        type=int,
        default=250,
        help="GMRES/GCRO restart dimension; ignored by standalone LSQR.",
    )
    parser.add_argument("--solver-gcro-recycle-dim", type=int, default=8)
    parser.add_argument("--solver-maxiter", type=int, default=20000)
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
    parser.add_argument(
        "--lateral-height-over-wavelength",
        type=float,
        default=20.0,
        help=(
            "Fixed occupied height for the lateral family in vacuum-wavelength units. "
            "The production default is 20.0; override it for diagnostic ladders."
        ),
    )
    parser.add_argument(
        "--vertical-period-over-wavelength",
        type=float,
        default=4.75,
        help=(
            "Fixed square period for the vertical family in vacuum-wavelength units. "
            "The default 4.75 is a broad Rayleigh-safe period."
        ),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-attempts-per-particle", type=int, default=5000)

    # Production precision policies: NumPy reference versus fast CuPy solve.
    parser.add_argument(
        "--numpy-compute-dtype", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument(
        "--numpy-accum-dtype", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument(
        "--cupy-compute-dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument(
        "--cupy-accum-dtype", choices=("complex64", "complex128"), default="complex128"
    )

    parser.add_argument(
        "--shell-tolerance",
        type=float,
        default=1.0e-7,
        help="Periodic structural tolerance; deliberately tighter than the 1e-4 Krylov target.",
    )
    parser.add_argument(
        "--max-shells",
        type=int,
        default=128,
        help=(
            "Maximum adaptive reciprocal shell for periodic Ewald/Rayleigh "
            "truncation. The larger default avoids rejecting the high-period "
            "tail of the scaling ladder; use --rayleigh-reciprocal-shells "
            "to pin an explicit Rayleigh aperture."
        ),
    )
    parser.add_argument("--rayleigh-z-cut", type=float, default=None)
    parser.add_argument("--rayleigh-reciprocal-shells", type=int, default=None)
    parser.add_argument(
        "--anomaly-policy",
        dest="anomaly_policy",
        choices=("error", "skip", "allow"),
        default="error",
        help=(
            "Handling for Rayleigh/Wood safety warnings: error stops the sweep, "
            "skip writes explicit skipped cases, and allow runs them. "
            "The policy applies to Ewald and Rayleigh because both use singular "
            "shifted reciprocal terms at an exact grazing order."
        ),
    )
    parser.add_argument(
        "--cache-translation-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build and retain the dense periodic Ewald translation-block cache. "
            "This is the practical small/medium-size Ewald mode used for the "
            "main comparison; it scales quadratically in the number of "
            "unknowns and must be disabled beyond the device-memory boundary. "
            "Without it, the bounded-memory matrix-free action is used."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.families = tuple(cast(tuple[GrowthFamily, ...], args.families))
    args.methods = tuple(cast(tuple[PeriodicMethod, ...], args.methods))
    args.backends = tuple(cast(tuple[Backend, ...], args.backends))

    if args.solver == "gcro" and "numpy" in args.backends:
        raise ValueError(
            "--solver gcro is available only with the CuPy backend; select "
            "--backends cupy or use --solver gmres."
        )
    if int(args.solver_gcro_recycle_dim) < 1:
        raise ValueError("--solver-gcro-recycle-dim must be positive.")

    if not (0.0 < float(args.target_volume_fraction) < 0.4):
        raise ValueError("--target-volume-fraction must lie in (0, 0.4).")
    if any(radius <= 0.0 for radius in cast(tuple[float, ...], args.radii)):
        raise ValueError("All radii must be positive.")

    root = _repo_root()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight_sweep(args)
    rows: list[dict[str, Any]] = []
    memory_exhausted: set[tuple[GrowthFamily, Backend, PeriodicMethod]] = set()
    status = "completed"
    try:
        for family in cast(tuple[GrowthFamily, ...], args.families):
            n_values = (
                cast(tuple[int, ...], args.n_values_lateral)
                if family == "lateral"
                else cast(tuple[int, ...], args.n_values_vertical)
            )
            for n_particles in n_values:
                geometry = _generate_geometry(
                    family=family,
                    n_particles=int(n_particles),
                    args=args,
                )
                geometry_summary = _geometry_summary(geometry, float(args.wavelength))
                safety = cast(dict[str, Any], preflight[(family, int(n_particles))]["safety"])
                if not bool(args.quiet):
                    print(
                        f"{family} N={n_particles}: "
                        f"L/lambda={geometry_summary['ax_over_wavelength']:.3f}, "
                        f"H/lambda={geometry_summary['height_over_wavelength']:.3f}, "
                        f"clearance={safety['min_clearance']:.3g}",
                        flush=True,
                    )

                for method in cast(tuple[PeriodicMethod, ...], args.methods):
                    for backend in cast(tuple[Backend, ...], args.backends):
                        run_config = _run_config(
                            args, geometry=geometry, backend=backend, method=method
                        )
                        fingerprint = _fingerprint(run_config)
                        path = _run_file(
                            out_dir,
                            family=family,
                            n_particles=int(n_particles),
                            backend=backend,
                            method=method,
                        )

                        if safety["warning_level"] != "ok" and args.anomaly_policy == "skip":
                            payload = {
                                "status": "skipped",
                                "fingerprint": fingerprint,
                                "config": run_config,
                                "family": family,
                                "n_particles": int(n_particles),
                                "backend": backend,
                                "periodic_method": method,
                                "geometry": geometry_summary,
                                "rayleigh_safety": safety,
                                "skip_reason": "Rayleigh/Wood safety warning",
                            }
                            _atomic_write_json(path, payload)
                            rows.append(payload)
                            _write_summary(
                                out_dir=out_dir,
                                status=status,
                                args=args,
                                rows=rows,
                            )
                            if not bool(args.quiet):
                                print(f"Skipping unsafe Rayleigh case: {path.name}", flush=True)
                            continue

                        existing = (
                            None if bool(args.force) else _existing_payload(path, fingerprint)
                        )
                        if existing is not None:
                            rows.append(existing)
                            if not bool(args.quiet):
                                print(f"Skipping {path.name}", flush=True)
                            continue

                        resource_key = (family, backend, method)
                        if resource_key in memory_exhausted:
                            payload = {
                                "status": "skipped",
                                "fingerprint": fingerprint,
                                "config": run_config,
                                "family": family,
                                "n_particles": int(n_particles),
                                "backend": backend,
                                "periodic_method": method,
                                "geometry": geometry_summary,
                                "rayleigh_safety": safety,
                                "skip_reason": (
                                    "A previous larger case exhausted backend memory "
                                    f"for {backend}/{method}; later cases were not attempted."
                                ),
                            }
                            _atomic_write_json(path, payload)
                            rows.append(payload)
                            _write_summary(
                                out_dir=out_dir,
                                status=status,
                                args=args,
                                rows=rows,
                            )
                            if not bool(args.quiet):
                                print(
                                    f"Skipping {path.name} after previous memory exhaustion",
                                    flush=True,
                                )
                            continue

                        compute_dtype, accum_dtype = _dtype_policy(args, backend)
                        try:
                            if not bool(args.quiet):
                                print(
                                    f"Running {family} N={n_particles} {backend}/{method} "
                                    f"({compute_dtype}/{accum_dtype})",
                                    flush=True,
                                )
                            payload = _run_one(
                                args, geometry=geometry, backend=backend, method=method
                            )
                            payload["fingerprint"] = fingerprint
                            payload["config"] = run_config
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            payload = {
                                "status": "failed",
                                "fingerprint": fingerprint,
                                "config": run_config,
                                "family": family,
                                "n_particles": int(n_particles),
                                "backend": backend,
                                "periodic_method": method,
                                "geometry": geometry_summary,
                                "rayleigh_safety": safety,
                                "error_type": type(exc).__name__,
                                "error": repr(exc),
                            }
                            _free_backend_memory(backend)
                            if _is_memory_exhaustion(exc):
                                memory_exhausted.add(resource_key)
                                payload["skip_reason"] = (
                                    "Later cases for this family/backend/method were "
                                    "skipped after backend memory exhaustion."
                                )
                                print(
                                    f"MEMORY EXHAUSTED for {family} {backend}/{method}; "
                                    "later cases for this resource will be skipped.",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            if bool(args.stop_on_error):
                                _atomic_write_json(path, payload)
                                rows.append(payload)
                                raise
                            print(f"FAILED {path.name}: {exc!r}", file=sys.stderr, flush=True)

                        _atomic_write_json(path, payload)
                        rows.append(payload)
                        _write_summary(
                            out_dir=out_dir,
                            status=status,
                            args=args,
                            rows=rows,
                        )
    except KeyboardInterrupt:
        status = "interrupted"
        _write_summary(
            out_dir=out_dir,
            status=status,
            args=args,
            rows=rows,
        )
        print(f"Interrupted. Partial summary: {out_dir / 'summary.json'}", flush=True)
        return

    _write_summary(
        out_dir=out_dir,
        status=status,
        args=args,
        rows=rows,
    )
    print(f"Wrote scaling summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
