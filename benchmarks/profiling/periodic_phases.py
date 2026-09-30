#!/usr/bin/env python3
"""Profile periodic pyceles solve and near-field phases for a 500-sphere cell.

This script is the periodic counterpart of ``finite_phases.py``.  It
uses the 500 sphere radii and refractive indices from
``examples/sphere_parameters.txt``
exactly once, places them into a random non-overlapping periodic cube, and
profiles the periodic solve plus optional near-field exterior-xy, interior-xy,
and xz maps.

The default periodic cube side is chosen to keep the particle filling fraction
roughly comparable to the finite CELES example cluster.  The original cluster's
sphere-inclusive extents are approximately

    dx = 3571.1 nm, dy = 3535.1 nm, dz = 3397.0696 nm.

Using the largest of these as the diameter of an equivalent sphere gives a
volume-matched cube side of 2878.29 nm, rounded to 3000 nm.

Near-field maps are evaluated with a resolution of 30 nm, to create maps with a
comparable number of pixels to the non-periodic example configuration.  The
profile includes both an exterior xy plane (above the particle slab) and an
interior xy plane (at the midpoint of the occupied slab), in addition to the
vertical xz map.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import pstats
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

import pyceles as pcl
from pyceles._optional import import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.operators import estimate_translation_cache_bytes

DEFAULT_PARTICLE_COUNT = 500
DEFAULT_PERIODIC_SIDE_NM = 3000.0
DEFAULT_CLUSTER_DIAMETERS_NM = (3571.1, 3535.1, 3397.0696)
DEFAULT_VOLUME_MATCHED_SIDE_NM = 2878.292233494106


@dataclass(frozen=True)
class GeneratedGeometry:
    positions: np.ndarray
    radii: np.ndarray
    refractive_indices: np.ndarray
    prototype_indices: np.ndarray
    attempted_candidates: int
    rejected_overlaps: int


def _find_repo_root() -> Path:
    here = Path.cwd().resolve()
    candidates = [here, Path(__file__).resolve().parent]
    for base in list(candidates):
        candidates.extend(base.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "examples" / "sphere_parameters.txt").exists():
            return candidate
    raise FileNotFoundError("Could not find examples/sphere_parameters.txt")


def _load_prototype_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns in {path}: radius x y z n_real n_imag. "
            f"Got shape {data.shape}."
        )
    radii = np.asarray(data[:, 0], dtype=float)
    positions = np.asarray(data[:, 1:4], dtype=float)
    refractive_indices = np.asarray(data[:, 4] + 1j * data[:, 5], dtype=np.complex128)
    if radii.size != DEFAULT_PARTICLE_COUNT:
        raise ValueError(
            f"This periodic profiler expects exactly {DEFAULT_PARTICLE_COUNT} prototype spheres. "
            f"Got {radii.size} in {path}."
        )
    return positions, radii, refractive_indices


def _cluster_volume_side_summary(positions: np.ndarray, radii: np.ndarray) -> dict[str, Any]:
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    r = np.asarray(radii, dtype=float).reshape(-1)
    center_diameters = np.max(pos, axis=0) - np.min(pos, axis=0)
    sphere_inclusive_diameters = np.max(pos + r[:, None], axis=0) - np.min(pos - r[:, None], axis=0)
    dmax = float(np.max(sphere_inclusive_diameters))
    sphere_volume = (4.0 / 3.0) * math.pi * (0.5 * dmax) ** 3
    cube_side = sphere_volume ** (1.0 / 3.0)
    return {
        "center_diameters_nm": [float(v) for v in center_diameters],
        "sphere_inclusive_diameters_nm": [float(v) for v in sphere_inclusive_diameters],
        "largest_sphere_inclusive_diameter_nm": dmax,
        "equivalent_spherical_volume_nm3": float(sphere_volume),
        "volume_matched_cube_side_nm": float(cube_side),
        "rounded_default_side_nm": float(round(cube_side / 100.0) * 100.0),
    }


def _minimum_image_delta(a: np.ndarray, b: np.ndarray, *, side_nm: float) -> np.ndarray:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= float(side_nm) * np.rint(delta / float(side_nm))
    return np.asarray(delta, dtype=float)


def _cell_index(
    position: np.ndarray, *, side_nm: float, cell_size: float, ngrid: int
) -> tuple[int, int, int]:
    coords = np.array(
        [
            float(position[0]) + 0.5 * float(side_nm),
            float(position[1]) + 0.5 * float(side_nm),
            float(position[2]),
        ],
        dtype=float,
    )
    coords = np.mod(coords, float(side_nm))
    idx = np.floor(coords / float(cell_size)).astype(np.int64)
    idx = np.mod(idx, int(ngrid))
    return int(idx[0]), int(idx[1]), int(idx[2])


def _generate_periodic_rsa_geometry(
    *,
    prototype_radii: np.ndarray,
    prototype_refractive_indices: np.ndarray,
    side_nm: float,
    seed: int,
    max_attempts_per_particle: int,
) -> GeneratedGeometry:
    """Place every prototype sphere exactly once into a periodic cube."""
    rng = np.random.default_rng(int(seed))
    radii_proto = np.asarray(prototype_radii, dtype=float).reshape(-1)
    indices_proto = np.asarray(prototype_refractive_indices, dtype=np.complex128).reshape(-1)
    if radii_proto.size != indices_proto.size:
        raise ValueError("Prototype radii and refractive-index arrays must have the same length.")
    n = int(radii_proto.size)
    order = np.asarray(rng.permutation(n), dtype=np.int64)
    radii_ordered = radii_proto[order]

    max_radius = float(np.max(radii_ordered))
    ngrid = max(1, math.floor(float(side_nm) / max(2.0 * max_radius, 1.0)))
    cell_size = float(side_nm) / float(ngrid)

    positions: list[np.ndarray] = []
    radii: list[float] = []
    prototype_indices: list[int] = []
    grid: dict[tuple[int, int, int], list[int]] = {}
    attempted = 0
    rejected = 0

    def overlaps(candidate: np.ndarray, radius: float) -> bool:
        if not positions:
            return False
        ci = _cell_index(candidate, side_nm=side_nm, cell_size=cell_size, ngrid=ngrid)
        span = math.ceil((float(radius) + max_radius) / cell_size)
        visited: set[tuple[int, int, int]] = set()
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for dz in range(-span, span + 1):
                    key = ((ci[0] + dx) % ngrid, (ci[1] + dy) % ngrid, (ci[2] + dz) % ngrid)
                    if key in visited:
                        continue
                    visited.add(key)
                    for j in grid.get(key, ()):
                        delta = _minimum_image_delta(candidate, positions[j], side_nm=side_nm)
                        min_dist = float(radius) + float(radii[j])
                        if float(np.dot(delta, delta)) < min_dist * min_dist:
                            return True
        return False

    for radius, original_index in zip(radii_ordered, order, strict=True):
        placed = False
        for _ in range(int(max_attempts_per_particle)):
            attempted += 1
            candidate = np.array(
                [
                    rng.uniform(-0.5 * float(side_nm), 0.5 * float(side_nm)),
                    rng.uniform(-0.5 * float(side_nm), 0.5 * float(side_nm)),
                    rng.uniform(0.0, float(side_nm)),
                ],
                dtype=float,
            )
            if overlaps(candidate, float(radius)):
                rejected += 1
                continue
            idx = len(positions)
            positions.append(candidate)
            radii.append(float(radius))
            prototype_indices.append(int(original_index))
            key = _cell_index(candidate, side_nm=side_nm, cell_size=cell_size, ngrid=ngrid)
            grid.setdefault(key, []).append(idx)
            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"Failed to place prototype sphere {int(original_index)} after "
                f"{max_attempts_per_particle} attempts. Try a larger --side-nm."
            )

    proto_idx = np.asarray(prototype_indices, dtype=np.int64)
    return GeneratedGeometry(
        positions=np.asarray(positions, dtype=float).reshape(n, 3),
        radii=np.asarray(radii, dtype=float).reshape(n),
        refractive_indices=indices_proto[proto_idx],
        prototype_indices=proto_idx,
        attempted_candidates=int(attempted),
        rejected_overlaps=int(rejected),
    )


def _save_geometry(out_dir: Path, geom: GeneratedGeometry) -> dict[str, str]:
    txt_path = out_dir / "periodic_500_geometry.txt"
    npz_path = out_dir / "periodic_500_geometry.npz"
    rows = np.column_stack(
        [
            geom.radii,
            geom.positions[:, 0],
            geom.positions[:, 1],
            geom.positions[:, 2],
            geom.refractive_indices.real,
            geom.refractive_indices.imag,
            geom.prototype_indices.astype(float),
        ]
    )
    np.savetxt(
        txt_path,
        rows,
        fmt="%.12g",
        header="radius_nm x_nm y_nm z_nm n_real n_imag prototype_index",
        comments="",
    )
    np.savez_compressed(
        npz_path,
        positions=geom.positions,
        radii=geom.radii,
        refractive_indices=geom.refractive_indices,
        prototype_indices=geom.prototype_indices,
    )
    return {"txt": str(txt_path), "npz": str(npz_path)}


def _periodic_summary(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {"available": False}
    power = payload.power
    return {
        "available": True,
        "order_count": int(np.asarray(payload.order_mn).shape[0]),
        "propagating_count": int(
            np.count_nonzero(np.asarray(payload.order_propagating, dtype=bool))
        ),
        "incident_flux": float(payload.incident_flux),
        **{
            key: value
            for key, value in power.to_mapping().items()
            if key != "local_absorbed_power_per_particle"
            and key != "local_absorptance_per_particle"
        },
    }


def _solver_phase_timings(solver_result: Any) -> dict[str, float]:
    metadata = getattr(solver_result, "block_metadata", None)
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get("simulation_phase_timings_s")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
            out[str(key)] = float(value)
    return out


def _save_periodic_payload(path: Path, payload: Any) -> None:
    if payload is None:
        return
    power_mapping = {
        key: np.asarray(value)
        for key, value in payload.power.to_mapping().items()
        if value is not None
    }
    np.savez_compressed(
        path,
        lattice_a1=np.asarray(payload.lattice_a1, dtype=float),
        lattice_a2=np.asarray(payload.lattice_a2, dtype=float),
        unit_cell_area=np.asarray(float(payload.unit_cell_area), dtype=float),
        incident_k_parallel=np.asarray(payload.incident_k_parallel, dtype=float),
        output_bmax=np.asarray(
            np.nan if payload.output_bmax is None else float(payload.output_bmax)
        ),
        order_mn=np.asarray(payload.order_mn, dtype=np.int32),
        order_k_parallel=np.asarray(payload.order_k_parallel, dtype=float),
        order_kz=np.asarray(payload.order_kz, dtype=np.complex128),
        order_propagating=np.asarray(payload.order_propagating, dtype=bool),
        reflected_amplitudes=np.asarray(payload.reflected_amplitudes, dtype=np.complex128),
        transmitted_amplitudes=np.asarray(payload.transmitted_amplitudes, dtype=np.complex128),
        reflected_flux_per_order=np.asarray(payload.reflected_flux_per_order, dtype=float),
        transmitted_flux_per_order=np.asarray(payload.transmitted_flux_per_order, dtype=float),
        incident_flux=np.asarray(float(payload.incident_flux), dtype=float),
        **cast(Any, power_mapping),
    )


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
    rows.sort(key=lambda item: item["cumtime_s"], reverse=True)
    return rows[: int(limit)]


def _sync_gpu() -> None:
    cupy, _ = import_cupy()
    cupy.cuda.Stream.null.synchronize()


def _profile_phase(
    *,
    phase: str,
    out_dir: Path,
    top_n: int,
    fn: Callable[[], Any],
    synchronize_gpu: bool = False,
    cuda_profiler_api: bool = False,
) -> tuple[Any, dict[str, Any]]:
    if synchronize_gpu:
        _sync_gpu()
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    if cuda_profiler_api:
        import cupyx.profiler

        profile_context = cupyx.profiler.profile()
    else:
        profile_context = nullcontext()
    with profile_context:
        result = prof.runcall(fn)
    if synchronize_gpu:
        _sync_gpu()
    elapsed = time.perf_counter() - t0

    prof_path = out_dir / f"{phase}.prof"
    txt_path = out_dir / f"{phase}.txt"
    prof.dump_stats(str(prof_path))
    with txt_path.open("w", encoding="utf-8") as f:
        stats = pstats.Stats(prof, stream=f)
        stats.sort_stats("cumulative")
        stats.print_stats(int(top_n))
    return result, {
        "phase": phase,
        "wall_time_s": float(elapsed),
        "prof_file": str(prof_path),
        "text_report": str(txt_path),
        "top_functions": _top_stats(prof, top_n),
    }


def _default_xy_nearfield_z(geom: GeneratedGeometry) -> float:
    rmax = float(np.max(geom.radii))
    top_surface = float(np.max(geom.positions[:, 2] + geom.radii))
    return top_surface + 2.0 * rmax


def _xy_slab_bounds(geom: GeneratedGeometry) -> tuple[float, float]:
    """Return the finite z interval occupied by the generated particle slab."""
    z_positions = np.asarray(geom.positions[:, 2], dtype=float)
    radii = np.asarray(geom.radii, dtype=float)
    z_lower = z_positions - radii
    z_upper = z_positions + radii
    zmin = float(np.min(z_lower))
    zmax = float(np.max(z_upper))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or not zmin < zmax:
        raise ValueError(f"Generated geometry has invalid z slab bounds: {zmin!r}, {zmax!r}.")
    return zmin, zmax


def _default_xy_interior_nearfield_z(geom: GeneratedGeometry) -> float:
    """Choose a reproducible horizontal plane strictly inside the particle slab."""
    zmin, zmax = _xy_slab_bounds(geom)
    return 0.5 * (zmin + zmax)


def _resolve_xy_interior_nearfield_z(geom: GeneratedGeometry, requested_z: float | None) -> float:
    value = _default_xy_interior_nearfield_z(geom) if requested_z is None else float(requested_z)
    zmin, zmax = _xy_slab_bounds(geom)
    if not zmin < value < zmax:
        raise ValueError(
            "The interior xy plane must lie strictly inside the occupied particle slab: "
            f"z={value:g} nm, bounds=({zmin:g}, {zmax:g}) nm."
        )
    return value


def _nearfield_z_extent(
    geom: GeneratedGeometry, *, excess_nm: float | None = None
) -> tuple[float, float]:
    zmin, zmax = _xy_slab_bounds(geom)
    rmax = float(np.max(geom.radii))
    excess = 2.0 * rmax if excess_nm is None else float(excess_nm)
    return zmin - excess, zmax + excess


def _field_bmax_for_evanescent_decay(*, k: float, distance_nm: float, decay: float) -> float:
    kf = float(k)
    d = max(float(distance_nm), 1.0e-12)
    q = max(float(decay), 0.0) / d
    return float(math.sqrt(kf * kf + q * q))


def _resolved_field_bmax(
    *,
    field_bmax: float | None,
    output_bmax: float | None,
    field_evanescent_decay: float,
    nearfield_excess_nm: float | None,
    geom: GeneratedGeometry,
    k: float,
) -> tuple[float, dict[str, Any]]:
    if field_bmax is not None:
        value = float(field_bmax)
        source = "explicit_field_bmax"
        distance = None
        decay = None
    elif output_bmax is not None:
        value = float(output_bmax)
        source = "output_bmax"
        distance = None
        decay = None
    else:
        rmax = float(np.max(geom.radii))
        distance = 2.0 * rmax if nearfield_excess_nm is None else float(nearfield_excess_nm)
        decay = float(field_evanescent_decay)
        value = _field_bmax_for_evanescent_decay(k=float(k), distance_nm=distance, decay=decay)
        source = "auto_evanescent_decay"
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Resolved field_bmax must be finite and positive. Got {value!r}.")
    return value, {
        "source": source,
        "value_nm_inv": float(value),
        "k_nm_inv": float(k),
        "value_over_k": float(value / float(k)) if float(k) > 0.0 else None,
        "characteristic_distance_nm": None if distance is None else float(distance),
        "evanescent_decay": None if decay is None else float(decay),
    }


def _save_nearfield_npz(path: Path, nf_slice: Any, field_bmax: float) -> dict[str, Any]:
    e_total = np.asarray(nf_slice.field_maps["total"][0], dtype=np.complex128)
    h_total = np.asarray(nf_slice.field_maps["total"][1], dtype=np.complex128)
    inside = np.asarray(nf_slice.inside, dtype=bool)
    intensity = np.sum(np.abs(e_total) ** 2, axis=-1)
    np.savez_compressed(
        path,
        axis_0=np.asarray(nf_slice.axis_0, dtype=float),
        axis_1=np.asarray(nf_slice.axis_1, dtype=float),
        inside=inside,
        E_total=e_total,
        H_total=h_total,
        intensity=intensity,
        plane=np.asarray(str(nf_slice.plane)),
        plane_value=np.asarray(float(nf_slice.plane_value)),
        field_bmax=np.asarray(float(field_bmax)),
    )
    return {
        "npz": str(path),
        "shape": [int(v) for v in intensity.shape],
        "intensity_min": float(np.nanmin(intensity)),
        "intensity_max": float(np.nanmax(intensity)),
        "intensity_mean": float(np.nanmean(intensity)),
        "inside_count": int(np.count_nonzero(inside)),
        "inside_fraction": float(np.count_nonzero(inside) / max(inside.size, 1)),
        "field_bmax": float(field_bmax),
    }


def _build_config(
    *,
    args: argparse.Namespace,
    side_nm: float,
    cache_blocks: bool,
    coupling_backend: Literal["pairwise", "mlfmm"],
    operator_backend: Literal["numpy", "cupy"],
    postprocessing_backend: Literal["inherit", "numpy", "cupy"],
) -> pcl.SimulationConfig:
    periodic = pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(ax=float(side_nm), ay=float(side_nm)),
        options=pcl.PeriodicOptions(
            method=cast(Literal["ewald", "rayleigh"], args.periodic_method),
            eta=args.eta,
            real_shells=args.real_shells,
            reciprocal_shells=args.reciprocal_shells,
            shell_tolerance=float(args.shell_tolerance),
            max_shells=int(args.max_shells),
            output_bmax=args.output_bmax,
            rayleigh_z_cut=args.rayleigh_z_cut,
            rayleigh_reciprocal_shells=args.rayleigh_reciprocal_shells,
        ),
    )
    return pcl.SimulationConfig(
        wavelength=float(args.wavelength),
        n_medium=complex(float(args.n_medium), 0.0),
        lmax=int(args.lmax),
        periodic=periodic,
        polar_angles=pcl.core.uniform_polar_grid(int(args.n_beta)),
        azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(int(args.n_alpha)),
        solver_method=cast(
            Literal["gmres", "bicgstab", "lgmres", "gcrotmk", "direct"],
            args.solver,
        ),
        solver_rtol=float(args.solver_rtol),
        solver_restart=int(args.solver_restart),
        solver_maxiter=int(args.solver_maxiter),
        solver_compute_final_residual=not bool(args.skip_final_residual_check),
        operator_backend=operator_backend,
        coupling_backend=coupling_backend,
        postprocessing_backend=postprocessing_backend,
        compute_dtype=cast(Literal["complex64", "complex128"], args.compute_dtype),
        accum_dtype=cast(Literal["complex64", "complex128"], args.accum_dtype),
        cache_translation_blocks=bool(cache_blocks),
        verbose=not bool(args.quiet),
    )


def _simulation_run_callable(
    cfg: pcl.SimulationConfig,
    source: pcl.PlaneWave,
    particles: Any,
    state: dict[str, Any],
) -> Callable[[], pcl.SimulationResult]:
    def _run() -> pcl.SimulationResult:
        simulation = pcl.Simulation(cfg, particles=particles)
        state["simulation"] = simulation
        return simulation.run(source, include_farfield=False)

    return _run


def _prepared_operator_diagnostics(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return optional hierarchy/memory diagnostics from the completed solve."""

    simulation = state.get("simulation")
    prepared = None if simulation is None else getattr(simulation, "_prepared_operator_cache", None)
    coupling = None if prepared is None else getattr(prepared, "coupling", None)
    if coupling is None:
        return None
    diagnostics: dict[str, Any] = {}
    for name in ("hierarchy_diagnostics", "memory_diagnostics"):
        method = getattr(coupling, name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                diagnostics[name] = value
    return diagnostics or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile pyceles periodic solve and near-field phases for the 500-sphere CELES prototype set."
    )
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--wavelength", type=float, default=550.0)
    parser.add_argument("--n-medium", type=float, default=1.0)
    parser.add_argument("--n-beta", type=int, default=1801)
    parser.add_argument("--n-alpha", type=int, default=360)
    parser.add_argument(
        "--dx", type=float, default=30.0, help="Periodic near-field map spacing in nm."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--side-nm",
        type=float,
        default=DEFAULT_PERIODIC_SIDE_NM,
        help=(
            "Periodic cube side in nm. Default 3000 nm matches the sphere-volume-equivalent "
            "side of the finite 500-sphere CELES cluster rounded up to 3000 nm."
        ),
    )
    parser.add_argument("--max-placement-attempts", type=int, default=10000)
    parser.add_argument("--polarization", choices=("TE", "TM"), default="TE")
    parser.add_argument("--polar-angle", type=float, default=0.0)
    parser.add_argument("--azimuthal-angle", type=float, default=0.0)
    parser.add_argument("--solver", type=str, default="gmres")
    parser.add_argument("--solver-rtol", type=float, default=1.0e-4)
    parser.add_argument("--solver-restart", type=int, default=80)
    parser.add_argument("--solver-maxiter", type=int, default=800)
    parser.add_argument(
        "--coupling-backend",
        choices=("pairwise", "mlfmm"),
        default="pairwise",
        help="Periodic coupling backend to profile.",
    )
    parser.add_argument(
        "--operator-backend",
        choices=("numpy", "cupy"),
        default="numpy",
        help="Prepared periodic operator backend used for the many-body solve.",
    )
    parser.add_argument(
        "--postprocessing-backend",
        choices=("inherit", "numpy", "cupy"),
        default="inherit",
        help="Periodic near-field backend. 'inherit' reuses the operator backend.",
    )
    parser.add_argument(
        "--compute-dtype", choices=("complex64", "complex128"), default="complex128"
    )
    parser.add_argument("--accum-dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument(
        "--cache-mode",
        choices=("off", "on", "both"),
        default="off",
        help="Profile solve with periodic W-block cache off/on/both.",
    )
    parser.add_argument(
        "--periodic-method",
        choices=("ewald", "rayleigh"),
        default="ewald",
        help="Periodic coupling apply: exact pairwise Ewald or hybrid exact-near/Rayleigh-far.",
    )
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--real-shells", type=int, default=None)
    parser.add_argument("--reciprocal-shells", type=int, default=None)
    parser.add_argument("--shell-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-shells", type=int, default=32)
    parser.add_argument("--rayleigh-z-cut", type=float, default=None)
    parser.add_argument("--rayleigh-reciprocal-shells", type=int, default=None)
    parser.add_argument("--output-bmax", type=float, default=None)
    parser.add_argument("--field-bmax", type=float, default=None)
    parser.add_argument("--field-evanescent-decay", type=float, default=8.0)
    parser.add_argument("--nearfield-excess-nm", type=float, default=None)
    parser.add_argument("--nearfield-z-nm", type=float, default=None)
    parser.add_argument(
        "--nearfield-interior-z-nm",
        type=float,
        default=None,
        help=(
            "Interior horizontal xy-plane z coordinate in nm. By default, use the "
            "midpoint of the occupied particle slab; the value must lie strictly "
            "inside that slab."
        ),
    )
    parser.add_argument("--skip-nearfield", action="store_true")
    parser.add_argument("--skip-final-residual-check", action="store_true")
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/profile_periodic"))
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.periodic_method == "rayleigh" and args.cache_mode != "off":
        parser.error("--periodic-method rayleigh requires --cache-mode off")
    if args.coupling_backend == "mlfmm":
        if args.periodic_method != "ewald":
            parser.error("--coupling-backend mlfmm currently requires --periodic-method ewald")
        if args.cache_mode != "off":
            parser.error("--coupling-backend mlfmm currently requires --cache-mode off")

    operator_backend = cast(Literal["numpy", "cupy"], args.operator_backend)
    coupling_backend = cast(Literal["pairwise", "mlfmm"], args.coupling_backend)
    postprocessing_backend = cast(Literal["inherit", "numpy", "cupy"], args.postprocessing_backend)
    compute_dtype = cast(Literal["complex64", "complex128"], args.compute_dtype)
    accum_dtype = cast(Literal["complex64", "complex128"], args.accum_dtype)

    root = _find_repo_root()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prototype_positions, prototype_radii, prototype_n = _load_prototype_geometry(
        root / "examples" / "sphere_parameters.txt"
    )
    cluster_summary = _cluster_volume_side_summary(prototype_positions, prototype_radii)
    geom = _generate_periodic_rsa_geometry(
        prototype_radii=prototype_radii,
        prototype_refractive_indices=prototype_n,
        side_nm=float(args.side_nm),
        seed=int(args.seed),
        max_attempts_per_particle=int(args.max_placement_attempts),
    )
    geometry_files = _save_geometry(out_dir, geom)

    particles = pcl.spheres_from_arrays(
        positions=geom.positions,
        radii=geom.radii,
        refractive_indices=geom.refractive_indices,
    )
    source = pcl.PlaneWave(
        wavelength=float(args.wavelength),
        medium_n=complex(float(args.n_medium), 0.0),
        polarization=cast(Literal["TE", "TM"], args.polarization),
        polar_angle=float(args.polar_angle),
        azimuthal_angle=float(args.azimuthal_angle),
        amplitude=1.0,
    )

    cache_flags: list[bool]
    if args.cache_mode == "off":
        cache_flags = [False]
    elif args.cache_mode == "on":
        cache_flags = [True]
    else:
        cache_flags = [False, True]

    n_particles = int(geom.positions.shape[0])
    nmodes = n_modes(int(args.lmax))
    dense_bytes_est = (n_particles * nmodes) ** 2 * np.dtype(compute_dtype).itemsize
    cache_bytes_est = estimate_translation_cache_bytes(
        n_particles,
        int(args.lmax),
        dtype=np.dtype(compute_dtype),
    )
    if not args.quiet:
        print(
            f"Preparing periodic problem: N={n_particles}, side={float(args.side_nm):g} nm, "
            f"lmax={int(args.lmax)}, method={args.periodic_method}, "
            f"coupling={coupling_backend}, backend={operator_backend}, post={postprocessing_backend}"
        )
        print(
            f"Unknowns={n_particles * nmodes}; dense A raw footprint ~{dense_bytes_est / 1024**3:.2f} GiB"
        )
        if any(cache_flags):
            print(f"Full periodic W-block cache raw payload ~{cache_bytes_est / 1024**3:.2f} GiB")

    phases: list[dict[str, Any]] = []
    solver_runs: list[dict[str, Any]] = []
    primary_run = None
    primary_cache_flag = None

    for cache_on in cache_flags:
        cfg = _build_config(
            args=args,
            side_nm=float(args.side_nm),
            cache_blocks=bool(cache_on),
            coupling_backend=coupling_backend,
            operator_backend=operator_backend,
            postprocessing_backend=postprocessing_backend,
        )
        phase_name = f"solve_cache_{'on' if cache_on else 'off'}"
        if not args.quiet:
            print(f"Profiling phase: {phase_name}")
        simulation_state: dict[str, Any] = {}
        run, phase_summary = _profile_phase(
            phase=phase_name,
            out_dir=out_dir,
            top_n=int(args.top_n),
            fn=_simulation_run_callable(cfg, source, particles, simulation_state),
            synchronize_gpu=operator_backend == "cupy",
            cuda_profiler_api=bool(args.cuda_profiler_api and operator_backend == "cupy"),
        )
        phases.append(phase_summary)
        solver = run.solver_result
        solver_phase_timings = _solver_phase_timings(solver)
        solver_runs.append(
            {
                "operator_backend": operator_backend,
                "coupling_backend": coupling_backend,
                "postprocessing_backend": cfg.resolved_postprocessing_backend(),
                "cache_translation_blocks": bool(cache_on),
                "periodic": _periodic_summary(run.periodic),
                "simulation_phase_timings_s": solver_phase_timings,
                "prepared_operator_diagnostics": _prepared_operator_diagnostics(simulation_state),
                "solver_result": {
                    "method": str(solver.method),
                    "iterations": int(np.asarray(solver.iterations)),
                    "info": int(np.asarray(solver.info)),
                    "relative_residual": float(np.asarray(solver.relative_residual)),
                    "residual_norm": float(np.asarray(solver.residual_norm)),
                },
                "summary": phase_summary,
            }
        )
        if primary_run is None:
            primary_run = run
            primary_cache_flag = bool(cache_on)
            _save_periodic_payload(out_dir / "periodic_orders_primary.npz", run.periodic)

    if primary_run is None:
        raise RuntimeError("No solve run was executed.")

    nearfield_payload: dict[str, Any] | None = None
    if not args.skip_nearfield:
        k = float(primary_run.k)
        field_bmax, field_bmax_info = _resolved_field_bmax(
            field_bmax=args.field_bmax,
            output_bmax=args.output_bmax,
            field_evanescent_decay=float(args.field_evanescent_decay),
            nearfield_excess_nm=args.nearfield_excess_nm,
            geom=geom,
            k=k,
        )
        xy_z = (
            float(args.nearfield_z_nm)
            if args.nearfield_z_nm is not None
            else _default_xy_nearfield_z(geom)
        )
        xy_interior_z = _resolve_xy_interior_nearfield_z(geom, args.nearfield_interior_z_nm)
        zmin, zmax = _nearfield_z_extent(geom, excess_nm=args.nearfield_excess_nm)
        side = float(args.side_nm)
        nearfield_payload = {
            "dx_nm": float(args.dx),
            "field_bmax": float(field_bmax),
            "field_bmax_policy": field_bmax_info,
            "primary_cache_translation_blocks": primary_cache_flag,
            "maps": {},
        }

        if not args.quiet:
            print("Profiling phase: nearfield_xy")
        nf_xy, xy_summary = _profile_phase(
            phase="nearfield_xy",
            out_dir=out_dir,
            top_n=int(args.top_n),
            fn=lambda: pcl.compute_periodic_near_field_slice(
                run=primary_run,
                plane="z",
                plane_value=float(xy_z),
                axis_0_min=-0.5 * side,
                axis_0_max=0.5 * side,
                axis_1_min=-0.5 * side,
                axis_1_max=0.5 * side,
                dx=float(args.dx),
                field_bmax=float(field_bmax),
                center_pixel_policy="none",
                show_progress=not bool(args.quiet),
            ),
            synchronize_gpu=primary_run.config.resolved_postprocessing_backend() == "cupy",
            cuda_profiler_api=bool(
                args.cuda_profiler_api
                and primary_run.config.resolved_postprocessing_backend() == "cupy"
            ),
        )
        phases.append(xy_summary)
        nearfield_payload["maps"]["xy"] = _save_nearfield_npz(
            out_dir / "nearfield_xy_total.npz", nf_xy, field_bmax
        ) | {"summary": xy_summary, "z_nm": float(xy_z)}

        if not args.quiet:
            print("Profiling phase: nearfield_xy_interior")
        nf_xy_interior, xy_interior_summary = _profile_phase(
            phase="nearfield_xy_interior",
            out_dir=out_dir,
            top_n=int(args.top_n),
            fn=lambda: pcl.compute_periodic_near_field_slice(
                run=primary_run,
                plane="z",
                plane_value=float(xy_interior_z),
                axis_0_min=-0.5 * side,
                axis_0_max=0.5 * side,
                axis_1_min=-0.5 * side,
                axis_1_max=0.5 * side,
                dx=float(args.dx),
                field_bmax=float(field_bmax),
                center_pixel_policy="none",
                show_progress=not bool(args.quiet),
            ),
            synchronize_gpu=primary_run.config.resolved_postprocessing_backend() == "cupy",
            cuda_profiler_api=bool(
                args.cuda_profiler_api
                and primary_run.config.resolved_postprocessing_backend() == "cupy"
            ),
        )
        phases.append(xy_interior_summary)
        nearfield_payload["maps"]["xy_interior"] = _save_nearfield_npz(
            out_dir / "nearfield_xy_interior_total.npz", nf_xy_interior, field_bmax
        ) | {"summary": xy_interior_summary, "z_nm": float(xy_interior_z)}

        if not args.quiet:
            print("Profiling phase: nearfield_xz")
        nf_xz, xz_summary = _profile_phase(
            phase="nearfield_xz",
            out_dir=out_dir,
            top_n=int(args.top_n),
            fn=lambda: pcl.compute_periodic_near_field_slice(
                run=primary_run,
                plane="y",
                plane_value=0.0,
                axis_0_min=-0.5 * side,
                axis_0_max=0.5 * side,
                axis_1_min=float(zmin),
                axis_1_max=float(zmax),
                dx=float(args.dx),
                field_bmax=float(field_bmax),
                center_pixel_policy="none",
                show_progress=not bool(args.quiet),
            ),
            synchronize_gpu=primary_run.config.resolved_postprocessing_backend() == "cupy",
            cuda_profiler_api=bool(
                args.cuda_profiler_api
                and primary_run.config.resolved_postprocessing_backend() == "cupy"
            ),
        )
        phases.append(xz_summary)
        nearfield_payload["maps"]["xz"] = _save_nearfield_npz(
            out_dir / "nearfield_xz_total.npz", nf_xz, field_bmax
        ) | {"summary": xz_summary, "y_nm": 0.0, "z_bounds_nm": [float(zmin), float(zmax)]}

    summary = {
        "config": {
            "n_particles": DEFAULT_PARTICLE_COUNT,
            "side_nm": float(args.side_nm),
            "seed": int(args.seed),
            "lmax": int(args.lmax),
            "wavelength": float(args.wavelength),
            "n_medium": float(args.n_medium),
            "n_beta": int(args.n_beta),
            "n_alpha": int(args.n_alpha),
            "dx": float(args.dx),
            "polarization": str(args.polarization),
            "polar_angle": float(args.polar_angle),
            "azimuthal_angle": float(args.azimuthal_angle),
            "solver": str(args.solver),
            "solver_rtol": float(args.solver_rtol),
            "solver_restart": int(args.solver_restart),
            "solver_maxiter": int(args.solver_maxiter),
            "operator_backend": operator_backend,
            "coupling_backend": coupling_backend,
            "postprocessing_backend": postprocessing_backend,
            "effective_postprocessing_backend": primary_run.config.resolved_postprocessing_backend(),
            "compute_dtype": compute_dtype,
            "accum_dtype": accum_dtype,
            "periodic_method": str(args.periodic_method),
            "cache_mode": str(args.cache_mode),
            "eta": None if args.eta is None else float(args.eta),
            "real_shells": None if args.real_shells is None else int(args.real_shells),
            "reciprocal_shells": None
            if args.reciprocal_shells is None
            else int(args.reciprocal_shells),
            "shell_tolerance": float(args.shell_tolerance),
            "max_shells": int(args.max_shells),
            "rayleigh_z_cut": None if args.rayleigh_z_cut is None else float(args.rayleigh_z_cut),
            "rayleigh_reciprocal_shells": None
            if args.rayleigh_reciprocal_shells is None
            else int(args.rayleigh_reciprocal_shells),
            "output_bmax": None if args.output_bmax is None else float(args.output_bmax),
            "field_bmax": None if args.field_bmax is None else float(args.field_bmax),
            "field_evanescent_decay": float(args.field_evanescent_decay),
            "nearfield_interior_z_nm": (
                None
                if args.nearfield_interior_z_nm is None
                else float(args.nearfield_interior_z_nm)
            ),
            "skip_nearfield": bool(args.skip_nearfield),
            "skip_final_residual_check": bool(args.skip_final_residual_check),
            "cuda_profiler_api": bool(args.cuda_profiler_api),
        },
        "source_geometry": {
            "sphere_parameters": str(root / "examples" / "sphere_parameters.txt"),
            "cluster_volume_side_summary": cluster_summary,
            "script_default_side_nm": DEFAULT_PERIODIC_SIDE_NM,
            "script_reference_cluster_diameters_nm": list(DEFAULT_CLUSTER_DIAMETERS_NM),
            "script_reference_volume_matched_side_nm": DEFAULT_VOLUME_MATCHED_SIDE_NM,
        },
        "generated_geometry": {
            "files": geometry_files,
            "attempted_candidates": int(geom.attempted_candidates),
            "rejected_overlaps": int(geom.rejected_overlaps),
            "x_bounds_nm": [-0.5 * float(args.side_nm), 0.5 * float(args.side_nm)],
            "y_bounds_nm": [-0.5 * float(args.side_nm), 0.5 * float(args.side_nm)],
            "z_bounds_nm": [0.0, float(args.side_nm)],
            "radius_min_nm": float(np.min(geom.radii)),
            "radius_max_nm": float(np.max(geom.radii)),
            "radius_mean_nm": float(np.mean(geom.radii)),
        },
        "estimated_dense_A_bytes": int(dense_bytes_est),
        "estimated_W_block_cache_bytes": int(cache_bytes_est),
        "solver_runs": solver_runs,
        "nearfield": nearfield_payload,
        "phases": phases,
    }
    summary_path = out_dir / "profile_periodic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote periodic profiling reports to {out_dir}")
    print(f"Machine-readable summary: {summary_path}")
    print("Phase wall times [s]:")
    for phase in phases:
        print(f"  - {phase['phase']}: {phase['wall_time_s']:.3f}")
    for run in solver_runs:
        timings = run.get("simulation_phase_timings_s", {})
        if not isinstance(timings, dict) or not timings:
            continue
        cache_label = "on" if bool(run["cache_translation_blocks"]) else "off"
        print(f"Solve breakdown [s], cache={cache_label}:")
        for key in (
            "source_projection_s",
            "prepare_operator_s",
            "rhs_Tb_s",
            "periodic_ewald_preparation_s",
            "periodic_rayleigh_preparation_s",
            "periodic_w_block_generation_s",
            "dense_operator_assembly_s",
            "dense_factorization_s",
            "linear_solve_s",
            "solve_sources_core_s",
        ):
            value = timings.get(key)
            if isinstance(value, (int, float)):
                print(f"  - {key}: {float(value):.3f}")


if __name__ == "__main__":
    main()
