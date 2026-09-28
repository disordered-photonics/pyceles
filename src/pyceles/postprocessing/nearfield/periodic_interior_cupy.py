"""CuPy owner for periodic in-slab near-field coefficients.

Exact Ewald batches evaluate all pairs for ``method="ewald"``.  The Rayleigh
path instead uses reciprocal scans for vertically separated sources and retains
exact Ewald only inside the configured z band.  Both paths contract directly
into the local regular ``l=1`` sector needed by field reconstruction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from pyceles._cupy_memory import cupy_allocator_snapshot
from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._optional import import_cupy
from pyceles.core.indexing import n_modes
from pyceles.core.periodic.ewald import resolve_ewald_eta, resolve_ewald_shell_counts
from pyceles.core.periodic.ewald_cupy import (
    CupyEwaldShellWorkspace,
    _shifted_reciprocal_compact_terms_cupy,
    ewald_structural_sums_2d_fixed_cupy,
)
from pyceles.core.periodic.rayleigh import near_point_source_csr, prepare_rayleigh_plan
from pyceles.core.periodic.rayleigh_cupy import (
    apply_rayleigh_far_to_points_cupy,
    scatter_add_complex,
)
from pyceles.core.periodic.scalar import same_plane_z_tolerance

from .periodic_projection import l1_compact_projection_data


def _build_source_projection_cupy(
    *,
    lmax: int,
    coeffs: np.ndarray,
    projection_dtype: np.dtype,
    cupy: Any,
) -> tuple[int, int, Any, Any, Any]:
    """Return compact source-specific ``l=1`` projection data on the GPU.

    ``projection_dtype`` is the reduction precision for the coefficient-weighted
    translation map. Periodic exact-near Ewald values are produced in
    complex128, so callers should use accumulation precision rather than round
    structural channels before this contraction.
    """
    (
        lmax_struct,
        structural_order,
        degree_indices,
        order_indices,
        kernel_np,
    ) = l1_compact_projection_data(int(lmax))
    dtype_cp = cupy.complex64 if projection_dtype == np.dtype(np.complex64) else cupy.complex128
    coeff_cp = cupy.asarray(np.asarray(coeffs, dtype=projection_dtype), dtype=dtype_cp)
    kernel_cp = cupy.asarray(np.asarray(kernel_np, dtype=projection_dtype), dtype=dtype_cp)
    source_projection = cupy.einsum("rck,jc->jkr", kernel_cp, coeff_cp, optimize=True)
    return (
        int(lmax_struct),
        int(structural_order),
        cupy.asarray(degree_indices, dtype=cupy.int32),
        cupy.asarray(order_indices, dtype=cupy.int32),
        source_projection,
    )


_PERIODIC_NEAR_PAIR_BATCH_CAP = 65_536
_PERIODIC_NEAR_WORKSPACE_HEADROOM_FRACTION = 0.5
_SHIFTED_RECIPROCAL_PLANE_MIN_POINTS = 8
_SHIFTED_RECIPROCAL_TERM_CHUNK = 128
_SHIFTED_RECIPROCAL_SOURCE_CHUNK_CAP = 256
_SHIFTED_RECIPROCAL_COMPACT_TEMP_BYTES = 128 * 1024**2


def _repeated_horizontal_planes(
    points: np.ndarray,
    *,
    minimum_points: int = _SHIFTED_RECIPROCAL_PLANE_MIN_POINTS,
) -> list[tuple[float, np.ndarray]]:
    """Return exact repeated-z point groups worth reciprocal plane synthesis.

    The factorization needs only a shared destination height, not a Cartesian
    lateral grid.  This deliberately uses exact floating-point equality: slice
    generators naturally repeat the same z value, while approximately coplanar
    arbitrary point clouds stay on the generic pair path.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] < int(minimum_points):
        return []
    unique_z, inverse, counts = np.unique(pts[:, 2], return_inverse=True, return_counts=True)
    groups: list[tuple[float, np.ndarray]] = []
    for group, count in enumerate(counts):
        if int(count) < int(minimum_points):
            continue
        indices = np.flatnonzero(inverse == group).astype(np.int64, copy=False)
        groups.append((float(unique_z[group]), indices))
    return groups


def _project_shifted_reciprocal_plane_cupy(
    *,
    destination_xy: Any,
    plane_z: float,
    source_positions: Any,
    source_projection: Any,
    workspace: CupyEwaldShellWorkspace,
    reciprocal_shell_count: int,
    structural_order: int,
) -> Any:
    """Project the shifted reciprocal Ewald sum on one destination plane.

    The expensive multipole recurrence is evaluated once per source and
    reciprocal order. Since the destination Fourier phase is independent of
    source index, source contributions are reduced in complex128 *before*
    lateral synthesis. This avoids materializing a ``source x destination x 6``
    tensor and leaves one dense ``destination x reciprocal`` matrix product per
    reciprocal chunk.
    """
    cp = workspace.cupy
    dest_xy = cp.asarray(destination_xy, dtype=cp.float64).reshape(-1, 2)
    sources = cp.asarray(source_positions, dtype=cp.float64).reshape(-1, 3)
    projection = cp.asarray(source_projection, dtype=cp.complex128)
    n_dest = int(dest_xy.shape[0])
    n_sources = int(sources.shape[0])
    out = cp.zeros((n_dest, 6), dtype=cp.complex128)
    if n_dest == 0 or n_sources == 0:
        return out

    reciprocal_terms = workspace.reciprocal_terms(int(reciprocal_shell_count))
    n_terms = int(reciprocal_terms.rho.shape[0])
    compact_width = (int(structural_order) + 1) ** 2
    bytes_per_source = (
        _SHIFTED_RECIPROCAL_TERM_CHUNK * (compact_width + 6) * np.dtype(np.complex128).itemsize
    )
    source_chunk = min(
        _SHIFTED_RECIPROCAL_SOURCE_CHUNK_CAP,
        max(1, _SHIFTED_RECIPROCAL_COMPACT_TEMP_BYTES // max(1, bytes_per_source)),
    )

    for start in range(0, n_terms, _SHIFTED_RECIPROCAL_TERM_CHUNK):
        stop = min(n_terms, start + _SHIFTED_RECIPROCAL_TERM_CHUNK)
        projected_sum = cp.zeros((stop - start, 6), dtype=cp.complex128)
        for source_start in range(0, n_sources, source_chunk):
            source_stop = min(n_sources, source_start + source_chunk)
            compact = _shifted_reciprocal_compact_terms_cupy(
                source_positions=sources[source_start:source_stop],
                plane_z=float(plane_z),
                workspace=workspace,
                reciprocal_shell_count=int(reciprocal_shell_count),
                order=int(structural_order),
                term_start=int(start),
                term_stop=int(stop),
            )
            projected = cp.einsum(
                "stk,skr->str",
                compact,
                projection[source_start:source_stop],
                optimize=True,
            )
            projected_sum += cp.sum(projected, axis=0, dtype=cp.complex128)

        destination_phase = cp.exp(1j * (dest_xy @ reciprocal_terms.kgt[start:stop].T))
        out += destination_phase @ projected_sum
    return out


def _periodic_near_pair_workspace_bytes_per_pair(
    *,
    lmax: int,
    projection_dtype: np.dtype,
) -> int:
    """Conservatively estimate live CuPy workspace for one exact-near pair."""
    lmax_i = max(1, int(lmax))
    structural_order = lmax_i + 1
    rectangular_channels = (structural_order + 1) * (2 * structural_order + 1)
    compact_channels = (structural_order + 1) ** 2
    projection_itemsize = int(np.dtype(projection_dtype).itemsize)
    # Structural sums are evaluated in complex128. The compact contraction also
    # materializes a source-specific (6, K) gather per pair. Keep extra room
    # for relative coordinates, masks, output rows, and allocator retention.
    return int(
        rectangular_channels * np.dtype(np.complex128).itemsize
        + compact_channels * projection_itemsize
        + 6 * compact_channels * projection_itemsize
        + 6 * projection_itemsize
        + 128
    )


def _periodic_near_pair_batch_size_for_workspace(
    *,
    total_pairs: int,
    lmax: int,
    projection_dtype: np.dtype,
    workspace_bytes: int,
) -> int:
    """Choose a nonzero exact-near pair batch within a workspace budget."""
    total = max(1, int(total_pairs))
    bytes_per_pair = _periodic_near_pair_workspace_bytes_per_pair(
        lmax=int(lmax),
        projection_dtype=np.dtype(projection_dtype),
    )
    memory_cap = max(1, int(workspace_bytes) // max(1, bytes_per_pair))
    return max(1, min(total, _PERIODIC_NEAR_PAIR_BATCH_CAP, memory_cap))


def _cupy_periodic_near_pair_batch_size(
    *,
    cupy: Any,
    total_pairs: int,
    lmax: int,
    projection_dtype: np.dtype,
) -> int:
    """Resolve an exact-near batch from current allocator headroom without mutation."""
    snapshot = cupy_allocator_snapshot(cupy, apply_pool_limit=False)
    reusable_or_fresh = int(snapshot.raw_free_bytes) + int(snapshot.pool_free_bytes)
    usable_headroom = max(
        1,
        min(int(snapshot.active_headroom_bytes), int(reusable_or_fresh)),
    )
    workspace_bytes = max(
        1,
        int(float(usable_headroom) * _PERIODIC_NEAR_WORKSPACE_HEADROOM_FRACTION),
    )
    return _periodic_near_pair_batch_size_for_workspace(
        total_pairs=int(total_pairs),
        lmax=int(lmax),
        projection_dtype=np.dtype(projection_dtype),
        workspace_bytes=int(workspace_bytes),
    )


def _resolve_eta(
    *,
    periodic: Any,
    k: float,
    k_parallel: np.ndarray,
    positions: np.ndarray,
    lmax: int,
    max_vertical_offset: float | None = None,
) -> float:
    return resolve_ewald_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        lmax=int(lmax),
        max_vertical_offset=max_vertical_offset,
    )


def periodic_local_regular_l1_coeffs_cupy(
    *,
    points: np.ndarray,
    positions: np.ndarray,
    coeffs: np.ndarray,
    lmax: int,
    k: float,
    periodic: Any,
    k_parallel: np.ndarray,
    circumscribing_radii: np.ndarray | None = None,
    point_batch_size: int = 128,
    source_batch_size: int | None = None,
    show_progress: bool = False,
    compute_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike = np.complex128,
) -> np.ndarray:
    """Return point-local regular ``l=1`` coefficients using the CuPy path.

    The function is intentionally specialized to periodic in-slab near fields.
    It uses explicit shell ranges when supplied, otherwise a centralized
    one-shot shell resolver chooses fixed ranges compatible with the NumPy
    adaptive options. With ``method='rayleigh'``, reciprocal scans evaluate
    vertically separated source-point pairs and only the exact near band enters
    the Ewald device loop.
    """
    cp, _ = import_cupy()
    compute_dtype, accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=compute_dtype,
        accum_dtype=accum_dtype,
    )
    accum_dtype_cp = cp.complex64 if accum_dtype == np.dtype(np.complex64) else cp.complex128
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    coeff_arr = np.asarray(coeffs, dtype=compute_dtype).reshape(pos.shape[0], n_modes(int(lmax)))
    out = np.zeros((pts.shape[0], 6), dtype=accum_dtype)
    if pts.shape[0] == 0 or pos.shape[0] == 0:
        return out
    method = str(periodic.options.method)
    if method not in {"ewald", "rayleigh"}:
        raise NotImplementedError(
            "Periodic in-slab near-field evaluation requires "
            "`periodic.options.method` to be 'ewald' or 'rayleigh'."
        )

    point_batch_size = max(1, int(point_batch_size))
    if source_batch_size is None:
        # The dense Ewald-only path still uses a rectangular point/source
        # batch. The hybrid Rayleigh path resolves its sparse pair batch from
        # allocator headroom below.
        source_batch_size = max(1, min(pos.shape[0], 2048 // point_batch_size))
    source_batch_size = max(1, int(source_batch_size))

    (
        lmax_struct,
        structural_order,
        degree_indices_cp,
        order_indices_cp,
        source_projection_cp,
    ) = _build_source_projection_cupy(
        lmax=int(lmax),
        coeffs=coeff_arr,
        projection_dtype=accum_dtype,
        cupy=cp,
    )

    rayleigh_z_cut: float | None = None
    if method == "rayleigh":
        plan = prepare_rayleigh_plan(
            lmax=int(lmax),
            k=float(k),
            positions=pos,
            circumscribing_radii=circumscribing_radii,
            periodic=periodic,
            k_parallel=k_parallel,
            dtype=compute_dtype,
        )
        local_cp = apply_rayleigh_far_to_points_cupy(
            plan,
            coeff_arr,
            pts,
            cupy=cp,
        ).astype(accum_dtype_cp, copy=False)
        _indptr, destination_indices, source_indices = near_point_source_csr(
            pts,
            pos,
            plan.z_cut,
        )
        rayleigh_z_cut = float(plan.z_cut)
        if destination_indices.size == 0:
            return np.asarray(cp.asnumpy(local_cp), dtype=accum_dtype)
    else:
        local_cp = None
        destination_indices = np.zeros((0,), dtype=np.int64)
        source_indices = np.zeros((0,), dtype=np.int64)

    eta = _resolve_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=pos,
        lmax=int(lmax),
        max_vertical_offset=rayleigh_z_cut,
    )

    shell_counts = resolve_ewald_shell_counts(
        periodic=periodic,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        positions=pos,
        lmax=int(lmax),
        eta=float(eta),
        max_vertical_offset=rayleigh_z_cut,
    )
    real_count = int(shell_counts.real_shells)
    recip_count = int(shell_counts.reciprocal_shells)
    coordinate_scale = float(
        max(
            np.max(np.abs(pos[:, 2]), initial=0.0),
            np.max(np.abs(pts[:, 2]), initial=0.0),
        )
    )
    workspace = CupyEwaldShellWorkspace(
        cupy=cp,
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=np.float64).reshape(2),
        eta=float(eta),
    )

    if method == "rayleigh":
        if local_cp is None:
            raise RuntimeError("Rayleigh near-field initialization failed.")
        pts_cp = cp.asarray(pts, dtype=cp.float64)
        pos_cp = cp.asarray(pos, dtype=cp.float64)

        # Horizontal slices admit a stronger exact factorization than the
        # generic point/source kernel.  The shifted reciprocal contribution is
        # a Fourier series in destination xy, while its expensive multipole
        # recurrence depends on the source and the shared plane z only.
        same_plane_atol = same_plane_z_tolerance(
            float(k),
            coordinate_scale=float(coordinate_scale),
        )
        plane_groups: list[tuple[float, np.ndarray, np.ndarray]] = []
        fast_destinations = np.zeros((pts.shape[0],), dtype=bool)
        if rayleigh_z_cut is not None:
            for plane_z, destination_group in _repeated_horizontal_planes(pts):
                near_sources = np.flatnonzero(
                    np.abs(pos[:, 2] - float(plane_z)) <= float(rayleigh_z_cut)
                ).astype(np.int64, copy=False)
                if near_sources.size == 0:
                    continue
                shifted_sources = near_sources[
                    np.abs(pos[near_sources, 2] - float(plane_z)) > float(same_plane_atol)
                ]
                if shifted_sources.size == 0:
                    continue
                fast_destinations[destination_group] = True
                plane_groups.append((float(plane_z), destination_group, shifted_sources))

        fast_pairs = fast_destinations[destination_indices]
        # Exact Ewald structural values are complex128. Contract them in the
        # configured accumulation precision instead of rounding to compute_dtype
        # before the reduction. This keeps the generic and factorized paths on
        # the same mixed-precision boundary.
        pair_batch_size = _cupy_periodic_near_pair_batch_size(
            cupy=cp,
            total_pairs=int(destination_indices.size),
            lmax=int(lmax),
            projection_dtype=accum_dtype,
        )
        progress = None
        if show_progress:
            from tqdm.auto import tqdm

            slow_count = int(np.count_nonzero(~fast_pairs))
            fast_count = int(np.count_nonzero(fast_pairs))
            pair_steps = (slow_count + pair_batch_size - 1) // pair_batch_size
            pair_steps += (fast_count + pair_batch_size - 1) // pair_batch_size
            progress = tqdm(
                total=int(pair_steps + len(plane_groups)),
                desc="Periodic slab field (CuPy)",
                unit="near-batch",
                leave=True,
            )

        def accumulate_pairs(
            destinations: np.ndarray,
            sources: np.ndarray,
            *,
            include_shifted_reciprocal: bool,
        ) -> None:
            if destinations.size == 0:
                return
            destinations_cp = cp.asarray(destinations, dtype=cp.int32)
            sources_cp = cp.asarray(sources, dtype=cp.int32)
            for pair_start in range(0, destinations.size, pair_batch_size):
                pair_stop = min(destinations.size, pair_start + pair_batch_size)
                dst = destinations_cp[pair_start:pair_stop]
                src = sources_cp[pair_start:pair_stop]
                rel_cp = pos_cp[src] - pts_cp[dst]
                sums = ewald_structural_sums_2d_fixed_cupy(
                    relative_source_minus_destination=rel_cp,
                    lmax_struct=int(lmax_struct),
                    structural_order=int(structural_order),
                    workspace=workspace,
                    real_shell_count=int(real_count),
                    reciprocal_shell_count=int(recip_count),
                    coordinate_scale=coordinate_scale,
                    include_shifted_reciprocal=bool(include_shifted_reciprocal),
                )
                compact = sums[:, degree_indices_cp, order_indices_cp].astype(
                    accum_dtype_cp, copy=False
                )
                pair_projection = source_projection_cp[src]
                pair_l1 = cp.einsum(
                    "nk,nkr->nr",
                    compact,
                    pair_projection,
                    optimize=True,
                )
                scatter_add_complex(local_cp, dst, pair_l1, cupy=cp)
                if progress is not None:
                    progress.update(1)

        try:
            # Unstructured destinations retain the established complete Ewald
            # path.  Plane-group pairs still need the real-space and any
            # same-plane reciprocal pieces, but their shifted reciprocal term
            # is supplied once per plane below.
            accumulate_pairs(
                destination_indices[~fast_pairs],
                source_indices[~fast_pairs],
                include_shifted_reciprocal=True,
            )
            accumulate_pairs(
                destination_indices[fast_pairs],
                source_indices[fast_pairs],
                include_shifted_reciprocal=False,
            )

            if plane_groups:
                for plane_z, destination_group, shifted_sources in plane_groups:
                    destination_group_cp = cp.asarray(destination_group, dtype=cp.int32)
                    shifted_sources_cp = cp.asarray(shifted_sources, dtype=cp.int32)
                    plane_values = _project_shifted_reciprocal_plane_cupy(
                        destination_xy=pts_cp[destination_group_cp, :2],
                        plane_z=float(plane_z),
                        source_positions=pos_cp[shifted_sources_cp],
                        source_projection=source_projection_cp[shifted_sources_cp],
                        workspace=workspace,
                        reciprocal_shell_count=int(recip_count),
                        structural_order=int(structural_order),
                    )
                    local_cp[destination_group_cp] += plane_values.astype(
                        accum_dtype_cp,
                        copy=False,
                    )
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()
        return np.asarray(cp.asnumpy(local_cp), dtype=accum_dtype)

    # Pure Ewald has the same shifted-reciprocal plane structure as the
    # Rayleigh-hybrid path.  For repeated horizontal destinations, evaluate
    # real-space and same-plane reciprocal terms through the ordinary batched
    # evaluator, then synthesize the off-plane shifted reciprocal term from its
    # source-reduced Fourier coefficients.  This branch is deliberately kept
    # separate from the Rayleigh sparse-pair path: pure Ewald has no z-cut and
    # therefore every source contributes to every destination.
    if method == "ewald":
        same_plane_atol = same_plane_z_tolerance(
            float(k),
            coordinate_scale=float(coordinate_scale),
        )
        ewald_plane_groups: list[tuple[float, np.ndarray, np.ndarray]] = []
        ewald_fast_destinations = np.zeros((pts.shape[0],), dtype=bool)
        for plane_z, destination_group in _repeated_horizontal_planes(pts):
            shifted_sources = np.flatnonzero(
                np.abs(pos[:, 2] - float(plane_z)) > float(same_plane_atol)
            ).astype(np.int64, copy=False)
            if shifted_sources.size == 0:
                continue
            ewald_fast_destinations[destination_group] = True
            ewald_plane_groups.append((float(plane_z), destination_group, shifted_sources))

        if ewald_plane_groups:
            pts_cp = cp.asarray(pts, dtype=cp.float64)
            pos_cp = cp.asarray(pos, dtype=cp.float64)
            local_ewald_cp = cp.zeros((pts.shape[0], 6), dtype=accum_dtype_cp)

            def accumulate_dense(
                destination_indices_dense: np.ndarray,
                *,
                include_shifted_reciprocal: bool,
            ) -> None:
                if destination_indices_dense.size == 0:
                    return
                for p0 in range(0, destination_indices_dense.size, point_batch_size):
                    p1 = min(destination_indices_dense.size, p0 + point_batch_size)
                    destination_batch = destination_indices_dense[p0:p1]
                    pts_batch = np.asarray(pts[destination_batch], dtype=np.float64)
                    n_points = int(pts_batch.shape[0])
                    local_batch = cp.zeros((n_points, 6), dtype=accum_dtype_cp)
                    for s0 in range(0, pos.shape[0], source_batch_size):
                        s1 = min(pos.shape[0], s0 + source_batch_size)
                        src_batch = np.asarray(pos[s0:s1], dtype=np.float64)
                        n_sources_batch = int(src_batch.shape[0])
                        rel = src_batch[None, :, :] - pts_batch[:, None, :]
                        rel_cp = cp.asarray(rel.reshape(-1, 3), dtype=cp.float64)
                        sums = ewald_structural_sums_2d_fixed_cupy(
                            relative_source_minus_destination=rel_cp,
                            lmax_struct=int(lmax_struct),
                            structural_order=int(structural_order),
                            workspace=workspace,
                            real_shell_count=int(real_count),
                            reciprocal_shell_count=int(recip_count),
                            coordinate_scale=float(coordinate_scale),
                            include_shifted_reciprocal=bool(include_shifted_reciprocal),
                        )
                        compact = (
                            sums[:, degree_indices_cp, order_indices_cp]
                            .astype(accum_dtype_cp, copy=False)
                            .reshape(n_points, n_sources_batch, -1)
                        )
                        local_batch += cp.einsum(
                            "psk,skr->pr",
                            compact,
                            source_projection_cp[s0:s1],
                            optimize=True,
                        )
                    local_ewald_cp[cp.asarray(destination_batch, dtype=cp.int64)] = local_batch

            accumulate_dense(
                np.flatnonzero(~ewald_fast_destinations).astype(np.int64, copy=False),
                include_shifted_reciprocal=True,
            )
            accumulate_dense(
                np.flatnonzero(ewald_fast_destinations).astype(np.int64, copy=False),
                include_shifted_reciprocal=False,
            )

            for plane_z, destination_group, shifted_sources in ewald_plane_groups:
                destination_group_cp = cp.asarray(destination_group, dtype=cp.int32)
                shifted_sources_cp = cp.asarray(shifted_sources, dtype=cp.int32)
                plane_values = _project_shifted_reciprocal_plane_cupy(
                    destination_xy=pts_cp[destination_group_cp, :2],
                    plane_z=float(plane_z),
                    source_positions=pos_cp[shifted_sources_cp],
                    source_projection=source_projection_cp[shifted_sources_cp],
                    workspace=workspace,
                    reciprocal_shell_count=int(recip_count),
                    structural_order=int(structural_order),
                )
                local_ewald_cp[destination_group_cp] += plane_values.astype(
                    accum_dtype_cp,
                    copy=False,
                )
            return np.asarray(cp.asnumpy(local_ewald_cp), dtype=accum_dtype)

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            n_point_batches = (pts.shape[0] + point_batch_size - 1) // point_batch_size
            n_source_batches = (pos.shape[0] + source_batch_size - 1) // source_batch_size
            progress = tqdm(
                total=int(n_point_batches * n_source_batches),
                desc="Periodic slab field (CuPy)",
                unit="batch",
                leave=True,
            )
        except Exception:  # pragma: no cover - progress is diagnostic only
            progress = None

    try:
        for p0 in range(0, pts.shape[0], point_batch_size):
            p1 = min(pts.shape[0], p0 + point_batch_size)
            pts_batch = np.asarray(pts[p0:p1], dtype=np.float64)
            local_cp = cp.zeros((pts_batch.shape[0], 6), dtype=accum_dtype_cp)
            for s0 in range(0, pos.shape[0], source_batch_size):
                s1 = min(pos.shape[0], s0 + source_batch_size)
                src_batch = np.asarray(pos[s0:s1], dtype=np.float64)
                rel = src_batch[None, :, :] - pts_batch[:, None, :]
                n_points = int(pts_batch.shape[0])
                n_sources = int(src_batch.shape[0])
                rel_cp = cp.asarray(rel.reshape(-1, 3), dtype=cp.float64)
                sums = ewald_structural_sums_2d_fixed_cupy(
                    relative_source_minus_destination=rel_cp,
                    lmax_struct=int(lmax_struct),
                    structural_order=int(structural_order),
                    workspace=workspace,
                    real_shell_count=int(real_count),
                    reciprocal_shell_count=int(recip_count),
                    coordinate_scale=float(coordinate_scale),
                )
                compact = (
                    sums[:, degree_indices_cp, order_indices_cp]
                    .astype(accum_dtype_cp, copy=False)
                    .reshape(n_points, n_sources, -1)
                )
                local_cp += cp.einsum(
                    "psk,skr->pr",
                    compact,
                    source_projection_cp[s0:s1],
                    optimize=True,
                )
                if progress is not None:
                    progress.update(1)
            out[p0:p1] = cp.asnumpy(local_cp)
    finally:
        if progress is not None:
            progress.close()
    return out


__all__ = ["periodic_local_regular_l1_coeffs_cupy"]
