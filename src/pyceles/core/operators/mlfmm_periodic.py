"""Prepared periodization of the pyceles MLFMM hierarchy.

The construction closes the infinite two-dimensional lattice at the sampled
MLFMM level with the safest near-image removal margin over the actual occupied
box offsets.  Same-plane structural sums use Ewald evaluation, while vertically
shifted sums use their exact reciprocal Rayleigh representation.  The finite
set of image interactions that are not yet well separated is descended through
the ordinary occupied hierarchy; well-separated descendants use the existing
sampled Rokhlin translator and the residual leaf pairs remain exact.

The repeated apply is therefore mesh-free and contains neither a particle-pair
Ewald cache nor a particle-level Rayleigh table.  Periodic summation is used
only during preparation to build a small family of coarse-box diagonals.
"""

from __future__ import annotations

import math
from dataclasses import replace
from time import perf_counter
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.periodic import PeriodicSpec
from pyceles.core.periodic.ewald import (
    EwaldShellWorkspace,
    ewald_self_correction,
    ewald_structural_sums_2d_batch,
    resolve_ewald_eta,
)
from pyceles.core.periodic.rayleigh import rayleigh_structural_sums_2d_batch
from pyceles.core.periodic.scalar import same_plane_z_tolerance, structural_sum_m_normalization
from pyceles.core.periodic.structural import free_space_structural_sums
from pyceles.core.spherical import legendre_normalized_trigon
from pyceles.core.translation import RadialLUT

from .mlfmm import (
    COMPLEX128_DTYPE,
    MLFMMCouplingOperator,
    MLFMMMultilevelOperators,
    MLFMMOptions,
    MLFMMPeriodicFarBatch,
    MLFMMPeriodicLeafBatch,
    MLFMMPeriodizationPlan,
    MLFMMResolvedPlan,
    _sampled_rokhlin_translator,
    build_multilevel_mlfmm_operators,
    resolve_mlfmm_plan,
)

Array = np.ndarray
Offset3 = tuple[int, int, int]
LatticeIndex = tuple[int, int]


def _compact_structural_table(raw: Array, *, max_degree: int) -> Array:
    """Recenter a dense structural table on the requested maximum degree."""

    degree_max = int(max_degree)
    values = np.asarray(raw, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError(f"Structural table must be two-dimensional. Got {values.shape}.")
    raw_degree = int(values.shape[0] - 1)
    raw_offset = int((values.shape[1] - 1) // 2)
    if raw_degree < degree_max or raw_offset < degree_max:
        raise ValueError(
            f"Structural table degree {raw_degree} does not cover requested {degree_max}."
        )
    compact = np.zeros((degree_max + 1, 2 * degree_max + 1), dtype=np.complex128)
    target_offset = degree_max
    for degree in range(degree_max + 1):
        compact[degree, target_offset - degree : target_offset + degree + 1] = values[
            degree, raw_offset - degree : raw_offset + degree + 1
        ]
    return compact


def sampled_transfer_from_structural_sums(
    structural_sums: Array,
    *,
    directions: Array,
    weights: Array,
    dtype: np.dtype = COMPLEX128_DTYPE,
) -> Array:
    """Evaluate a sampled Rokhlin translator from scalar structural sums.

    For one displacement this map is algebraically identical to
    :func:`_sampled_rokhlin_translator`.  Linearity then turns an Ewald lattice
    sum into the same diagonal sampled representation used by ordinary MLFMM
    M2L interactions.  No Cartesian multipole or plane-wave basis is exposed
    outside the existing directional hierarchy.
    """

    return np.asarray(
        sampled_transfers_from_structural_sums(
            np.asarray(structural_sums, dtype=np.complex128)[None, ...],
            directions=directions,
            weights=weights,
            dtype=dtype,
        )[0],
        dtype=np.dtype(dtype),
    )


def sampled_transfers_from_structural_sums(
    structural_sums: Array,
    *,
    directions: Array,
    weights: Array,
    dtype: np.dtype = COMPLEX128_DTYPE,
    matmul_backend: Literal["numpy", "cupy"] = "numpy",
) -> Array:
    """Evaluate sampled translators for a batch of structural tables."""
    if matmul_backend not in {"numpy", "cupy"}:
        raise ValueError(f"`matmul_backend` must be 'numpy' or 'cupy'. Got {matmul_backend!r}.")
    sums = np.asarray(structural_sums, dtype=np.complex128)
    if sums.ndim != 3 or sums.shape[2] != 2 * sums.shape[1] - 1:
        raise ValueError(f"structural_sums must have shape (N, L+1, 2*L+1). Got {sums.shape}.")
    degree_max = int(sums.shape[1] - 1)
    offset = degree_max
    dirs = np.asarray(directions, dtype=float).reshape(-1, 3)
    quadrature_weights = np.asarray(weights, dtype=float).reshape(-1)
    if quadrature_weights.size != dirs.shape[0]:
        raise ValueError("Directional weights must match the direction count.")

    ct = np.asarray(dirs[:, 2], dtype=float)
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    plm = legendre_normalized_trigon(ct, st, degree_max, xp=np)
    channel_count = (degree_max + 1) ** 2
    basis = np.empty((dirs.shape[0], channel_count), dtype=np.complex128)
    coefficients = np.empty((sums.shape[0], channel_count), dtype=np.complex128)
    channel = 0
    for degree in range(degree_max + 1):
        degree_phase = 4.0 * np.pi * (1j**degree)
        for order in range(-degree, degree + 1):
            ylm = (
                plm[degree, abs(order)]
                * np.exp(1j * float(order) * phi)
                / structural_sum_m_normalization(order)
            )
            coefficients[:, channel] = (
                ((-1.0) ** order)
                * sums[:, degree, -order + offset]
                / structural_sum_m_normalization(-order)
            )
            basis[:, channel] = degree_phase * ylm
            channel += 1
    if matmul_backend == "numpy":
        transfer = basis @ coefficients.T
    else:
        from pyceles._optional import import_cupy

        cupy, _ = import_cupy()
        transfer = cupy.asnumpy(cupy.asarray(basis) @ cupy.asarray(coefficients.T))
    return np.asarray(
        transfer.T * quadrature_weights[None, :],
        dtype=np.dtype(dtype),
    )


def _finite_image_structural_sums(
    *,
    max_degree: int,
    k: float,
    base_displacement: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    lattice_indices: Array,
    excluded_singular_image: LatticeIndex | None = None,
) -> Array:
    """Sum selected image translations for one coarse-box displacement."""

    degree_max = int(max_degree)
    result = np.zeros((degree_max + 1, 2 * degree_max + 1), dtype=np.complex128)
    base = np.asarray(base_displacement, dtype=float).reshape(3)
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    for p_raw, q_raw in np.asarray(lattice_indices, dtype=np.int64).reshape(-1, 2):
        p = int(p_raw)
        q = int(q_raw)
        if p == 0 and q == 0:
            continue
        if excluded_singular_image == (p, q):
            # The Ewald table is regularized by removing this coincident
            # expansion-center image and adding the corresponding self term.
            # Its particle-level interaction is descended exactly instead.
            continue
        shift = lattice.lattice_vector(p, q)
        phase = complex(np.exp(1j * float(np.dot(kp, shift[:2]))))
        result += phase * free_space_structural_sums(
            max_degree=degree_max,
            k=float(k),
            displacement=base - shift,
        )
    return result


def _lattice_equivalent_image(
    displacement: Array,
    *,
    lattice: RectangularLattice2D,
) -> LatticeIndex | None:
    """Return the image whose lattice vector matches one displacement.

    Coarse expansion centers can coincide after a lattice translation even
    when every physical particle remains disjoint from its replicas.  The
    corresponding scalar periodic sum has a removable self singularity at the
    expansion-center level and must be regularized before its finite near image
    is descended through the hierarchy.
    """

    delta = np.asarray(displacement, dtype=float).reshape(3)
    p = int(np.rint(float(delta[0]) / float(lattice.ax)))
    q = int(np.rint(float(delta[1]) / float(lattice.ay)))
    shift = lattice.lattice_vector(p, q)
    residual = delta - shift
    scale = max(
        1.0,
        abs(float(delta[0])),
        abs(float(delta[1])),
        abs(float(delta[2])),
        float(lattice.ax),
        float(lattice.ay),
    )
    tolerance = 64.0 * np.finfo(float).eps * scale
    if bool(np.max(np.abs(residual), initial=0.0) <= tolerance):
        return p, q
    return None


def _boxes_well_separated(delta: Array, *, half_size: float) -> bool:
    """Return the conservative same-level MLFMM separation predicate."""

    delta_arr = np.asarray(delta, dtype=float).reshape(3)
    threshold = 4.0 * float(half_size)
    scale = max(1.0, threshold, float(np.max(np.abs(delta_arr), initial=0.0)))
    tolerance = 64.0 * np.finfo(float).eps * scale
    return bool(np.any(np.abs(delta_arr) + tolerance >= threshold))


def _translated_leaf_pair_max_distance(
    *,
    source_min: Array,
    source_max: Array,
    destination_min: Array,
    destination_max: Array,
    lattice_shift: Array,
) -> float:
    """Return an AABB bound for one directed periodic leaf-pair distance."""

    shift = np.asarray(lattice_shift, dtype=float).reshape(3)
    image_source_min = np.asarray(source_min, dtype=float).reshape(3) + shift
    image_source_max = np.asarray(source_max, dtype=float).reshape(3) + shift
    destination_min_arr = np.asarray(destination_min, dtype=float).reshape(3)
    destination_max_arr = np.asarray(destination_max, dtype=float).reshape(3)
    axis_extent = np.maximum(
        np.abs(image_source_max - destination_min_arr),
        np.abs(destination_max_arr - image_source_min),
    )
    return float(np.linalg.norm(axis_extent))


def _near_lattice_indices_for_offset(
    *,
    base_displacement: Array,
    half_size: float,
    lattice: RectangularLattice2D,
) -> Array:
    """Return nonzero images not yet well separated for one box offset."""

    delta = np.asarray(base_displacement, dtype=float).reshape(3)
    threshold = 4.0 * float(half_size)
    if abs(float(delta[2])) >= threshold:
        return np.zeros((0, 2), dtype=np.int32)

    p_min = math.floor((float(delta[0]) - threshold) / float(lattice.ax)) - 1
    p_max = math.ceil((float(delta[0]) + threshold) / float(lattice.ax)) + 1
    q_min = math.floor((float(delta[1]) - threshold) / float(lattice.ay)) - 1
    q_max = math.ceil((float(delta[1]) + threshold) / float(lattice.ay)) + 1
    near: list[LatticeIndex] = []
    for p in range(p_min, p_max + 1):
        for q in range(q_min, q_max + 1):
            if p == 0 and q == 0:
                continue
            shift = lattice.lattice_vector(p, q)
            if not _boxes_well_separated(delta - shift, half_size=float(half_size)):
                near.append((p, q))
    near.sort(key=lambda item: (max(abs(item[0]), abs(item[1])), item[0], item[1]))
    return np.asarray(near, dtype=np.int32).reshape(-1, 2)


def _coarse_pair_groups(
    operators: MLFMMMultilevelOperators,
    *,
    closure_level: int,
) -> dict[Offset3, tuple[list[int], list[int], Array]]:
    """Group every ordered coarse-box pair by its translation-invariant offset."""

    level = operators.levels[int(closure_level)]
    groups: dict[Offset3, tuple[list[int], list[int], Array]] = {}
    for source_index in range(int(level.coords.shape[0])):
        for destination_index in range(int(level.coords.shape[0])):
            coord_delta_arr = np.asarray(
                level.coords[destination_index] - level.coords[source_index], dtype=np.int64
            )
            key = (
                int(coord_delta_arr[0]),
                int(coord_delta_arr[1]),
                int(coord_delta_arr[2]),
            )
            grouped = groups.get(key)
            if grouped is None:
                groups[key] = (
                    [source_index],
                    [destination_index],
                    np.asarray(
                        level.centers[destination_index] - level.centers[source_index],
                        dtype=float,
                    ),
                )
            else:
                grouped[0].append(source_index)
                grouped[1].append(destination_index)
    return groups


def _periodic_nonzero_structural_sums(
    *,
    max_degree: int,
    k: float,
    periodic: PeriodicSpec,
    k_parallel: Array,
    offsets: list[Offset3],
    displacements: Array,
    matmul_backend: Literal["numpy", "cupy"] = "numpy",
) -> tuple[dict[Offset3, Array], dict[Offset3, LatticeIndex | None], float]:
    """Evaluate all nonzero lattice images for coarse-box offsets in one batch.

    Same-plane offsets retain Ewald summation.  Shifted offsets use the exact
    reciprocal Rayleigh representation, which is both faster and numerically
    well scaled for the vertically separated boxes in a tall periodic cell.
    """

    degree_max = int(max_degree)
    structural_lmax = int((degree_max + 1) // 2)
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    deltas = np.asarray(displacements, dtype=float).reshape(-1, 3)
    z_atol = same_plane_z_tolerance(
        float(k),
        coordinate_scale=float(np.max(np.abs(deltas), initial=0.0)),
    )
    same_plane = np.abs(deltas[:, 2]) <= z_atol
    eta_probe_positions = np.vstack(
        (
            np.zeros((1, 3), dtype=float),
            deltas[same_plane],
        )
    )
    eta = resolve_ewald_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=kp,
        positions=eta_probe_positions,
        # The periodizer requests structural degrees set by the MLFMM box
        # order, which can be much higher than the particle lmax.  Eta
        # preflight must therefore probe the actual closure degree.
        lmax=structural_lmax,
        max_vertical_offset=0.0,
    )
    workspace = EwaldShellWorkspace(
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=kp,
        eta=float(eta),
    )
    singular_images: dict[Offset3, LatticeIndex | None] = {}
    evaluation_deltas = np.array(deltas, dtype=float, copy=True)
    for index, key in enumerate(offsets):
        singular = _lattice_equivalent_image(deltas[index], lattice=periodic.lattice)
        singular_images[key] = singular
        if singular is not None:
            # Snap an ULP-level lattice coincidence to the exact image so the
            # Ewald real-space branch removes the singular term deterministically.
            evaluation_deltas[index] = periodic.lattice.lattice_vector(*singular)
    raw_batch = np.zeros(
        (len(offsets), 2 * structural_lmax + 1, 4 * structural_lmax + 1),
        dtype=np.complex128,
    )
    if np.any(same_plane):
        raw_batch[same_plane] = ewald_structural_sums_2d_batch(
            lmax_struct=structural_lmax,
            k=float(k),
            destinations=evaluation_deltas[same_plane],
            source=np.zeros((3,), dtype=float),
            lattice=periodic.lattice,
            k_parallel=kp,
            eta=float(eta),
            real_shells=periodic.options.real_shells,
            reciprocal_shells=periodic.options.reciprocal_shells,
            shell_tolerance=float(periodic.options.shell_tolerance),
            max_shells=int(periodic.options.max_shells),
            dtype=np.complex128,
            workspace=workspace,
        )
    shifted = ~same_plane
    if np.any(shifted):
        raw_batch[shifted] = rayleigh_structural_sums_2d_batch(
            max_degree=2 * structural_lmax,
            k=float(k),
            displacements=evaluation_deltas[shifted],
            lattice=periodic.lattice,
            k_parallel=kp,
            tolerance=float(periodic.options.shell_tolerance),
            max_shells=int(periodic.options.max_shells),
            requested_half_width=None,
            matmul_backend=matmul_backend,
        )
    raw_order = int(raw_batch.shape[1] - 1)
    result: dict[Offset3, Array] = {}
    for index, key in enumerate(offsets):
        raw = np.asarray(raw_batch[index], dtype=np.complex128).copy()
        delta = deltas[index]
        singular = singular_images[key]
        if singular is not None:
            shift = periodic.lattice.lattice_vector(*singular)
            phase = complex(np.exp(1j * float(np.dot(kp, shift[:2]))))
            raw[0, raw_order] += (
                phase
                * structural_sum_m_normalization(0)
                * ewald_self_correction(float(k), float(eta))
            )
        compact = _compact_structural_table(raw, max_degree=degree_max)
        if key != (0, 0, 0):
            compact -= free_space_structural_sums(
                max_degree=degree_max,
                k=float(k),
                displacement=delta,
            )
        result[key] = np.asarray(compact, dtype=np.complex128)
    return result, singular_images, float(eta)


def _descend_near_image_pairs(
    *,
    k: float,
    positions: Array,
    operators: MLFMMMultilevelOperators,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    closure_level: int,
    coarse_groups: dict[Offset3, tuple[list[int], list[int], Array]],
    near_by_offset: dict[Offset3, Array],
) -> tuple[
    tuple[tuple[MLFMMPeriodicFarBatch, ...], ...],
    tuple[MLFMMPeriodicLeafBatch, ...],
    int,
    int,
    float,
]:
    """Descend the finite non-well-separated image correction through the tree."""

    levels = operators.levels
    leaf_level = int(operators.leaf_level)
    far_grouped: list[
        dict[tuple[int, int, int, int, int], tuple[list[int], list[int], Array, complex]]
    ] = [dict() for _ in levels]
    leaf_grouped: dict[LatticeIndex, tuple[list[int], list[int], Array, complex]] = {}
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    half_sizes = tuple(
        float(operators.partition.root_half_size) / float(1 << level_index)
        for level_index in range(len(levels))
    )
    image_metadata: dict[LatticeIndex, tuple[Array, complex]] = {}

    stack: list[tuple[int, int, int, int, int]] = []
    for offset, (sources, destinations, _delta) in coarse_groups.items():
        near_indices = near_by_offset[offset]
        for p_raw, q_raw in np.asarray(near_indices, dtype=np.int64).reshape(-1, 2):
            p = int(p_raw)
            q = int(q_raw)
            image_key = (p, q)
            if image_key not in image_metadata:
                shift = np.asarray(lattice.lattice_vector(p, q), dtype=float)
                image_metadata[image_key] = (
                    shift,
                    complex(np.exp(1j * float(np.dot(kp, shift[:2])))),
                )
            for source_index, destination_index in zip(sources, destinations, strict=True):
                stack.append((int(closure_level), int(source_index), int(destination_index), p, q))

    while stack:
        level_index, source_index, destination_index, p, q = stack.pop()
        level = levels[level_index]
        shift, phase = image_metadata[(p, q)]
        delta = np.asarray(
            level.centers[destination_index] - level.centers[source_index] - shift,
            dtype=float,
        )
        half_size = half_sizes[level_index]
        if _boxes_well_separated(delta, half_size=half_size):
            coord_delta = np.asarray(
                level.coords[destination_index] - level.coords[source_index], dtype=np.int64
            )
            key = (
                p,
                q,
                int(coord_delta[0]),
                int(coord_delta[1]),
                int(coord_delta[2]),
            )
            grouped = far_grouped[level_index].get(key)
            if grouped is None:
                far_grouped[level_index][key] = (
                    [source_index],
                    [destination_index],
                    delta,
                    phase,
                )
            else:
                grouped[0].append(source_index)
                grouped[1].append(destination_index)
            continue
        if level_index == leaf_level:
            grouped_leaf = leaf_grouped.get((p, q))
            if grouped_leaf is None:
                leaf_grouped[(p, q)] = (
                    [source_index],
                    [destination_index],
                    np.asarray(shift, dtype=float),
                    phase,
                )
            else:
                grouped_leaf[0].append(source_index)
                grouped_leaf[1].append(destination_index)
            continue
        for source_child in level.children[source_index]:
            for destination_child in level.children[destination_index]:
                stack.append(
                    (
                        level_index + 1,
                        int(source_child),
                        int(destination_child),
                        p,
                        q,
                    )
                )

    far_batches_by_level: list[tuple[MLFMMPeriodicFarBatch, ...]] = []
    sampled_pair_count = 0
    for level_index, grouped_level in enumerate(far_grouped):
        level = levels[level_index]
        batches: list[MLFMMPeriodicFarBatch] = []
        for far_key in sorted(grouped_level):
            source_indices, destination_indices, delta, phase = grouped_level[far_key]
            diagonal = phase * _sampled_rokhlin_translator(
                np.asarray(delta, dtype=float),
                k=complex(k),
                truncation_order=int(level.translator_order),
                directions=level.directional.grid.directions,
                weights=level.directional.grid.weights,
                dtype=COMPLEX128_DTYPE,
            )
            source_array = np.asarray(source_indices, dtype=np.int32)
            destination_array = np.asarray(destination_indices, dtype=np.int32)
            sampled_pair_count += int(source_array.size)
            batches.append(
                MLFMMPeriodicFarBatch(
                    source_indices=source_array,
                    destination_indices=destination_array,
                    diagonal=np.asarray(diagonal, dtype=COMPLEX128_DTYPE),
                )
            )
        far_batches_by_level.append(tuple(batches))

    leaf_batches: list[MLFMMPeriodicLeafBatch] = []
    exact_particle_pair_count = 0
    exact_leaf_r_max_bound = 0.0
    leaves = operators.partition.leaves
    positions_arr = np.asarray(positions, dtype=float).reshape(-1, 3)
    leaf_bounds: dict[int, tuple[Array, Array]] = {}

    def bounds_for_leaf(leaf_index: int) -> tuple[Array, Array]:
        cached = leaf_bounds.get(int(leaf_index))
        if cached is not None:
            return cached
        particle_positions = positions_arr[leaves[int(leaf_index)].particle_indices]
        bounds = (
            np.min(particle_positions, axis=0),
            np.max(particle_positions, axis=0),
        )
        leaf_bounds[int(leaf_index)] = bounds
        return bounds

    for leaf_key in sorted(leaf_grouped):
        source_indices, destination_indices, shift, phase = leaf_grouped[leaf_key]
        source_array = np.asarray(source_indices, dtype=np.int32)
        destination_array = np.asarray(destination_indices, dtype=np.int32)
        for source_leaf, destination_leaf in zip(source_array, destination_array, strict=True):
            source_leaf_i = int(source_leaf)
            destination_leaf_i = int(destination_leaf)
            exact_particle_pair_count += int(
                leaves[source_leaf_i].particle_indices.size
                * leaves[destination_leaf_i].particle_indices.size
            )
            source_min, source_max = bounds_for_leaf(source_leaf_i)
            destination_min, destination_max = bounds_for_leaf(destination_leaf_i)
            exact_leaf_r_max_bound = max(
                exact_leaf_r_max_bound,
                _translated_leaf_pair_max_distance(
                    source_min=source_min,
                    source_max=source_max,
                    destination_min=destination_min,
                    destination_max=destination_max,
                    lattice_shift=shift,
                ),
            )
        leaf_batches.append(
            MLFMMPeriodicLeafBatch(
                source_leaf_indices=source_array,
                destination_leaf_indices=destination_array,
                lattice_shift=np.asarray(shift, dtype=float),
                bloch_phase=complex(phase),
            )
        )
    return (
        tuple(far_batches_by_level),
        tuple(leaf_batches),
        int(sampled_pair_count),
        int(exact_particle_pair_count),
        float(exact_leaf_r_max_bound),
    )


def _resolve_periodic_closure_level(
    operators: MLFMMMultilevelOperators,
    *,
    k: float,
    lattice: RectangularLattice2D,
) -> int:
    """Choose the sampled level with the safest near-image removal margin."""

    start = int(operators.hf_start_level)
    end = int(operators.hf_end_level)
    best_level = start
    best_margin = -math.inf
    for level_index in range(start, end + 1):
        level = operators.levels[level_index]
        half_size = float(operators.partition.root_half_size) / float(1 << level_index)
        groups = _coarse_pair_groups(operators, closure_level=level_index)
        margin = math.inf
        for _offset, (_sources, _destinations, delta) in groups.items():
            singular = _lattice_equivalent_image(delta, lattice=lattice)
            near_indices = _near_lattice_indices_for_offset(
                base_displacement=delta,
                half_size=half_size,
                lattice=lattice,
            )
            for p_raw, q_raw in np.asarray(near_indices, dtype=np.int64).reshape(-1, 2):
                image = (int(p_raw), int(q_raw))
                if image == singular:
                    continue
                separation = float(np.linalg.norm(delta - lattice.lattice_vector(*image)))
                # High-order spherical Hankel coefficients become poorly scaled when
                # their degree substantially exceeds k*r. The closure subtracts these
                # finite near images before descending them exactly, so choose the
                # level that maximizes the worst such margin. This is a representation-
                # conditioning criterion, not a hardware/performance threshold.
                margin = min(
                    margin,
                    abs(float(k)) * separation / float(int(level.translator_order) + 1),
                )
        if margin >= best_margin:
            best_level = int(level_index)
            best_margin = float(margin)
    return best_level


def _build_periodization_plan_at_level(
    *,
    k: float,
    positions: Array,
    periodic: PeriodicSpec,
    k_parallel: Array,
    operators: MLFMMMultilevelOperators,
    closure_level: int,
    show_progress: bool = False,
    matmul_backend: Literal["numpy", "cupy"] = "numpy",
) -> MLFMMPeriodizationPlan:
    """Build one periodic closure candidate at an explicit sampled level."""

    level = operators.levels[closure_level]
    coarse_groups = _coarse_pair_groups(operators, closure_level=closure_level)
    offsets = sorted(coarse_groups)
    displacements = np.asarray([coarse_groups[key][2] for key in offsets], dtype=float)
    if show_progress:
        half_size = float(operators.partition.root_half_size) / float(1 << closure_level)
        tqdm.write(
            "[MLFMM] periodic closure prepare "
            f"level={closure_level} order={level.translator_order} offsets={len(offsets)} "
            f"box_side={2.0 * half_size:.6g} lattice_min="
            f"{min(float(periodic.lattice.ax), float(periodic.lattice.ay)):.6g}"
        )
    structural_started = perf_counter()
    periodic_nonzero, singular_images, eta = _periodic_nonzero_structural_sums(
        max_degree=int(level.translator_order),
        k=float(k),
        periodic=periodic,
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        offsets=offsets,
        displacements=displacements,
        matmul_backend=matmul_backend,
    )
    if show_progress:
        tqdm.write(
            "[MLFMM] periodic structural closure "
            f"backend={matmul_backend} elapsed_s={perf_counter() - structural_started:.2f}"
        )

    half_size = float(operators.partition.root_half_size) / float(1 << closure_level)
    near_by_offset: dict[Offset3, Array] = {}
    near_union: set[LatticeIndex] = set()
    far_structural_tables: list[Array] = []
    closure_indices: list[tuple[Array, Array]] = []
    diagonal_started = perf_counter()
    for offset in offsets:
        sources, destinations, delta = coarse_groups[offset]
        near_indices = _near_lattice_indices_for_offset(
            base_displacement=delta,
            half_size=half_size,
            lattice=periodic.lattice,
        )
        near_by_offset[offset] = near_indices
        near_union.update(
            (int(p), int(q)) for p, q in np.asarray(near_indices, dtype=np.int64).reshape(-1, 2)
        )
        far_structural = np.asarray(periodic_nonzero[offset], dtype=np.complex128).copy()
        if near_indices.size:
            near_structural = _finite_image_structural_sums(
                max_degree=int(level.translator_order),
                k=float(k),
                base_displacement=delta,
                lattice=periodic.lattice,
                k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
                lattice_indices=near_indices,
                excluded_singular_image=singular_images[offset],
            )
            far_structural -= near_structural
        far_structural_tables.append(far_structural)
        closure_indices.append(
            (
                np.asarray(sources, dtype=np.int32),
                np.asarray(destinations, dtype=np.int32),
            )
        )
    stacked_far = np.stack(far_structural_tables, axis=0)
    if not np.all(np.isfinite(stacked_far)):
        raise FloatingPointError(
            "Periodic MLFMM far structural remainder contains non-finite coefficients "
            f"at closure level {closure_level}."
        )
    diagonals = sampled_transfers_from_structural_sums(
        stacked_far,
        directions=level.directional.grid.directions,
        weights=level.directional.grid.weights,
        dtype=COMPLEX128_DTYPE,
        matmul_backend=matmul_backend,
    )
    if not np.all(np.isfinite(diagonals)):
        raise FloatingPointError(
            "Periodic MLFMM sampled closure contains non-finite coefficients "
            f"at closure level {closure_level}."
        )
    closure_batches: list[MLFMMPeriodicFarBatch] = []
    closure_pair_count = 0
    for (source_array, destination_array), diagonal in zip(closure_indices, diagonals, strict=True):
        closure_pair_count += int(source_array.size)
        closure_batches.append(
            MLFMMPeriodicFarBatch(
                source_indices=source_array,
                destination_indices=destination_array,
                diagonal=np.asarray(diagonal, dtype=COMPLEX128_DTYPE),
            )
        )
    if show_progress:
        tqdm.write(
            f"[MLFMM] periodic sampled diagonals elapsed_s={perf_counter() - diagonal_started:.2f}"
        )

    descent_started = perf_counter()
    far_batches, leaf_batches, descended_far_pairs, exact_particle_pairs, exact_leaf_r_max_bound = (
        _descend_near_image_pairs(
            k=float(k),
            positions=np.asarray(positions, dtype=float),
            operators=operators,
            lattice=periodic.lattice,
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            closure_level=closure_level,
            coarse_groups=coarse_groups,
            near_by_offset=near_by_offset,
        )
    )
    if show_progress:
        tqdm.write(
            f"[MLFMM] periodic near-image descent elapsed_s={perf_counter() - descent_started:.2f}"
        )
    far_mutable = [list(batches) for batches in far_batches]
    far_mutable[closure_level] = [*closure_batches, *far_mutable[closure_level]]
    combined_far = tuple(tuple(batches) for batches in far_mutable)
    exact_leaf_pairs = sum(int(batch.source_leaf_indices.size) for batch in leaf_batches)
    return MLFMMPeriodizationPlan(
        closure_level=closure_level,
        far_batches_by_level=combined_far,
        leaf_batches=leaf_batches,
        periodic_offset_count=len(offsets),
        near_image_count=len(near_union),
        sampled_far_box_pair_count=int(closure_pair_count + descended_far_pairs),
        exact_leaf_box_pair_count=int(exact_leaf_pairs),
        exact_particle_pair_count=int(exact_particle_pairs),
        exact_leaf_r_max_bound=float(exact_leaf_r_max_bound),
        ewald_eta=float(eta),
    )


def build_periodization_plan(
    *,
    k: float,
    positions: Array,
    periodic: PeriodicSpec,
    k_parallel: Array,
    operators: MLFMMMultilevelOperators,
    show_progress: bool = False,
    matmul_backend: Literal["numpy", "cupy"] = "numpy",
) -> MLFMMPeriodizationPlan:
    """Build a geometry-conditioned periodic closure and image correction."""

    if periodic.options.method != "ewald":
        raise NotImplementedError(
            "Periodized MLFMM requires PeriodicOptions(method='ewald') as its closure policy."
        )
    closure_level = _resolve_periodic_closure_level(
        operators,
        k=float(k),
        lattice=periodic.lattice,
    )
    return _build_periodization_plan_at_level(
        k=float(k),
        positions=positions,
        periodic=periodic,
        k_parallel=k_parallel,
        operators=operators,
        closure_level=int(closure_level),
        show_progress=bool(show_progress),
        matmul_backend=matmul_backend,
    )


def prepare_periodized_mlfmm_coupling(
    *,
    lmax: int,
    k: float,
    positions: Array,
    particle_circumscribing_radii: Array,
    radial_lut: RadialLUT,
    periodic: PeriodicSpec,
    k_parallel: Array,
    options: MLFMMOptions | None = None,
    dtype: np.dtype = COMPLEX128_DTYPE,
    cache_translation_blocks: bool = False,
    show_progress: bool = False,
    box_order: int | None = None,
    leaf_map_backend: Literal["numpy", "cupy"] = "numpy",
    build_leaf_maps: bool = True,
    closure_matmul_backend: Literal["numpy", "cupy"] = "numpy",
) -> MLFMMCouplingOperator:
    """Prepare a periodized MLFMM coupling plan for NumPy or CuPy upload.

    The central cell and every finite image correction share one occupied
    hierarchy. Periodic structural sums are evaluated only while preparing
    coarse periodizing diagonals; repeated applications use exact leaf translations and ordinary
    sampled MLFMM passes. ``leaf_map_backend`` and ``build_leaf_maps`` control
    the host staging representation. ``closure_matmul_backend`` independently
    selects the array backend for the large preparation-only contractions.
    """

    prepare_started = perf_counter()
    if cache_translation_blocks:
        raise NotImplementedError("Periodic MLFMM does not cache exact lattice-image leaf blocks.")
    if periodic.options.method != "ewald":
        raise NotImplementedError(
            "Periodic MLFMM currently requires PeriodicOptions(method='ewald')."
        )
    out_dtype = np.dtype(dtype)
    pts = np.asarray(positions, dtype=float).reshape(-1, 3)
    radii = np.asarray(particle_circumscribing_radii, dtype=float).reshape(-1)
    resolved_options = MLFMMOptions() if options is None else options
    resolved: MLFMMResolvedPlan = resolve_mlfmm_plan(
        pts,
        particle_circumscribing_radii=radii,
        options=resolved_options,
    )
    if (
        int(resolved.selected_depth) >= 2
        and resolved_options.hf_start_level is not None
        and int(resolved_options.hf_start_level) != 2
    ):
        raise NotImplementedError(
            "Periodic MLFMM currently requires the canonical same-cell sampled "
            "ownership to start at level 2; leave mlfmm_options.hf_start_level "
            "unset or set it to 2. Periodic lattice closure is selected independently."
        )
    if show_progress:
        occupancies = np.asarray(
            [leaf.particle_indices.size for leaf in resolved.partition.leaves],
            dtype=np.int64,
        )
        occ_min = int(np.min(occupancies)) if occupancies.size else 0
        occ_med = int(np.median(occupancies)) if occupancies.size else 0
        occ_max = int(np.max(occupancies)) if occupancies.size else 0
        tqdm.write(
            "[MLFMM] plan stage=multilevel "
            f"depth={resolved.selected_depth} leaves={resolved.occupied_leaf_count} "
            f"leaf_occ_min/med/max={occ_min}/{occ_med}/{occ_max} periodic=true"
        )

    radial_lut_hf = radial_lut
    if np.dtype(radial_lut.dtype) != np.dtype(np.complex128):
        radial_lut_hf = RadialLUT(
            lmax=int(lmax),
            k=float(k),
            r_max=float(radial_lut.r_grid[-1]),
            dr=float(radial_lut.dr),
            dtype=np.complex128,
        )
    hierarchy_started = perf_counter()
    operators = build_multilevel_mlfmm_operators(
        lmax=int(lmax),
        k=float(k),
        positions=pts,
        partition=resolved.partition,
        radial_lut=radial_lut_hf,
        box_order=box_order,
        accuracy_level=int(resolved_options.accuracy_level),
        order_additive=int(resolved_options.order_additive),
        dtype=np.complex128,
        leaf_map_backend=leaf_map_backend,
        build_leaf_maps=bool(build_leaf_maps),
        show_progress=bool(show_progress),
        hf_start_level=resolved_options.hf_start_level,
        hf_wavelength_divisor=float(resolved_options.hf_wavelength_divisor),
    )
    if show_progress:
        tqdm.write(f"[MLFMM] periodic hierarchy elapsed_s={perf_counter() - hierarchy_started:.2f}")
    closure_started = perf_counter()
    periodization = build_periodization_plan(
        k=float(k),
        positions=pts,
        periodic=periodic,
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        operators=operators,
        show_progress=bool(show_progress),
        matmul_backend=closure_matmul_backend,
    )
    if show_progress:
        summary = periodization.summary()
        tqdm.write(
            "[MLFMM] periodic closure "
            f"level={summary['closure_level']} offsets={summary['periodic_offset_count']} "
            f"near_images={summary['near_image_count']} "
            f"elapsed_s={perf_counter() - closure_started:.2f}"
        )

    # Exact residual image leaves share the canonical outgoing-wave radial
    # table. The periodization planner already knows the actual directed leaf
    # pairs, so size the table from their AABB distance bound instead of the
    # much looser diameter of the complete translated supercell.
    required_r_max = float(periodization.exact_leaf_r_max_bound)
    covered_r_max = float(radial_lut_hf.r_grid[-1])
    if covered_r_max < required_r_max:
        radial_lut_hf = RadialLUT(
            lmax=int(lmax),
            k=float(k),
            r_max=required_r_max,
            dr=float(radial_lut_hf.dr),
            dtype=np.complex128,
        )

    coupling = MLFMMCouplingOperator(
        lmax=int(lmax),
        k=float(k),
        positions=pts,
        radial_lut=radial_lut_hf,
        resolved_plan=replace(resolved, stage="multilevel"),
        dtype=out_dtype,
        near_dtype=out_dtype,
        far_dtype=np.dtype(np.complex128),
        cache_translation_blocks=False,
        single_level=None,
        multilevel=operators,
        periodization=periodization,
    )
    if show_progress:
        tqdm.write(
            f"[MLFMM] periodic prepare total elapsed_s={perf_counter() - prepare_started:.2f}"
        )
    return coupling


__all__ = [
    "build_periodization_plan",
    "prepare_periodized_mlfmm_coupling",
    "sampled_transfer_from_structural_sums",
]
