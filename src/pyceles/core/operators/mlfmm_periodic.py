"""Experimental NumPy periodization of the pyceles MLFMM hierarchy.

The construction closes the infinite two-dimensional lattice at the first
sampled MLFMM interaction level.  For every relative coarse-box offset, an
Ewald structural sum represents all nonzero lattice images that are already
well separated at that level.  The finite set of image interactions that are
not yet well separated is descended through the ordinary occupied hierarchy;
well-separated descendants use the existing sampled Rokhlin translator and
the residual leaf pairs remain exact.

The repeated apply is therefore mesh-free and contains neither a particle-pair
Ewald cache nor a Rayleigh particle/mode table.  Ewald is used only during
preparation to build a small family of coarse-box periodizing diagonals.
"""

from __future__ import annotations

import math
from dataclasses import replace
from time import perf_counter

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
from pyceles.core.periodic.scalar import structural_sum_m_normalization
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

    sums = np.asarray(structural_sums, dtype=np.complex128)
    if sums.ndim != 2 or sums.shape[1] != 2 * sums.shape[0] - 1:
        raise ValueError(f"structural_sums must have shape (L+1, 2*L+1). Got {sums.shape}.")
    degree_max = int(sums.shape[0] - 1)
    offset = degree_max
    dirs = np.asarray(directions, dtype=float).reshape(-1, 3)
    quadrature_weights = np.asarray(weights, dtype=float).reshape(-1)
    if quadrature_weights.size != dirs.shape[0]:
        raise ValueError("Directional weights must match the direction count.")

    ct = np.asarray(dirs[:, 2], dtype=float)
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    plm = legendre_normalized_trigon(ct, st, degree_max, xp=np)
    transfer = np.zeros((dirs.shape[0],), dtype=np.complex128)
    for degree in range(degree_max + 1):
        degree_phase = 4.0 * np.pi * (1j**degree)
        for order in range(-degree, degree + 1):
            ylm = (
                plm[degree, abs(order)]
                * np.exp(1j * float(order) * phi)
                / structural_sum_m_normalization(order)
            )
            reflected = (
                ((-1.0) ** order)
                * sums[degree, -order + offset]
                / structural_sum_m_normalization(-order)
            )
            transfer += degree_phase * reflected * ylm
    return np.asarray(transfer * quadrature_weights, dtype=np.dtype(dtype))


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
    positions: Array,
    periodic: PeriodicSpec,
    k_parallel: Array,
    offsets: list[Offset3],
    displacements: Array,
) -> tuple[dict[Offset3, Array], dict[Offset3, LatticeIndex | None], float]:
    """Evaluate all nonzero lattice images for coarse-box offsets in one batch."""

    degree_max = int(max_degree)
    structural_lmax = int((degree_max + 1) // 2)
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    eta = resolve_ewald_eta(
        periodic=periodic,
        k=float(k),
        k_parallel=kp,
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        # The periodizer requests structural degrees set by the MLFMM box
        # order, which can be much higher than the particle lmax.  Eta
        # preflight must therefore probe the actual closure degree.
        lmax=structural_lmax,
    )
    workspace = EwaldShellWorkspace(
        lattice=periodic.lattice,
        k=float(k),
        k_parallel=kp,
        eta=float(eta),
    )
    deltas = np.asarray(displacements, dtype=float).reshape(-1, 3)
    singular_images: dict[Offset3, LatticeIndex | None] = {}
    evaluation_deltas = np.array(deltas, dtype=float, copy=True)
    for index, key in enumerate(offsets):
        singular = _lattice_equivalent_image(deltas[index], lattice=periodic.lattice)
        singular_images[key] = singular
        if singular is not None:
            # Snap an ULP-level lattice coincidence to the exact image so the
            # Ewald real-space branch removes the singular term deterministically.
            evaluation_deltas[index] = periodic.lattice.lattice_vector(*singular)
    raw_batch = ewald_structural_sums_2d_batch(
        lmax_struct=structural_lmax,
        k=float(k),
        destinations=evaluation_deltas,
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
]:
    """Descend the finite non-well-separated image correction through the tree."""

    levels = operators.levels
    leaf_level = int(operators.leaf_level)
    far_grouped: list[
        dict[tuple[int, int, int, int, int], tuple[list[int], list[int], Array, complex]]
    ] = [dict() for _ in levels]
    leaf_grouped: dict[LatticeIndex, tuple[list[int], list[int], Array, complex]] = {}
    kp = np.asarray(k_parallel, dtype=float).reshape(2)

    stack: list[tuple[int, int, int, int, int]] = []
    for offset, (sources, destinations, _delta) in coarse_groups.items():
        near_indices = near_by_offset[offset]
        for p_raw, q_raw in np.asarray(near_indices, dtype=np.int64).reshape(-1, 2):
            p = int(p_raw)
            q = int(q_raw)
            for source_index, destination_index in zip(sources, destinations, strict=True):
                stack.append((int(closure_level), int(source_index), int(destination_index), p, q))

    while stack:
        level_index, source_index, destination_index, p, q = stack.pop()
        level = levels[level_index]
        shift = lattice.lattice_vector(p, q)
        delta = np.asarray(
            level.centers[destination_index] - level.centers[source_index] - shift,
            dtype=float,
        )
        half_size = float(operators.partition.root_half_size) / float(1 << level_index)
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
            phase = complex(np.exp(1j * float(np.dot(kp, shift[:2]))))
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
            phase = complex(np.exp(1j * float(np.dot(kp, shift[:2]))))
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
    leaves = operators.partition.leaves
    for leaf_key in sorted(leaf_grouped):
        source_indices, destination_indices, shift, phase = leaf_grouped[leaf_key]
        source_array = np.asarray(source_indices, dtype=np.int32)
        destination_array = np.asarray(destination_indices, dtype=np.int32)
        for source_leaf, destination_leaf in zip(source_array, destination_array, strict=True):
            exact_particle_pair_count += int(
                leaves[int(source_leaf)].particle_indices.size
                * leaves[int(destination_leaf)].particle_indices.size
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
    )


def build_periodization_plan(
    *,
    k: float,
    positions: Array,
    periodic: PeriodicSpec,
    k_parallel: Array,
    operators: MLFMMMultilevelOperators,
) -> MLFMMPeriodizationPlan:
    """Build the complete Ewald closure and finite-image hierarchy correction."""

    if periodic.options.method != "ewald":
        raise NotImplementedError(
            "The reference periodized MLFMM uses Ewald only during coarse-level "
            "closure; set PeriodicOptions(method='ewald')."
        )
    closure_level = int(operators.hf_start_level)
    level = operators.levels[closure_level]
    coarse_groups = _coarse_pair_groups(operators, closure_level=closure_level)
    offsets = sorted(coarse_groups)
    displacements = np.asarray([coarse_groups[key][2] for key in offsets], dtype=float)
    periodic_nonzero, singular_images, eta = _periodic_nonzero_structural_sums(
        max_degree=int(level.translator_order),
        k=float(k),
        positions=np.asarray(positions, dtype=float),
        periodic=periodic,
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        offsets=offsets,
        displacements=displacements,
    )

    half_size = float(operators.partition.root_half_size) / float(1 << closure_level)
    near_by_offset: dict[Offset3, Array] = {}
    near_union: set[LatticeIndex] = set()
    closure_batches: list[MLFMMPeriodicFarBatch] = []
    closure_pair_count = 0
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
            far_structural -= _finite_image_structural_sums(
                max_degree=int(level.translator_order),
                k=float(k),
                base_displacement=delta,
                lattice=periodic.lattice,
                k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
                lattice_indices=near_indices,
                excluded_singular_image=singular_images[offset],
            )
        diagonal = sampled_transfer_from_structural_sums(
            far_structural,
            directions=level.directional.grid.directions,
            weights=level.directional.grid.weights,
            dtype=COMPLEX128_DTYPE,
        )
        source_array = np.asarray(sources, dtype=np.int32)
        destination_array = np.asarray(destinations, dtype=np.int32)
        closure_pair_count += int(source_array.size)
        closure_batches.append(
            MLFMMPeriodicFarBatch(
                source_indices=source_array,
                destination_indices=destination_array,
                diagonal=np.asarray(diagonal, dtype=COMPLEX128_DTYPE),
            )
        )

    far_batches, leaf_batches, descended_far_pairs, exact_particle_pairs = (
        _descend_near_image_pairs(
            k=float(k),
            operators=operators,
            lattice=periodic.lattice,
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            closure_level=closure_level,
            coarse_groups=coarse_groups,
            near_by_offset=near_by_offset,
        )
    )
    far_mutable = [list(batches) for batches in far_batches]
    far_mutable[closure_level] = [*closure_batches, *far_mutable[closure_level]]
    combined_far = tuple(tuple(batches) for batches in far_mutable)
    exact_leaf_pairs = sum(int(batch.source_leaf_indices.size) for batch in leaf_batches)
    near_indices_array = np.asarray(
        sorted(near_union, key=lambda item: (max(abs(item[0]), abs(item[1])), item[0], item[1])),
        dtype=np.int32,
    ).reshape(-1, 2)
    return MLFMMPeriodizationPlan(
        closure_level=closure_level,
        far_batches_by_level=combined_far,
        leaf_batches=leaf_batches,
        near_lattice_indices=near_indices_array,
        periodic_offset_count=len(offsets),
        near_image_count=len(near_union),
        sampled_far_box_pair_count=int(closure_pair_count + descended_far_pairs),
        exact_leaf_box_pair_count=int(exact_leaf_pairs),
        exact_particle_pair_count=int(exact_particle_pairs),
        ewald_eta=float(eta),
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
) -> MLFMMCouplingOperator:
    """Prepare the experimental NumPy periodized MLFMM coupling operator.

    The central cell and every finite image correction share one occupied
    hierarchy.  Ewald is evaluated only while preparing coarse periodizing
    diagonals; repeated applications use exact leaf translations and ordinary
    MLFMM sampled passes.
    """

    prepare_started = perf_counter()
    if cache_translation_blocks:
        raise NotImplementedError(
            "Periodic MLFMM does not cache exact lattice-image leaf blocks in this "
            "reference implementation."
        )
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
            "Periodic MLFMM currently closes the lattice at the canonical first "
            "sampled level 2; leave mlfmm_options.hf_start_level unset or set it to 2."
        )
    if show_progress:
        tqdm.write(
            "[MLFMM] periodic plan "
            f"depth={resolved.selected_depth} leaves={resolved.occupied_leaf_count}"
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
        leaf_map_backend="numpy",
        build_leaf_maps=True,
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
    )
    if show_progress:
        summary = periodization.summary()
        tqdm.write(
            "[MLFMM] periodic closure "
            f"level={summary['closure_level']} offsets={summary['periodic_offset_count']} "
            f"near_images={summary['near_image_count']} "
            f"elapsed_s={perf_counter() - closure_started:.2f}"
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
