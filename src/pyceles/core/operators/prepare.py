"""Operator-preparation orchestration for the many-body `A = I - T W` system."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from pyceles._cupy_memory import cupy_allocator_snapshot
from pyceles._dtypes import resolve_compute_accum_dtypes
from pyceles._optional import import_cupy
from pyceles.core.geometry_bounds import conservative_set_diameter
from pyceles.core.indexing import n_modes
from pyceles.core.particles import Particle, ParticleCollection, Sphere, particle_t_signature
from pyceles.core.periodic import PeriodicSpec
from pyceles.core.tmatrix import particle_T_diagonal, particle_T_matrix_blocks, sphere_T_diagonal
from pyceles.core.translation import RadialLUT, translation_ab5_table

from .base import CouplingOperator, PreparedOperator
from .coupling_pairwise import PairwiseCouplingOperator
from .coupling_pairwise_cupy import CuPyPairwiseCouplingOperator
from .coupling_periodic import PeriodicCouplingOperator
from .coupling_periodic_cupy import CuPyPeriodicCouplingOperator
from .groups import (
    AxisymmetricTGroup,
    DenseTGroup,
    DiagonalTGroup,
    ParticleTGroupFactories,
    ParticleTGroupPlan,
    ParticleTPreparationContext,
    PreparedParticleTGroup,
    plan_particle_t_groups,
)
from .mlfmm import MLFMMCouplingOperator, MLFMMOptions, prepare_mlfmm_coupling
from .mlfmm_cupy import CuPyMLFMMHostCachePolicy, prepare_mlfmm_cupy_coupling
from .single_body import CompositeParticleTOperator, ParticleTOperator, build_T_mode_diagonal
from .single_body_cupy import wrap_particle_t_groups_cupy

Array = np.ndarray


_CUPY_MLFMM_DEFAULT_OPTIONS = MLFMMOptions(
    max_leaf_particles=32,
    max_depth=12,
    leaf_size_radius_factor=4.0,
    accuracy_level=3,
    order_additive=2,
)
"""CuPy-oriented MLFMM defaults.

These defaults intentionally target a higher leaf occupancy than the NumPy
reference baseline so the resolved tree lands in a shallower (GPU-friendlier)
regime on large workloads, while still honoring the leaf-size floor and max
depth safety cap.
"""


def _wrap_mlfmm_cupy_coupling(
    coupling: MLFMMCouplingOperator,
    *,
    options: MLFMMOptions,
) -> CouplingOperator:
    """Upload one CPU-built finite MLFMM plan consistently."""

    return cast(
        CouplingOperator,
        prepare_mlfmm_cupy_coupling(
            coupling,
            host_cache_policy=CuPyMLFMMHostCachePolicy(
                collect_stream_stats=bool(options.collect_stream_stats)
            ),
        ),
    )


def _prepare_diagonal_group(
    *,
    plan: ParticleTGroupPlan,
    context: ParticleTPreparationContext,
) -> DiagonalTGroup:
    """Prepare one diagonal operator per unique archetype in the group."""
    archetypes = context.archetypes_for(plan)
    T_M, T_N = precompute_T_diagonal(
        lmax=context.lmax,
        k=context.k,
        particles=archetypes,
        n_medium=context.n_medium,
        dtype=context.dtype,
    )
    return DiagonalTGroup(
        particle_indices=np.asarray(plan.particle_indices, dtype=np.int64),
        operator_indices=np.asarray(plan.operator_indices, dtype=np.int64),
        T_M=T_M,
        T_N=T_N,
        T_diag=build_T_mode_diagonal(context.lmax, T_M, T_N),
        dtype=context.dtype,
    )


def _default_group_factory(
    *,
    plan: ParticleTGroupPlan,
    context: ParticleTPreparationContext,
) -> PreparedParticleTGroup:
    if plan.representation == "diagonal":
        return _prepare_diagonal_group(plan=plan, context=context)

    if plan.representation == "dense":
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        archetypes = context.archetypes_for(plan)
        try:
            blocks = particle_T_matrix_blocks(
                lmax=context.lmax,
                k_medium=context.k,
                particles=archetypes,
                n_medium=context.n_medium,
                dtype=context.dtype,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(str(exc)) from exc
        except TypeError as exc:
            raise NotImplementedError(
                "dense particle-T preparation requires canonical spherical-basis "
                "T blocks for every particle in the selected group."
            ) from exc
        return DenseTGroup(
            particle_indices=ids,
            operator_indices=np.asarray(plan.operator_indices, dtype=np.int64),
            T_blocks=blocks,
            dtype=context.dtype,
        )

    if plan.representation == "axisymmetric":
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        archetypes = context.archetypes_for(plan)
        try:
            blocks = particle_T_matrix_blocks(
                lmax=context.lmax,
                k_medium=context.k,
                particles=archetypes,
                n_medium=context.n_medium,
                dtype=context.dtype,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(str(exc)) from exc
        except TypeError as exc:
            raise NotImplementedError(
                "axisymmetric particle-T preparation requires canonical spherical-basis "
                "T blocks for every particle in the selected group."
            ) from exc
        return AxisymmetricTGroup(
            particle_indices=ids,
            operator_indices=np.asarray(plan.operator_indices, dtype=np.int64),
            T_blocks=blocks,
            body_metadata={
                "storage": "shared_spherical_basis_dense_blocks",
                "n_unique_archetypes": len(archetypes),
            },
            dtype=context.dtype,
        )

    particle_labels = ", ".join(
        f"{int(i)}:{type(context.particles[int(i)]).__name__}"
        for i in np.asarray(plan.particle_indices)
    )
    raise NotImplementedError(
        "Single-body operator planning selected the "
        f"'{plan.representation}' representation for particle(s) {particle_labels}, "
        "but no preparation factory was provided for that representation."
    )


def _prepare_particle_t_operator(
    *,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex,
    dtype: np.dtype,
    group_factories: ParticleTGroupFactories | None = None,
) -> CompositeParticleTOperator:
    """Prepare the particle-local operator using the selected representation groups."""
    part = ParticleCollection.from_particles(particles)
    context = ParticleTPreparationContext(
        lmax=int(lmax),
        k=float(k),
        particles=part,
        n_medium=complex(n_medium),
        dtype=dtype,
    )
    plans = plan_particle_t_groups(part)
    factories = ParticleTGroupFactories() if group_factories is None else group_factories
    groups: list[PreparedParticleTGroup] = []
    for plan in plans:
        factory = factories.for_representation(plan.representation)
        groups.append(
            _default_group_factory(plan=plan, context=context)
            if factory is None
            else factory(plan, context)
        )

    return CompositeParticleTOperator(
        lmax=int(lmax),
        n_particles=len(part),
        groups=tuple(groups),
        dtype=dtype,
    )


def _infer_rmax(positions: Array) -> float:
    """Conservative upper bound on center-to-center separation for LUT sizing."""
    return conservative_set_diameter(np.asarray(positions, dtype=float))


def prepare_matvec(
    *,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex = 1.0 + 0j,
    radial_lut_dr: float,
    cache_translation_blocks: bool = False,
    operator_dtype: npt.DTypeLike = np.complex128,
    accum_dtype: npt.DTypeLike | None = None,
    particle_t_group_factories: ParticleTGroupFactories | None = None,
    coupling_backend: Literal["pairwise", "mlfmm"] = "pairwise",
    mlfmm_options: MLFMMOptions | None = None,
    periodic: PeriodicSpec | None = None,
    k_parallel: npt.ArrayLike | None = None,
    backend: Literal["numpy", "cupy"] = "numpy",
    show_progress: bool = False,
) -> PreparedOperator:
    """Prepare reusable `A = I - T W` data from a particle collection.

    ``operator_dtype`` controls stored operator arrays. ``accum_dtype`` is an
    optional wider precision budget for sensitive reductions; it is consumed by
    periodic coupling operators and defaults to the operator dtype here.
    """
    part = ParticleCollection.from_particles(particles)
    positions = part.positions
    circumscribing_radii = part.circumscribing_radii
    coupling_name = str(coupling_backend).lower()
    if coupling_name not in {"pairwise", "mlfmm"}:
        raise ValueError(
            f"Unknown coupling backend '{coupling_backend}'. Use 'pairwise' or 'mlfmm'."
        )
    backend_name = str(backend).lower()
    if backend_name not in {"numpy", "cupy"}:
        raise ValueError(f"Unknown operator backend '{backend}'. Use 'numpy' or 'cupy'.")

    op_dtype = np.dtype(operator_dtype)
    _, op_accum_dtype = resolve_compute_accum_dtypes(
        compute_dtype=op_dtype,
        accum_dtype=op_dtype if accum_dtype is None else accum_dtype,
    )
    k_f = float(k)
    periodic_spec = periodic
    k_parallel_arr: np.ndarray | None = None
    if periodic_spec is not None:
        if coupling_name == "mlfmm":
            raise NotImplementedError(
                "Periodic MLFMM is not implemented. Use coupling_backend='pairwise' "
                "with the periodic Rayleigh/Ewald operator."
            )
        if k_parallel is None:
            raise ValueError("`k_parallel` is required when preparing a periodic operator.")
        k_parallel_arr = np.asarray(k_parallel, dtype=float).reshape(2)
        if backend_name == "cupy" and periodic_spec.options.method not in {"ewald", "rayleigh"}:
            raise NotImplementedError(
                "CuPy periodic workflows currently support Ewald or Rayleigh coupling."
            )

    k_abs = float(abs(k_f))
    if k_abs <= 0.0:
        raise ValueError(f"`k` must be non-zero for operator preparation. Got {k_f!r}.")
    dr_user = float(radial_lut_dr)
    if dr_user < 0.0:
        raise ValueError(f"radial_lut_dr must be >= 0, got {dr_user}.")

    def make_radial_lut(lut_dtype: np.dtype | None = None) -> RadialLUT:
        dr = (1.0e-2 / k_abs) if dr_user == 0.0 else dr_user
        return RadialLUT(
            lmax=int(lmax),
            k=k_f,
            r_max=_infer_rmax(positions),
            dr=dr,
            dtype=op_dtype if lut_dtype is None else np.dtype(lut_dtype),
        )

    particle_t: ParticleTOperator
    coupling: CouplingOperator
    if backend_name == "numpy":
        particle_t = _prepare_particle_t_operator(
            lmax=int(lmax),
            k=k_f,
            particles=part,
            n_medium=n_medium,
            dtype=op_dtype,
            group_factories=particle_t_group_factories,
        )
        if periodic_spec is not None:
            if k_parallel_arr is None:
                raise RuntimeError("Internal error: periodic k_parallel was not normalized.")
            ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)
            coupling = PeriodicCouplingOperator(
                lmax=int(lmax),
                k=k_f,
                positions=positions,
                ab5=ab5,
                periodic=periodic_spec,
                k_parallel=k_parallel_arr,
                dtype=op_dtype,
                accum_dtype=op_accum_dtype,
                cache_blocks=bool(cache_translation_blocks),
                circumscribing_radii=circumscribing_radii,
            )
        elif coupling_name == "pairwise":
            lut = make_radial_lut()
            ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)
            coupling = PairwiseCouplingOperator(
                lmax=int(lmax),
                k=k_f,
                positions=positions,
                ab5=ab5,
                radial_lut=lut,
                dtype=op_dtype,
                cache_translation_blocks=bool(cache_translation_blocks),
            )
        elif coupling_name == "mlfmm":
            lut = make_radial_lut()
            coupling = prepare_mlfmm_coupling(
                lmax=int(lmax),
                k=k_f,
                positions=positions,
                particle_circumscribing_radii=circumscribing_radii,
                radial_lut=lut,
                ab5=None,
                options=mlfmm_options,
                dtype=op_dtype,
                cache_translation_blocks=bool(cache_translation_blocks),
                show_progress=bool(show_progress),
                leaf_map_backend="numpy",
            )
    elif backend_name == "cupy":
        cupy, _ = import_cupy()
        # Establish one process-wide CuPy pool ceiling below physical VRAM
        # before any backend-specific persistent state or Krylov workspace is
        # allocated.  Method-specific planners may impose stricter budgets,
        # but no CuPy simulation path should rely on WDDM/shared-memory spill.
        cupy_allocator_snapshot(cupy, apply_pool_limit=True)
        if cache_translation_blocks and periodic_spec is None:
            raise NotImplementedError(
                "`cache_translation_blocks=True` is not supported with finite `operator_backend='cupy'` "
                "(direct raw-kernel coupling path)."
            )
        # The CuPy backend accepts mixed diagonal/dense groups, including
        # axisymmetric particles such as spheroids, by uploading explicit
        # spherical-basis T blocks to the GPU. It does not currently wrap the
        # narrower callback-only axisymmetric group hooks onto device, because
        # the current particle families can already use the explicit-block path
        # and future database-driven particle types are expected to do the same.
        cpu_particle_t = _prepare_particle_t_operator(
            lmax=int(lmax),
            k=k_f,
            particles=part,
            n_medium=n_medium,
            dtype=op_dtype,
            group_factories=particle_t_group_factories,
        )
        particle_t = cast(
            ParticleTOperator,
            wrap_particle_t_groups_cupy(
                cast(
                    tuple[DiagonalTGroup | DenseTGroup | AxisymmetricTGroup, ...],
                    tuple(cpu_particle_t.groups),
                ),
                lmax=int(lmax),
                n_particles=len(part),
                dtype=op_dtype,
            ),
        )
        cpu_mlfmm: CouplingOperator
        if periodic_spec is not None:
            if k_parallel_arr is None:
                raise RuntimeError("Internal error: periodic k_parallel was not normalized.")
            ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)
            coupling = cast(
                CouplingOperator,
                CuPyPeriodicCouplingOperator(
                    lmax=int(lmax),
                    k=k_f,
                    positions=positions,
                    ab5=ab5,
                    periodic=periodic_spec,
                    k_parallel=k_parallel_arr,
                    dtype=op_dtype,
                    accum_dtype=op_accum_dtype,
                    cache_blocks=bool(cache_translation_blocks),
                    circumscribing_radii=circumscribing_radii,
                ),
            )
        elif coupling_name == "pairwise":
            lut = make_radial_lut()
            ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)
            coupling = cast(
                CouplingOperator,
                CuPyPairwiseCouplingOperator(
                    lmax=int(lmax),
                    k=k_f,
                    positions=positions,
                    ab5=ab5,
                    radial_lut=lut,
                    dtype=op_dtype,
                ),
            )
        elif coupling_name == "mlfmm":
            resolved_mlfmm_options = (
                _CUPY_MLFMM_DEFAULT_OPTIONS if mlfmm_options is None else mlfmm_options
            )
            lut = make_radial_lut()
            cpu_mlfmm = prepare_mlfmm_coupling(
                lmax=int(lmax),
                k=k_f,
                positions=positions,
                particle_circumscribing_radii=circumscribing_radii,
                radial_lut=lut,
                ab5=None,
                options=resolved_mlfmm_options,
                dtype=op_dtype,
                cache_translation_blocks=False,
                show_progress=bool(show_progress),
                leaf_map_backend="cupy",
                # CuPy repeated-apply defaults to on-the-fly leaf translation blocks.
                # Skip dense CPU leaf-map materialization during the staging build to
                # avoid pathological host-RAM blow-ups at shallow/high-order trees.
                build_leaf_maps=False,
            )
            if isinstance(cpu_mlfmm, PairwiseCouplingOperator):
                ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)
                coupling = cast(
                    CouplingOperator,
                    CuPyPairwiseCouplingOperator(
                        lmax=int(lmax),
                        k=k_f,
                        positions=positions,
                        ab5=ab5,
                        radial_lut=lut,
                        dtype=op_dtype,
                    ),
                )
            else:
                coupling = _wrap_mlfmm_cupy_coupling(
                    cast(MLFMMCouplingOperator, cpu_mlfmm),
                    options=resolved_mlfmm_options,
                )

    return PreparedOperator(
        lmax=int(lmax),
        k=k_f,
        positions=positions,
        particle_t=particle_t,
        coupling=coupling,
        dtype=op_dtype,
    )


def precompute_T_diagonal(
    *,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex = 1.0 + 0j,
    dtype: npt.DTypeLike = np.complex128,
) -> tuple[Array, Array]:
    """Precompute per-particle diagonal T entries for the current geometry."""
    lmax = int(lmax)
    out_dtype = np.dtype(dtype)
    ns = len(particles)
    T_M = np.zeros((ns, lmax + 1), dtype=out_dtype)
    T_N = np.zeros((ns, lmax + 1), dtype=out_dtype)

    max_tmemo_entries = 100_000
    diagonal_memo: dict[tuple[object, ...], tuple[Array, Array]] = {}
    for i, p in enumerate(particles):
        key = (
            particle_t_signature(p),
            complex(k),
            complex(n_medium),
            int(lmax),
        )
        cached = diagonal_memo.get(key)
        if cached is None:
            if isinstance(p, Sphere):
                Td = sphere_T_diagonal(lmax, k, p.radius, p.refractive_index, n_medium)
            else:
                Td = particle_T_diagonal(lmax=lmax, k_medium=k, particle=p, n_medium=n_medium)
            cached = (
                np.asarray(Td[1], dtype=out_dtype).copy(),
                np.asarray(Td[2], dtype=out_dtype).copy(),
            )
            if len(diagonal_memo) < max_tmemo_entries:
                diagonal_memo[key] = cached

        T_M[i, :] = cached[0]
        T_N[i, :] = cached[1]

    return T_M, T_N


def rhs_Tb_numpy(
    lmax: int,
    b: Array,
    *,
    T_M: Array,
    T_N: Array,
    T_diag: Array | None = None,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Compute r = T b (NumPy) where b is stacked incident coefficients per sphere."""
    out_dtype = np.dtype(dtype)
    lmax = int(lmax)
    nm = n_modes(lmax)

    arr = np.asarray(b, dtype=out_dtype)
    ns = arr.size // nm
    arr = arr.reshape(ns, nm)

    diag = (
        build_T_mode_diagonal(lmax, T_M, T_N)
        if T_diag is None
        else np.asarray(T_diag, dtype=out_dtype)
    )
    r = diag * arr
    return cast(np.ndarray, r.reshape(ns * nm))


__all__ = ["build_T_mode_diagonal", "precompute_T_diagonal", "prepare_matvec", "rhs_Tb_numpy"]
