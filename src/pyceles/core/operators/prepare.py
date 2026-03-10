from __future__ import annotations

"""Operator-preparation orchestration for the many-body `A = I - T W` system."""

from typing import Sequence

import numpy as np
import numpy.typing as npt

from pyceles.core.geometry_bounds import conservative_set_diameter
from pyceles.core.indexing import n_modes
from pyceles.core.particles import Particle, Sphere, particle_t_signature
from pyceles.core.tmatrix import particle_T_diagonal, particle_T_matrix_blocks, sphere_T_diagonal
from pyceles.core.translation import RadialLUT, translation_ab5_table

from .base import PreparedOperator
from .coupling_pairwise import PairwiseCouplingOperator
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
from .single_body import CompositeParticleTOperator

Array = np.ndarray


def build_T_mode_diagonal(lmax: int, T_M: Array, T_N: Array) -> Array:
    """Expand per-(sphere,l) diagonal entries to per-(sphere,mode) factors."""
    lmax = int(lmax)
    T_M = np.asarray(T_M)
    T_N = np.asarray(T_N)

    if T_M.shape != T_N.shape:
        raise ValueError(
            f"T_M and T_N must have identical shapes. Got {T_M.shape} and {T_N.shape}."
        )

    ns = T_M.shape[0]
    nm = n_modes(lmax)
    nscl = lmax * (lmax + 2)
    out_dtype = np.result_type(T_M.dtype, T_N.dtype, np.complex64)
    T_diag = np.zeros((ns, nm), dtype=out_dtype)

    for l in range(1, lmax + 1):
        start = (l - 1) * (l + 1)
        end = start + (2 * l + 1)
        T_diag[:, start:end] = T_M[:, l : l + 1]
        T_diag[:, nscl + start : nscl + end] = T_N[:, l : l + 1]

    return T_diag


def _prepare_diagonal_group(
    *,
    plan: ParticleTGroupPlan,
    lmax: int,
    k: float,
    particles: Sequence[Particle],
    n_medium: complex,
    dtype: np.dtype,
) -> DiagonalTGroup:
    ids = np.asarray(plan.particle_indices, dtype=np.int64)
    group_particles = [particles[int(i)] for i in ids]
    T_M, T_N = precompute_T_diagonal(
        lmax=int(lmax),
        k=float(k),
        particles=group_particles,
        n_medium=n_medium,
        dtype=dtype,
    )
    return DiagonalTGroup(
        particle_indices=ids,
        T_M=T_M,
        T_N=T_N,
        T_diag=build_T_mode_diagonal(int(lmax), T_M, T_N),
        dtype=dtype,
    )


def _default_group_factory(
    *,
    plan: ParticleTGroupPlan,
    context: ParticleTPreparationContext,
) -> PreparedParticleTGroup:
    if plan.representation == "diagonal":
        return _prepare_diagonal_group(
            plan=plan,
            lmax=context.lmax,
            k=context.k,
            particles=context.particles,
            n_medium=context.n_medium,
            dtype=context.dtype,
        )

    if plan.representation == "dense":
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        group_particles = [context.particles[int(i)] for i in ids]
        try:
            blocks = particle_T_matrix_blocks(
                lmax=context.lmax,
                k_medium=context.k,
                particles=group_particles,
                n_medium=context.n_medium,
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
            T_blocks=blocks.astype(context.dtype, copy=False),
            dtype=context.dtype,
        )

    if plan.representation == "axisymmetric":
        ids = np.asarray(plan.particle_indices, dtype=np.int64)
        axis_particles = tuple(context.particles[int(i)] for i in ids)
        try:
            blocks = particle_T_matrix_blocks(
                lmax=context.lmax,
                k_medium=context.k,
                particles=list(axis_particles),
                n_medium=context.n_medium,
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
            T_blocks=blocks.astype(context.dtype, copy=False),
            body_metadata={"storage": "spherical_basis_dense_blocks"},
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
    """Prepare the particle-local operator using planned representation groups."""
    part = tuple(particles)
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
    particle_t_group_factories: ParticleTGroupFactories | None = None,
) -> PreparedOperator:
    """Prepare reusable `A = I - T W` data from explicit particle descriptors."""
    part = list(particles)
    positions = np.asarray(
        [np.asarray(p.position, dtype=float) for p in part], dtype=float
    ).reshape(-1, 3)
    op_dtype = np.dtype(operator_dtype)
    k_f = float(k)
    ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)

    dr_user = float(radial_lut_dr)
    if dr_user < 0.0:
        raise ValueError(f"radial_lut_dr must be >= 0, got {dr_user}.")
    k_abs = float(abs(k_f))
    if k_abs <= 0.0:
        raise ValueError(f"`k` must be non-zero for radial LUT setup. Got {k_f!r}.")
    dr = (1.0e-2 / k_abs) if dr_user == 0.0 else dr_user
    lut = RadialLUT(lmax=int(lmax), k=k_f, r_max=_infer_rmax(positions), dr=dr, dtype=op_dtype)

    particle_t = _prepare_particle_t_operator(
        lmax=int(lmax),
        k=k_f,
        particles=part,
        n_medium=n_medium,
        dtype=op_dtype,
        group_factories=particle_t_group_factories,
    )
    coupling = PairwiseCouplingOperator(
        lmax=int(lmax),
        k=k_f,
        positions=positions,
        ab5=ab5,
        radial_lut=lut,
        dtype=op_dtype,
        cache_translation_blocks=bool(cache_translation_blocks),
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
    return r.reshape(ns * nm)


__all__ = ["build_T_mode_diagonal", "precompute_T_diagonal", "prepare_matvec", "rhs_Tb_numpy"]
