from __future__ import annotations

"""Prepared many-body operator package."""

from .base import Array, CouplingOperator, PrecomputableCouplingOperator, PreparedOperator
from .coupling_dense import (
    assemble_dense_A_numpy,
    estimate_translation_cache_bytes,
    make_prepared_A_and_rhs,
)
from .coupling_pairwise import (
    PairwiseCouplingOperator,
    apply_A_numpy,
    apply_W_numpy,
    require_pairwise_coupling,
)
from .groups import (
    AxisymmetricTGroup,
    DenseTGroup,
    DiagonalTGroup,
    ParticleTGroupFactories,
    ParticleTGroupPlan,
    ParticleTPreparationContext,
    PreparedParticleTGroup,
    make_axisymmetric_block_group_factory,
    make_axisymmetric_group_factory,
    make_dense_group_factory,
    plan_particle_t_groups,
)
from .prepare import build_T_mode_diagonal, precompute_T_diagonal, prepare_matvec, rhs_Tb_numpy
from .single_body import CompositeParticleTOperator, ParticleTOperator

__all__ = [
    "Array",
    "AxisymmetricTGroup",
    "CompositeParticleTOperator",
    "CouplingOperator",
    "DenseTGroup",
    "DiagonalTGroup",
    "PairwiseCouplingOperator",
    "ParticleTOperator",
    "ParticleTGroupFactories",
    "ParticleTGroupPlan",
    "ParticleTPreparationContext",
    "PrecomputableCouplingOperator",
    "PreparedOperator",
    "PreparedParticleTGroup",
    "apply_A_numpy",
    "apply_W_numpy",
    "assemble_dense_A_numpy",
    "build_T_mode_diagonal",
    "estimate_translation_cache_bytes",
    "make_axisymmetric_block_group_factory",
    "make_axisymmetric_group_factory",
    "make_dense_group_factory",
    "make_prepared_A_and_rhs",
    "plan_particle_t_groups",
    "precompute_T_diagonal",
    "prepare_matvec",
    "require_pairwise_coupling",
    "rhs_Tb_numpy",
]
