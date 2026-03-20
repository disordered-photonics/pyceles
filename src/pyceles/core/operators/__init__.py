from __future__ import annotations

"""Prepared many-body operator package.

This package owns the solver-side split between particle-local scattering
operators `T` and inter-particle coupling operators `W`. New coupling backends
such as periodic, FFT, FMM, or GPU variants should plug in here through the
owned coupling and preparation modules rather than re-entangling the operator
stack into one file.
"""

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
from .coupling_pairwise_cupy import CuPyPairwiseCouplingOperator
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
from .mlfmm import (
    MLFMMCouplingOperator,
    MLFMMLevelOperators,
    MLFMMMultilevelOperators,
    MLFMMOptions,
    MLFMMResolvedPlan,
    MLFMMSingleLevelOperators,
    MLFMMTransferOperators,
    apply_multilevel_mlfmm,
    apply_single_level_mlfmm,
    box_order_rokhlin_like,
    build_multilevel_mlfmm_operators,
    build_single_level_mlfmm_operators,
    estimate_rokhlin_order,
    prepare_mlfmm_coupling,
    resolve_mlfmm_plan,
    select_mlfmm_stage,
)
from .mlfmm_directional import (
    MLFMMDirectionalGrid,
    MLFMMDirectionalInterpolation,
    MLFMMDirectionalTransforms,
    apply_directional_reflection,
    box_outgoing_to_directional,
    directional_anterpolation,
    directional_grid,
    directional_interpolation,
    directional_to_box_regular,
    directional_transforms,
)
from .mlfmm_partition import MLFMMBox, MLFMMPartition, validate_leaf_size_floor
from .prepare import build_T_mode_diagonal, precompute_T_diagonal, prepare_matvec, rhs_Tb_numpy
from .single_body import CompositeParticleTOperator, ParticleTOperator
from .single_body_cupy import CuPyDiagonalParticleTOperator

__all__ = [
    "Array",
    "AxisymmetricTGroup",
    "CompositeParticleTOperator",
    "CouplingOperator",
    "CuPyDiagonalParticleTOperator",
    "CuPyPairwiseCouplingOperator",
    "DenseTGroup",
    "DiagonalTGroup",
    "MLFMMDirectionalGrid",
    "MLFMMDirectionalInterpolation",
    "MLFMMDirectionalTransforms",
    "MLFMMBox",
    "MLFMMCouplingOperator",
    "MLFMMLevelOperators",
    "MLFMMMultilevelOperators",
    "MLFMMOptions",
    "MLFMMPartition",
    "MLFMMResolvedPlan",
    "MLFMMSingleLevelOperators",
    "MLFMMTransferOperators",
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
    "apply_multilevel_mlfmm",
    "apply_single_level_mlfmm",
    "assemble_dense_A_numpy",
    "apply_directional_reflection",
    "build_T_mode_diagonal",
    "box_order_rokhlin_like",
    "box_outgoing_to_directional",
    "build_multilevel_mlfmm_operators",
    "build_single_level_mlfmm_operators",
    "directional_anterpolation",
    "directional_grid",
    "directional_interpolation",
    "directional_to_box_regular",
    "directional_transforms",
    "estimate_translation_cache_bytes",
    "estimate_rokhlin_order",
    "make_axisymmetric_block_group_factory",
    "make_axisymmetric_group_factory",
    "make_dense_group_factory",
    "make_prepared_A_and_rhs",
    "prepare_mlfmm_coupling",
    "plan_particle_t_groups",
    "precompute_T_diagonal",
    "prepare_matvec",
    "resolve_mlfmm_plan",
    "require_pairwise_coupling",
    "rhs_Tb_numpy",
    "select_mlfmm_stage",
    "validate_leaf_size_floor",
]
