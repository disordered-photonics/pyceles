"""Prepared many-body operator package.

This package owns the solver-side split between particle-local scattering
operators `T` and inter-particle coupling operators `W`. New coupling backends
such as periodic, FFT, FMM, or GPU variants should plug in here through the
owned coupling and preparation modules rather than re-entangling the operator
stack into one file.
"""

from __future__ import annotations

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
    make_axisymmetric_block_group_factory,
    make_axisymmetric_group_factory,
    make_dense_group_factory,
    plan_particle_t_groups,
)
from .mlfmm import (
    MLFMMCouplingOperator,
    MLFMMOptions,
    MLFMMResolvedPlan,
    prepare_mlfmm_coupling,
    resolve_mlfmm_plan,
)
from .mlfmm_cupy import (
    CuPyMLFMMCouplingOperator,
    CuPyMLFMMHostCacheData,
    CuPyMLFMMHostCachePolicy,
    CuPyMLFMMPreparedData,
    build_mlfmm_cupy_host_cache,
    prepare_mlfmm_cupy_coupling,
    prepare_mlfmm_cupy_data,
)
from .prepare import precompute_T_diagonal, prepare_matvec, rhs_Tb_numpy
from .single_body import CompositeParticleTOperator, ParticleTOperator, build_T_mode_diagonal
from .single_body_cupy import CuPyDiagonalParticleTOperator

__all__ = [
    "Array",
    "AxisymmetricTGroup",
    "CompositeParticleTOperator",
    "CouplingOperator",
    "CuPyDiagonalParticleTOperator",
    "CuPyMLFMMCouplingOperator",
    "CuPyMLFMMHostCacheData",
    "CuPyMLFMMHostCachePolicy",
    "CuPyMLFMMPreparedData",
    "CuPyPairwiseCouplingOperator",
    "CuPyPeriodicCouplingOperator",
    "DenseTGroup",
    "DiagonalTGroup",
    "MLFMMCouplingOperator",
    "MLFMMOptions",
    "MLFMMResolvedPlan",
    "PairwiseCouplingOperator",
    "ParticleTGroupFactories",
    "ParticleTGroupPlan",
    "ParticleTOperator",
    "ParticleTPreparationContext",
    "PeriodicCouplingOperator",
    "PrecomputableCouplingOperator",
    "PreparedOperator",
    "PreparedParticleTGroup",
    "apply_A_numpy",
    "apply_W_numpy",
    "assemble_dense_A_numpy",
    "build_T_mode_diagonal",
    "build_mlfmm_cupy_host_cache",
    "estimate_translation_cache_bytes",
    "make_axisymmetric_block_group_factory",
    "make_axisymmetric_group_factory",
    "make_dense_group_factory",
    "make_prepared_A_and_rhs",
    "plan_particle_t_groups",
    "precompute_T_diagonal",
    "prepare_matvec",
    "prepare_mlfmm_coupling",
    "prepare_mlfmm_cupy_coupling",
    "prepare_mlfmm_cupy_data",
    "require_pairwise_coupling",
    "resolve_mlfmm_plan",
    "rhs_Tb_numpy",
]
