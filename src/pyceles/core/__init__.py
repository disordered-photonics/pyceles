"""Core numerical kernels and assembly helpers."""

from .angular import uniform_periodic_azimuth_grid, uniform_polar_grid
from .fields import (
    AngularSpectrumSource,
    BesselBeam,
    CartesianPolarizedBesselBeam,
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    DipoleCollection,
    DipoleSource,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    JonesPolarizedSource,
    LaguerreGaussianBeam,
    LocalExpansionSource,
    PlaneWave,
    SLMSource,
    Source,
    angular_spectrum_to_svwf_regular,
    polarization_to_jones,
    project_source_basis_to_svwf,
    project_source_to_svwf,
    pwp_to_svwf_regular,
    svwf_outgoing_to_pwp,
    svwf_regular_to_pwp,
)
from .operators import (
    AxisymmetricTGroup,
    CompositeParticleTOperator,
    CouplingOperator,
    DenseTGroup,
    DiagonalTGroup,
    ParticleTGroupFactories,
    ParticleTGroupPlan,
    ParticleTPreparationContext,
    PreparedOperator,
    apply_A_numpy,
    apply_W_numpy,
    assemble_dense_A_numpy,
    estimate_translation_cache_bytes,
    make_axisymmetric_block_group_factory,
    make_axisymmetric_group_factory,
    make_dense_group_factory,
    make_prepared_A_and_rhs,
    plan_particle_t_groups,
    precompute_T_diagonal,
    prepare_matvec,
    rhs_Tb_numpy,
)
from .particles import (
    LayeredSphere,
    Particle,
    Sphere,
    Spheroid,
    layered_spheres_from_arrays,
    spheres_from_arrays,
    spheroids_from_arrays,
)
from .spherical import clear_caches as _clear_spherical_caches
from .svwf_rotation import clear_caches as _clear_svwf_rotation_caches
from .svwf_rotation import rotate_svwf_tmatrix_block, svwf_rotation_matrix
from .tmatrix import (
    layered_internal_ab_ratios,
    layered_mie_ab,
    layered_sphere_T_diagonal,
    particle_internal_ratios,
    particle_T_diagonal,
    particle_T_matrix_block,
    particle_T_matrix_blocks,
)
from .translation import clear_caches as _clear_translation_caches
from .wigner import clear_caches as _clear_wigner_caches


def clear_caches() -> None:
    """Clear process-global core precompute caches.

    This is an explicit memory-management tool for interactive work and large
    parameter sweeps. `pyceles` does not clear these caches automatically
    because repeated solves in the same process often benefit from keeping the
    translation, rotation, and angular-recurrence tables warm.
    """

    _clear_translation_caches()
    _clear_spherical_caches()
    _clear_svwf_rotation_caches()
    _clear_wigner_caches()


__all__ = [
    "AngularSpectrumSource",
    "AxisymmetricTGroup",
    "BesselBeam",
    "CartesianPolarizedBesselBeam",
    "CartesianPolarizedFocusedLaguerreGaussianBeam",
    "CompositeParticleTOperator",
    "CouplingOperator",
    "DenseTGroup",
    "DiagonalTGroup",
    "DipoleCollection",
    "DipoleSource",
    "FocusedLaguerreGaussianBeam",
    "GaussianBeam",
    "JonesPolarizedSource",
    "LaguerreGaussianBeam",
    "LayeredSphere",
    "LocalExpansionSource",
    "Particle",
    "ParticleTGroupFactories",
    "ParticleTGroupPlan",
    "ParticleTPreparationContext",
    "PlaneWave",
    "PreparedOperator",
    "SLMSource",
    "Source",
    "Sphere",
    "Spheroid",
    "angular_spectrum_to_svwf_regular",
    "apply_A_numpy",
    "apply_W_numpy",
    "assemble_dense_A_numpy",
    "clear_caches",
    "estimate_translation_cache_bytes",
    "layered_internal_ab_ratios",
    "layered_mie_ab",
    "layered_sphere_T_diagonal",
    "layered_spheres_from_arrays",
    "make_axisymmetric_block_group_factory",
    "make_axisymmetric_group_factory",
    "make_dense_group_factory",
    "make_prepared_A_and_rhs",
    "particle_T_diagonal",
    "particle_T_matrix_block",
    "particle_T_matrix_blocks",
    "particle_internal_ratios",
    "plan_particle_t_groups",
    "polarization_to_jones",
    "precompute_T_diagonal",
    "prepare_matvec",
    "project_source_basis_to_svwf",
    "project_source_to_svwf",
    "pwp_to_svwf_regular",
    "rhs_Tb_numpy",
    "rotate_svwf_tmatrix_block",
    "spheres_from_arrays",
    "spheroids_from_arrays",
    "svwf_outgoing_to_pwp",
    "svwf_regular_to_pwp",
    "svwf_rotation_matrix",
    "uniform_periodic_azimuth_grid",
    "uniform_polar_grid",
]
