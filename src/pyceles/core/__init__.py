"""Core numerical kernels and assembly helpers."""

from .angular import uniform_periodic_azimuth_grid, uniform_polar_grid
from .fields import (
    AngularSpectrumSource,
    DipoleCollection,
    DipoleSource,
    GaussianBeam,
    PlaneWave,
    Source,
    incident_coeffs_from_angular_spectrum,
    incident_coeffs_from_pwp,
    polarization_to_jones,
    project_source_basis_to_svwf,
    project_source_to_svwf,
    source_jones,
)
from .matvec import (
    PreparedMatvec,
    apply_A_numpy,
    apply_W_numpy,
    assemble_dense_A_numpy,
    estimate_translation_cache_bytes,
    make_prepared_A_and_rhs,
    precompute_T_diagonal,
    precompute_T_diagonal_from_particles,
    prepare_matvec,
    rhs_Tb_numpy,
)
from .particles import Ellipsoid, LayeredSphere, Particle, Sphere
from .tmatrix import particle_internal_ratios, particle_T_diagonal

__all__ = [
    "Ellipsoid",
    "AngularSpectrumSource",
    "DipoleSource",
    "DipoleCollection",
    "uniform_periodic_azimuth_grid",
    "uniform_polar_grid",
    "GaussianBeam",
    "LayeredSphere",
    "Particle",
    "PlaneWave",
    "Source",
    "PreparedMatvec",
    "Sphere",
    "assemble_dense_A_numpy",
    "apply_A_numpy",
    "apply_W_numpy",
    "estimate_translation_cache_bytes",
    "make_prepared_A_and_rhs",
    "particle_T_diagonal",
    "particle_internal_ratios",
    "precompute_T_diagonal",
    "precompute_T_diagonal_from_particles",
    "prepare_matvec",
    "incident_coeffs_from_angular_spectrum",
    "incident_coeffs_from_pwp",
    "polarization_to_jones",
    "project_source_to_svwf",
    "project_source_basis_to_svwf",
    "rhs_Tb_numpy",
    "source_jones",
]
