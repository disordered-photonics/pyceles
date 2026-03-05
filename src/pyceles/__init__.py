from pyceles import core, io, linear, postprocessing
from pyceles._logo import print_logo
from pyceles._version import __version__
from pyceles.core.fields import (
    BesselBeam,
    CartesianPolarizedBesselBeam,
    CartesianPolarizedFocusedLaguerreGaussianBeam,
    DipoleCollection,
    DipoleSource,
    FocusedLaguerreGaussianBeam,
    GaussianBeam,
    LaguerreGaussianBeam,
    PlaneWave,
    SLMSource,
    project_source_basis_to_svwf,
    project_source_to_svwf,
)
from pyceles.core.particles import (
    Ellipsoid,
    LayeredSphere,
    Particle,
    Sphere,
    ellipsoids_from_arrays,
    layered_spheres_from_arrays,
    spheres_from_arrays,
)
from pyceles.io.workflows import load_simulation_h5, save_simulation_h5
from pyceles.postprocessing.dipole_metrics import (
    DipolePowerLDOSResult,
    compute_dipole_ldos_enhancement,
    compute_dipole_power_ldos,
)
from pyceles.postprocessing.workflows import (
    NearFieldSlice,
    compute_near_field,
    compute_near_field_slice,
    mix_near_field_components,
    mix_near_field_slices,
)
from pyceles.simulation import (
    MultiSourceSimulationResult,
    Simulation,
    SimulationConfig,
    SimulationResult,
    SolvedSourcesResult,
)

__all__ = [
    "__version__",
    "print_logo",
    "GaussianBeam",
    "LaguerreGaussianBeam",
    "FocusedLaguerreGaussianBeam",
    "BesselBeam",
    "CartesianPolarizedBesselBeam",
    "CartesianPolarizedFocusedLaguerreGaussianBeam",
    "PlaneWave",
    "SLMSource",
    "DipoleSource",
    "DipoleCollection",
    "Particle",
    "Sphere",
    "LayeredSphere",
    "Ellipsoid",
    "spheres_from_arrays",
    "layered_spheres_from_arrays",
    "ellipsoids_from_arrays",
    "project_source_basis_to_svwf",
    "project_source_to_svwf",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "SolvedSourcesResult",
    "MultiSourceSimulationResult",
    "NearFieldSlice",
    "DipolePowerLDOSResult",
    "compute_near_field",
    "compute_near_field_slice",
    "compute_dipole_power_ldos",
    "compute_dipole_ldos_enhancement",
    "mix_near_field_components",
    "mix_near_field_slices",
    "save_simulation_h5",
    "load_simulation_h5",
    "core",
    "io",
    "linear",
    "postprocessing",
]
