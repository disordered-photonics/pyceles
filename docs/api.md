# API map

This page is a curated map of the intended public API. It is not generated from
docstrings yet, and it intentionally does not list every importable internal
module.

## Simulation workflow

Most user workflows start from:

- `pyceles.SimulationConfig`
- `pyceles.Simulation`
- `pyceles.SimulationResult`
- `pyceles.MultiSourceSimulationResult`
- `pyceles.SolvedSourcesResult`
- `pyceles.ResultRetention`

The high-level pattern is:

```python
import pyceles as pcl

config = pcl.SimulationConfig(...)
result = pcl.Simulation(config, particles=[...]).run()
```

## Particles

Particle descriptors are passed to `Simulation`:

- `pyceles.Sphere`
- `pyceles.PECSphere`
- `pyceles.LayeredSphere`
- `pyceles.Spheroid`
- `pyceles.ParticleCollection`
- `pyceles.spheres_from_arrays`
- `pyceles.pec_spheres_from_arrays`
- `pyceles.layered_spheres_from_arrays`
- `pyceles.spheroids_from_arrays`

`ParticleCollection.from_archetypes(...)` is the generic constructor for many
instances that reuse immutable particle metadata. See
[particle_storage.md](particle_storage.md) for storage and preparation
semantics.

## Sources

Common source descriptors include:

- `pyceles.PlaneWave`
- `pyceles.GaussianBeam`
- `pyceles.LaguerreGaussianBeam`
- `pyceles.FocusedLaguerreGaussianBeam`
- `pyceles.BesselBeam`
- `pyceles.SLMSource`
- `pyceles.DipoleSource`
- `pyceles.DipoleCollection`

## Postprocessing

Near-field helpers:

- `pyceles.compute_near_field`
- `pyceles.compute_near_field_slice`
- `pyceles.compute_periodic_near_field`
- `pyceles.compute_periodic_near_field_slice`
- `pyceles.mix_near_field_components`
- `pyceles.mix_near_field_slices`

Dipole diagnostics:

- `pyceles.compute_dipole_power_ldos`
- `pyceles.compute_dipole_ldos_enhancement`

Power accounting is exposed uniformly through `SimulationResult.power`, an
immutable `pyceles.PowerBalance` for finite beams, periodic plane waves, and
local-source workflows where the corresponding quantities exist:

- absolute fields: `incident_power`, `reflected_power`, `transmitted_power`,
  `local_absorbed_power`, and `local_absorbed_power_per_particle`,
- derived flux diagnostics: `flux_defect = incident - reflected - transmitted`
  and `closure_error = flux_defect - local_absorbed_power`,
- normalized fields: `reflectance`, `transmittance`, `local_absorptance`,
  `flux_defect_fraction`, `closure_error_fraction`, and
  `local_absorptance_per_particle`.

`flux_defect_fraction` is deliberately not named absorptance: it is only a
physical absorption estimate when the local coefficient identity and the
far-field flux evaluation close. For an inexact solve, the local estimate also
contains the equation residual; this makes inadequate solver accuracy visible.

Periodic order-resolved results remain on `pyceles.PeriodicFarFieldPayload`:

- `reflected_amplitudes` and `transmitted_amplitudes`,
- `reflected_flux_per_order` and `transmitted_flux_per_order`,
- `incident_flux`, order indices/wavevectors, and propagation masks,
- `power`, which is the same common balance exposed as `SimulationResult.power`.

There are no `absorptance`, `A_raw`, or historical power-key aliases.

## Periodic descriptors

Experimental periodic workflows use:

- `pyceles.RectangularLattice2D`
- `pyceles.PeriodicSpec`
- `pyceles.PeriodicOptions`
  - `method="ewald"`: exact pairwise Ewald operator (default),
  - `method="directsum"`: small-case NumPy oracle,
  - `method="rayleigh"`: exact self/vertical-near Ewald plus far Rayleigh scans,
  - `rayleigh_z_cut`: exact-near half-band in the simulation length unit,
  - `rayleigh_reciprocal_shells`: fixed reciprocal square half-width or `None`
    for automatic truncation,
  - `shell_tolerance` and `max_shells`: shared Ewald/Rayleigh automatic
    shell-truncation policy. Reciprocal work chunks are selected internally from
    a bounded temporary-memory budget.
- `SimulationConfig(coupling_backend="mlfmm", operator_backend=...)` can be
  combined experimentally with `PeriodicOptions(method="ewald")` on NumPy or
  CuPy. Ewald prepares the periodizing coarse-box operators on the CPU;
  repeated coupling applications use the mesh-free MLFMM hierarchy rather than
  the pairwise Ewald path. Other periodic summation methods are not combined
  with periodized MLFMM.
- `pyceles.core.periodic.rayleigh_report`
- `pyceles.core.periodic.rayleigh_threshold_scales`
- `pyceles.core.periodic.suggest_safe_period_scales`

## I/O

Workflow-level HDF5 helpers:

- `pyceles.save_simulation_h5`
- `pyceles.load_simulation_h5`

## Implementation modules

Modules under `pyceles.core`, `pyceles.linear`, `pyceles.simulation`, and
`pyceles.postprocessing` are importable because pyceles is a Python package, but
not every internal module is a stable user-facing contract. In particular,
operator implementations, CuPy RawKernel helpers, cache objects, and low-level
periodic kernels should be treated as implementation details unless they are
documented here or in the root README.
