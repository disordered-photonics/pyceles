# API map

This page is a curated map of the intended public API. It is not generated from
docstrings yet, and it intentionally does not list every importable internal
module.

## Simulation workflow

Most user workflows start from:

- `pyceles.SimulationConfig`
- `pyceles.Simulation`
- `pyceles.SimulationResult` for one directly solved source
- `pyceles.ChannelResult` for a channel owned by a block result
- `pyceles.MultiSourceResult` and `pyceles.MultiSourceSolveResult`
- `pyceles.PolarizationResult`
- `pyceles.ResultRetention`

The high-level pattern is:

```python
import pyceles as pcl

config = pcl.SimulationConfig(...)
simulation = pcl.Simulation(config, particles=[...])
result = simulation.run(source)

multi = simulation.run_sources({"left": source_a, "right": source_b})
polarized = simulation.run_polarizations(jones_source)
```

## Linear solvers and precision

`SimulationConfig.solver_method` accepts `None`, `auto`, `direct`, `gmres`,
`fgmres`, `lgmres`, `bicgstab`, `gcro`, and `gcrotmk`. NumPy/SciPy provides direct,
GMRES, BiCGSTAB, LGMRES, and GCROTMK; CuPy provides direct and native GMRES,
FGMRES, LGMRES, BiCGSTAB, and harmonic GCRO-DR. CuPy GMRES also handles a
two-dimensional right-hand side with native block GMRES. `fgmres` and `gcro`
are CuPy-only and `gcrotmk` is SciPy-only. GCRO is a forward-only
harmonic-recycling method with a single-RHS recurrence; multi-RHS workflows
solve each column independently rather than using block GCRO. It exposes only
`solver_recycle_dim` in addition to the common restart/tolerance/budget
controls, while its extraction policy remains fixed internally. The initial
GCRO path does not accept a custom preconditioner. With `None`
(the high-level default), finite systems select BiCGSTAB and periodic systems
select restarted GMRES. Explicit `auto` selects a direct solve below
`solver_direct_max_n` and GMRES otherwise.

`compute_dtype` controls compact operator and contraction arithmetic, while
`accum_dtype` sets the wider precision budget for reductions on paths that
support it. Native CuPy Krylov routines honor this policy; SciPy Krylov
routines retain SciPy's own internal arithmetic. Periodic Ewald scalar lattice
sums retain complex128 arithmetic for cancellation safety even when compact
operator data use complex64.

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
- `pyceles.CartesianPolarizedBesselBeam`
- `pyceles.CartesianPolarizedFocusedLaguerreGaussianBeam`
- `pyceles.AngularSpectrumSLMSource`
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

Power accounting is exposed uniformly through `ChannelResult.power`
(including the directly solved `SimulationResult` subtype), as an immutable
`pyceles.PowerBalance` for finite beams, periodic plane waves, and
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
- `power`, which is the same common balance exposed as `ChannelResult.power`.

There are no `absorptance`, `A_raw`, or historical power-key aliases.
Finite-beam forward/backward field contributions are exposed as immutable
`pyceles.PowerFluxDecomposition` objects with `initial_power`,
`scattered_power`, `interference_power`, and `total_power`.

Plane-wave cross sections use the same typed-diagnostic policy through
`ChannelResult.cross_sections`, an immutable `pyceles.CrossSectionBalance`:

- `extinction`, `scattering`, and `local_absorption`,
- `absorption_by_difference = extinction - scattering`,
- `closure_error = absorption_by_difference - local_absorption`.

The low-level constructor is `plane_wave_cross_section_balance(...)`; it
requires the local absorption estimate explicitly and exposes no `C_*` aliases.
`PolarizationResult.te` and `.tm` expose one balance per solved basis channel,
while `.unpolarized` is a typed `pyceles.UnpolarizedDiagnostics`. The coherent
Jones channel is available as the lazily materialized `.mixed` channel.

## Periodic descriptors

Experimental periodic workflows use:

- `pyceles.RectangularLattice2D`
- `pyceles.PeriodicSpec`
- `pyceles.PeriodicOptions`
  - `method="ewald"`: exact pairwise Ewald operator (default),
  - `method="directsum"`: small-case NumPy oracle (NumPy backend only),
  - `method="rayleigh"`: exact self/vertical-near Ewald plus far Rayleigh scans,
  - `rayleigh_z_cut`: exact-near half-band in the simulation length unit,
  - `rayleigh_reciprocal_shells`: fixed reciprocal square half-width or `None`
    for automatic truncation,
  - `shell_tolerance` and `max_shells`: shared Ewald/Rayleigh automatic
    shell-truncation policy. Reciprocal work chunks are selected internally from
    a bounded temporary-memory budget.
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
