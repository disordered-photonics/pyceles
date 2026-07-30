# Workflow notes

This page collects usage and API notes that are useful for advanced workflows
but too detailed for the root README.

## Source capability contract

New source classes should satisfy the internal `Source` protocol in
`pyceles.core.sources` and explicitly implement:

- `incident_coeffs(...)`,
- `with_polarization(...)` and `jones_coefficients()` for TE/TM propagating sources,
- `has_finite_incident_power()` for finite-power diagnostics policy.

Finite-beam-only diagnostics, including `finite_beam_power_balance`,
source-aware `pwp_power_decomposition`, and the common `SimulationResult.power`
report, are enabled only when `has_finite_incident_power()` returns `True`.
This keeps policy centralized and avoids class-name-specific special cases as
new source wrappers/classes are added.

## Angular-grid policy

pyceles keeps a single shared default angular grid to stay close to CELES usage
and avoid unnecessary complexity in standard workflows.

The API still keeps source and far-field grids distinct in principle:

- `source_polar_angles`, `source_azimuthal_angles`: quadrature nodes used to
  project the source onto SVWF coefficients,
- `farfield_polar_angles`, `farfield_azimuthal_angles`: output bins used for
  scattered, initial, and total far-field PWPs.

Physics and numerical implications:

- The linear operator `(I - T W)` is independent of these angular grids.
- For finite-width beams, source-grid quality controls RHS fidelity. A finer
  far-field grid cannot recover source angular content that was undersampled
  during source projection.
- Plane-wave source projection is analytic, so this source-grid issue does not
  apply to plane-wave excitation.

Default behavior remains shared unless overrides are explicitly set:

```python
cfg = pcl.SimulationConfig(
    source=source,
    polar_angles=pcl.core.uniform_polar_grid(3601),
    azimuthal_angles=pcl.core.uniform_periodic_azimuth_grid(180),
    # Optional advanced overrides:
    # source_polar_angles=...,
    # source_azimuthal_angles=...,
    # farfield_polar_angles=...,
    # farfield_azimuthal_angles=...,
)
```

Relation to CELES and SMUTHI:

- CELES typically uses one shared numerics grid for both source-related
  quadratures and far-field postprocessing.
- SMUTHI exposes more independent angular controls, especially in layered-media
  workflows. pyceles does not currently target planar interfaces, but keeps this
  separation available for future flexibility and expert use.

## Polarization API

All propagating sources support:

- `polarization="TE"` or `"TM"`,
- `polarization=(a_te, a_tm)` with complex Jones-like amplitudes.

An SLM-style modulation wrapper is available for angular-spectrum sources:

```python
source = pcl.SLMSource(
    base_source=pcl.GaussianBeam(
        wavelength=550.0,
        medium_n=1.0 + 0j,
        polarization=(1.0 + 0j, 0.3j),
        beam_width=1800.0,
    ),
    modulation=lambda alpha, beta: np.exp(-1j * 0.2 * np.cos(alpha) * np.sin(beta)),
)
```

Jones-like plane-wave example:

```python
source = pcl.PlaneWave(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    polarization=(1.0 + 0j, 1.0j),
    polar_angle=0.3,
    azimuthal_angle=0.1,
)
```

## Multi-source solves

Use `solve_sources(...)` when multiple channels share one geometry, for example
TE/TM basis sources, dipole x/y/z orientations, or SLM pattern sweeps. Then call
`postprocess_sources(...)` when channel-level far-field or power diagnostics are
needed:

```python
sim = pcl.Simulation(cfg, particles=particles)
solved = sim.solve_sources(
    {
        "te": source.with_polarization("TE"),
        "tm": source.with_polarization("TM"),
    }
)
multi = sim.postprocess_sources(solved)

run_te = multi["te"]
run_tm = multi["tm"]
print(solved.solver_result.rhs_count)
print(solved.solver_result.method)
```

## Result retention

Completed results retain the full solve and postprocessing payload by default.
For large solves where only the solution and observables are needed, optional
duplicate arrays and residual histories can be omitted:

```python
run = sim.run(
    include_farfield=False,
    retention=pcl.ResultRetention.minimal(),
)
```

Solved multipole coefficients in `run.coeffs` are always retained because they
are needed for restarts and deferred postprocessing. Minimal retention omits
the incident coefficients, right-hand side, residual histories, and optional
TE/TM basis coefficient maps. It does not reduce the information available
during the solve or the requested postprocessing itself. The same policy can
be passed to `postprocess_sources(...)`.

## Dual-basis convenience runs

For mixed+basis+unpolarized outputs in one `SimulationResult`, use:

```python
cfg = pcl.SimulationConfig(
    source=source,
    solve_polarization_basis=True,
    solver_method="gmres",
)
run = pcl.Simulation(cfg, particles=particles).run()

mixed_coeffs = run.coeffs
te_coeffs = run.coeffs_basis["te"]
tm_coeffs = run.coeffs_basis["tm"]
assert run.unpolarized is not None
print(run.unpolarized.power)
print(run.unpolarized.cross_sections)
```

Near-field evaluation can target mixed or basis channels:

```python
nf_mixed = pcl.compute_near_field_slice(run, channel="mixed")
nf_te = pcl.compute_near_field_slice(run, channel="te")
nf_tm = pcl.compute_near_field_slice(run, channel="tm")
```

For `postprocess_sources(solve_sources(...))` outputs, each channel run is
already pure. Use `channel="mixed"` on that channel result:

```python
nf_te = pcl.compute_near_field_slice(multi["te"], channel="mixed")
nf_tm = pcl.compute_near_field_slice(multi["tm"], channel="mixed")
```

For far-field plotting, an unpolarized intensity convenience helper is
available:

```python
I_u = pcl.io.far_field_intensity_from_result(run, channel="unpolarized")
```

## Local dipole sources

Use dipole moments/orientations directly, without a TE/TM polarization basis:

```python
source = pcl.DipoleSource(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    position=(0.0, 0.0, 0.0),
    dipole_moment=(1.0 + 0j, 0.0 + 0j, 0.0 + 0j),
)
```

`dipole_moment` is a complex 3-vector. You can treat it as
`dipole_moment = amplitude * direction`, where `direction` can itself be complex
for relative component phases or elliptical source orientation.

For multiple dipoles in one source channel:

```python
source = pcl.DipoleCollection(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    positions=np.array([[0, 0, 0], [200, 0, 0]], dtype=float),
    dipole_moments=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.complex128),
)
```

Current dipole scope:

- homogeneous medium is still restricted to real `n_medium` in the main solver.
  This restriction was inherited from the early beam/plane-wave workflow; it is
  not a fundamental limitation of local dipole sources and may be lifted later.
- `solve_polarization_basis=True` is not defined for dipole sources. Use
  `DipoleSource.cartesian_basis_sources()` when the x/y/z dipole orientation
  basis should be solved in one multi-RHS call.
- dipole homogeneous-background dissipated-power helpers are available:
  - `DipoleSource.dissipated_power_homogeneous_background()`,
  - `DipoleCollection.dissipated_power_homogeneous_background()`,
  - `DipoleCollection.dissipated_power_homogeneous_background_per_dipole()`,
- dipole far-field helper provides direct, particle-scattered, and coherent
  total PWPs,
- dipole `SimulationResult` objects do not carry TE/TM Jones metadata,
- dipole centers are expected in the homogeneous host medium outside particle
  circumscribing spheres.

Dipole moments use the same length-unit convention as geometry and wavelength.
A practical reference scale is `|p| ~ k0^-3 = (wavelength / (2*pi))^3`; for
`wavelength=550` in nanometer units this is about `6.7e5`.

Power/LDOS helpers:

```python
res = pcl.compute_dipole_power_ldos(run)
print(res.power_total, res.power_homogeneous, res.enhancement)

purcell = pcl.compute_dipole_ldos_enhancement(run)
```

These helpers evaluate the particle-scattered field at dipole positions and do
not sample direct self-fields at `r=0`.

To solve one dipole position in x/y/z orientations with one multi-RHS call:

```python
dip = pcl.DipoleSource(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    position=(0.0, 0.0, 0.0),
)
solved = sim.solve_sources(dip.cartesian_basis_sources())
multi = sim.postprocess_sources(solved, include_farfield=False)
```

## Mixed particle descriptors

For layered or mixed spherical geometries, construct simulations from explicit
particle descriptors:

```python
sim = pcl.Simulation(
    cfg,
    particles=[
        pcl.Sphere(position=(0.0, 0.0, 0.0), radius=60.0, refractive_index=1.5 + 0j),
        pcl.LayeredSphere(
            position=(250.0, 0.0, 0.0),
            layer_radii=(40.0, 90.0),
            layer_refractive_indices=(1.8 + 0j, 1.35 + 0.02j),
        ),
    ],
)
run = sim.run(include_farfield=False)
print(sim.n_particles, run.n_particles)
```

Layered spheres are treated as one particle each: one center and one solved
outgoing-coefficient block.

Array-to-particle helper for homogeneous spheres:

```python
particles = pcl.spheres_from_arrays(
    positions=positions,
    radii=radii,
    refractive_indices=n_particle,
)
sim = pcl.Simulation(cfg, particles=particles)
```

The array helpers return immutable `ParticleCollection` objects. All helpers,
including layered spheres and spheroids, deduplicate repeated metadata into
shared archetypes. Combine separate material or particle-family batches without
expanding them into a mutable list:

```python
particles = pcl.ParticleCollection.concatenate(batch_a, batch_b)
```

For third-party or imported particle descriptions, construct the same layout
directly with `ParticleCollection.from_archetypes(...)`. Scaling then follows
the number of unique archetypes/operators, not a special-case particle class.

## Periodic Rayleigh/Wood safety checks

Artificial periodic cells have geometric Rayleigh/Wood anomalies when a
reciprocal-lattice diffraction order becomes grazing:

```text
|k_parallel + m*b1 + n*b2| = k_host
```

These are not Ewald bugs; they are singular channels of the artificial lattice.
Before using a periodic cell to mimic a laterally large disordered sample, check
the host-wavelength-scaled periods and their nearest grazing orders:

```python
source = pcl.PlaneWave(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    polarization="TE",
)
lattice = pcl.RectangularLattice2D(ax=7.5 * source.wavelength, ay=7.5 * source.wavelength)
k = 2 * np.pi * source.medium_n.real / source.wavelength

report = pcl.core.periodic.rayleigh_report(
    lattice,
    k=k,
    k_parallel=pcl.core.plane_wave_k_parallel(source),
)
print(report.warning_level)
print(report.message)
```

For square or scaled rectangular cells, `suggest_safe_period_scales(...)` ranks
wide Rayleigh-clear candidates in units of host wavelength:

```python
candidates = pcl.core.periodic.suggest_safe_period_scales(
    scale_min=5.0,
    scale_max=10.0,
)
for candidate in candidates[:3]:
    print(candidate.scale, candidate.clearance)
```

This helper is intentionally geometry-only. It predicts where singularities can
occur, but it does not decide whether a particular material system will respond
strongly or where the minimum of a full coupling-norm scan sits inside a gap.
More expensive coupling-norm scans can be useful for internal diagnostics, but
they depend on modal truncation, norm convention, representative displacement,
and Ewald convergence policy.

## Source-only runs

For beam inspection/debugging, run a simulation with no particles:

```python
run = pcl.Simulation(
    cfg,
    particles=[],
).run()
```

This yields zero scattered coefficients/fields and preserves incident-field
outputs.

## Warm starts and preconditioners

```python
cfg = pcl.SimulationConfig(
    source=source,
    solver_method="gmres",
    solver_warm_start=x0,
    solver_preconditioner=M_inv_mv,
)
```

Notes:

- `direct_max_n` limits the matrix size `n` of the linear system, not the number
  of RHS columns.
- For many-sphere systems, `n = N_spheres * n_modes(lmax)`.
- `SimulationConfig.solver_direct_max_n` is the high-level knob passed to the
  low-level direct solver guard.
- Repeated direct solves on the same `Simulation` instance reuse both dense `A`
  and its LU factorization.
- `SimulationConfig.solver_compute_final_residual` controls true-residual
  verification/diagnostics policy.
- For native CuPy restarted GMRES/FGMRES/LGMRES, final-residual checks are
  performed at restart boundaries by default; set the option to `False` only
  for profiling-focused runs.
- SciPy GMRES progress reports SciPy's cheap preconditioned residual
  (`pr_rel_res`). SciPy BiCGSTAB, LGMRES, and GCROTMK callbacks do not expose a
  cheap residual scalar, so pyceles reports iteration-only progress for those
  methods instead of spending an extra matrix-vector product per callback.
- `solver_warm_start` is a user-supplied initial guess. It is not automatically
  loaded from HDF5, but you can pass previously solved coefficients from a
  nearby configuration or wavelength sweep when dimensions and mode ordering
  match.
- `solver_preconditioner` is a custom callable hook: `M_inv_mv(v)` should return
  an approximate application of `M^{-1} v` for the current linear system.
- pyceles no longer ships a built-in grid-block preconditioner in the high-level
  simulation API. That implementation was explored, but it did not deliver
  robust speedups relative to its maintenance cost.

## HDF5 output with basis channels

`pcl.io.save_simulation_h5(...)` stores:

- mixed solution/far-field groups,
- particle-native geometry under `geometry/particles`,
- optional basis groups when available:
  - `solution_basis/te`,
  - `solution_basis/tm`,
  - `far_field_basis/te`,
  - `far_field_basis/tm`,
- diagnostics under `diagnostics`:
  - mixed and basis power/cross-sections/decompositions,
  - unpolarized diagnostics,
  - Jones weights.

For `postprocess_sources(solve_sources(...))`, save each channel result
separately, for example `save_simulation_h5(multi["te"], ...)`.

Loading helpers are available via `pcl.io`:

- `load_geometry_h5`,
- `load_solution_h5`,
- `load_far_field_h5`,
- `load_near_field_components_h5`,
- `load_mapping_h5`,
- `load_simulation_h5`.
