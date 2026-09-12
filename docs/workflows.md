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
source-aware `pwp_power_decomposition`, and the common `ChannelResult.power`
report, are enabled only when `has_finite_incident_power()` returns `True`.
`pwp_power_decomposition` returns an immutable `PowerFluxDecomposition` rather
than a string-keyed mapping.
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

## Explicit source cardinality

`SimulationConfig` describes the reusable physical and numerical system; source
cardinality is explicit in the operation being called:

```python
sim = pcl.Simulation(cfg, particles=particles)

single = sim.run(source)
multi = sim.run_sources({"left": source_a, "right": source_b})
polarized = sim.run_polarizations(source)
```

`run_sources(...)` assembles all right-hand sides and sends them through one
shared multi-RHS solve. Labels are explicit non-empty strings and preserve mapping
insertion order; they are never normalized or stringified implicitly. This is the
preferred path for dipole bases, SLM/Hadamard pattern batches, wavelength-compatible
source scans, and any other channels that share one operator. The result owns one
authoritative block solver report and a labeled mapping of solver-free
`ChannelResult` objects:

```python
multi = sim.run_sources(
    {
        "te": source.with_polarization("TE"),
        "tm": source.with_polarization("TM"),
    }
)

run_te = multi["te"]
run_tm = multi["tm"]
print(multi.labels)
print(multi.solver_result.rhs_count)
```

For staged workflows, `solve_sources(...)` returns a `MultiSourceSolveResult`
that can be passed later to `postprocess_sources(...)`. This preserves the same
block solve while allowing custom far-field grids or deferred postprocessing.

## Result retention

Completed results retain solved multipole coefficients because they are the
restart and deferred-postprocessing state. Optional incident coefficients,
right-hand sides, and residual histories can be omitted:

```python
run = sim.run(
    source,
    include_farfield=False,
    retention=pcl.ResultRetention.minimal(),
)
```

For multi-source and polarization runs, every solved RHS remains essential and
is retained exactly once through channel views of the shared block solution.
Minimal retention does not keep hidden basis copies or a permanently materialized
coherent mixture.

## Polarization runs

`run_polarizations(...)` is a typed specialization of the same multi-RHS path.
It solves the orthogonal TE/TM basis in one block Krylov or direct solve and
returns a `PolarizationResult`:

```python
polarized = sim.run_polarizations(source)

te = polarized.te
tm = polarized.tm
mixed = polarized.mixed

print(polarized.solver_result.rhs_count)  # 2
print(polarized.unpolarized.power)
print(polarized.unpolarized.cross_sections)
```

`te` and `tm` are the two solved `ChannelResult` objects. `mixed` is the coherent
Jones combination requested by `source`; it is derived on demand and therefore
does not pretend to own an independent solver report or permanently retain a
third coefficient vector. `unpolarized` contains the incoherent TE/TM averages
of scalar diagnostics. The same API works for finite and periodic plane-wave
configurations.

Near-field and plotting helpers accept explicit channels rather than string
selectors. Finite and periodic channels also use distinct entry points, so the
physical field formulation cannot be selected accidentally:

```python
nf_mixed = pcl.compute_near_field_slice(polarized.mixed, ...)
nf_te = pcl.compute_near_field_slice(polarized.te, ...)
nf_tm = pcl.compute_near_field_slice(polarized.tm, ...)

# For a periodic `PolarizationResult` named `periodic_result`:
# periodic_nf = pcl.compute_periodic_near_field_slice(periodic_result.mixed, ...)
I_unpolarized = pcl.io.unpolarized_far_field_intensity(polarized)
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
- Dipole sources do not have a TE/TM Jones basis. Use
  `DipoleSource.cartesian_basis_sources()` with `run_sources(...)` when the
  x/y/z dipole orientation basis should be solved in one multi-RHS call.
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
multi = sim.run_sources(
    dip.cartesian_basis_sources(),
    include_farfield=False,
)
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
run = sim.run(source, include_farfield=False)
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
).run(source)
```

This yields zero scattered coefficients/fields and preserves incident-field
outputs.

## Warm starts and preconditioners

Warm starts are execution state, so they are passed to the operation rather
than stored in the reusable configuration:

```python
def M_inv_mv(v):
    # Replace this identity with the desired approximate inverse.
    return v


cfg = pcl.SimulationConfig(
    solver_method="gmres",
    solver_preconditioner=M_inv_mv,
)
sim = pcl.Simulation(cfg, particles=particles)

run = sim.run(source, warm_start=previous.coeffs)
multi = sim.run_sources(
    {"left": source_a, "right": source_b},
    warm_start={"left": previous_left.coeffs, "right": previous_right.coeffs},
)
```

For multi-source operations, one coefficient vector is broadcast to every RHS.
A block array may instead use its final axis in source-label order, or a mapping
may provide one initial guess per label. `run_polarizations(...)` follows the
same rule with the labels `"te"` and `"tm"`.

Notes:

- For many-sphere systems, `n = N_spheres * n_modes(lmax)`.
- An explicit `solver_method="direct"` assembles and factorizes the full dense
  operator without a pyceles size heuristic; the caller is responsible for
  memory availability. An oversized request therefore fails at the backend's
  normal allocation or factorization boundary.
- Repeated direct solves on the same `Simulation` instance reuse both dense `A`
  and its LU factorization.
- `SimulationConfig.solver_compute_final_residual` controls true-residual
  verification/diagnostics policy.
- `solver_method="gcro"` selects the native CuPy harmonic recycling solver for
  single-RHS runs; `SimulationConfig.solver_gcro_recycle_dim` controls its bounded
  recycle rank. The common `solver_restart` value is the total augmented
  dimension.
- `solver_method="lsqr"` selects a single-RHS LSQR solver. The NumPy backend
  delegates to SciPy's reference implementation; CuPy uses the native device
  recurrence. Both require an exact Hermitian adjoint of the prepared
  `A = I - T W` action. Pairwise finite coupling and periodic Rayleigh, Ewald,
  and direct-sum coupling provide this when particle-T groups expose matching
  forward/adjoint actions, including explicit dense blocks. Finite MLFMM also
  provides matching forward/adjoint actions on NumPy and CuPy; on CuPy, a
  multilevel reverse apply currently retains the sampled hierarchy. Periodic
  direct-sum is available on the NumPy/reference
  backend; CuPy periodic workflows currently use Ewald or Rayleigh. LSQR does
  not restart; `solver_maxiter` is its iteration budget and `solver_restart` is
  ignored.
- For native CuPy restarted GMRES/FGMRES/LGMRES, final-residual checks are
  performed at restart boundaries by default; set the option to `False` only
  for profiling-focused runs.
- SciPy GMRES progress reports SciPy's cheap preconditioned residual
  (`pr_rel_res`). SciPy BiCGSTAB, LGMRES, and GCROTMK callbacks do not expose a
  cheap residual scalar, so pyceles reports iteration-only progress for those
  methods instead of spending an extra matrix-vector product per callback.
- `warm_start` is a user-supplied initial guess on `run(...)`, `solve_sources(...)`,
  `run_sources(...)`, or `run_polarizations(...)`. It is not automatically loaded
  from HDF5, but previously solved coefficients can be supplied when dimensions
  and mode ordering match.
- `solver_preconditioner` is a custom callable hook: `M_inv_mv(v)` should return
  an approximate application of `M^{-1} v` for the current linear system.
- pyceles does not select a built-in grid-block preconditioner in the high-level
  simulation API. Use `solver_preconditioner` when a workload has a suitable
  custom preconditioner.

## HDF5 output for explicit channels

`pcl.io.save_simulation_h5(...)` stores one explicit `ChannelResult` or
`SimulationResult`: particle-native geometry, coefficients, optional incident/RHS
arrays, near field, finite far field or periodic diffraction orders, and typed
scalar diagnostics. A direct `SimulationResult` also carries its matching
single-RHS solver metadata.

Multi-source and polarization envelopes are not serialized as disguised single
runs. Select the physical channel deliberately:

```python
pcl.save_simulation_h5(multi["left"], near_left, "left.h5")
pcl.save_simulation_h5(polarized.te, near_te, "te.h5")
pcl.save_simulation_h5(polarized.mixed, near_mixed, "mixed.h5")
```

The shared block solver report remains on `multi.solver_result` or
`polarized.solver_result`; derived coherent channels never receive fabricated
solver provenance. Loading helpers are available via `pcl.io`:

- `load_geometry_h5`,
- `load_solution_h5`,
- `load_far_field_h5`,
- `load_near_field_components_h5`,
- `load_mapping_h5`,
- `load_simulation_h5`.
