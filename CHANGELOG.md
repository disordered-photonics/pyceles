# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `PECSphere` particle descriptors and PEC-sphere helpers based on the
  analytic perfect-conductor Mie limit.
- Added geometry-only Rayleigh/Wood anomaly diagnostics for rectangular
  periodic cells, including threshold enumeration and safe period-scale
  suggestions.
- Added separable MLFMM directional-transform helpers as the NumPy reference
  path for lower-memory sampled-basis evaluation.
- Added opt-in CuPy MLFMM stream diagnostics through `MLFMMOptions` and the
  pairwise-vs-MLFMM scaling benchmark.

### Fixed
- Fixed native CuPy BiCGSTAB device-vector lifetime so stale per-iteration
  temporaries are released before the next matrix-free operator application,
  avoiding avoidable OOMs for memory-tight streamed MLFMM solves.
- Fixed CuPy MLFMM streamed outgoing construction so recursive child outgoing
  chunks share the same in-flight memory budget as their live ancestor outgoing
  arrays.
- Fixed the automatic multilevel MLFMM sampled-level policy so it no longer
  skips coarse same-level far interactions when the resolved hierarchy grows
  beyond depth 3.
- Fixed CuPy MLFMM streamed traversal so recursive child-frontier buffers share
  one in-flight memory budget instead of each level treating the frontier budget
  as fresh headroom.
- Fixed CuPy MLFMM streamed budget selection so reusable CuPy memory-pool
  blocks count as available memory, avoiding severe over-fragmentation of
  deep sampled-far traversals after large prepare/apply phases.
- Reduced CuPy MLFMM directional-transform upload and resident memory by using
  separable alpha/beta factors instead of dense sampled-direction matrices, so
  larger multilevel runs avoid avoidable out-of-memory failures.

### Changed
- Parallelized CuPy pairwise angular-table setup across each destination block,
  reducing serial phase and associated-Legendre work before mode contraction.
- Accelerated CuPy periodic Ewald evaluation by caching the invariant nodes and
  exponential weights of the Faddeeva quadrature on device.
- Reduced CuPy shifted-reciprocal Ewald contention by warp-reducing structural
  contributions before shared-memory atomic accumulation, accelerating
  cache-off matvecs, cache construction, and periodic interior near fields.
- Accelerated CuPy periodic Ewald evaluation by caching reciprocal-term
  azimuths, powers, square-root branches, and combinatorial factors on device,
  removing repeated pair-local setup from cache-off matvecs, cache population,
  and periodic interior near-field evaluation.
- Accelerated MLFMM directional-basis preparation by reusing Legendre tables
  across azimuthal samples on each polar ring.
- Fused CuPy MLFMM on-the-fly leaf aggregation so sampled-far applies avoid
  materializing aggregate-side dense leaf translation blocks.
- Reduced CUDA arithmetic overhead in CuPy pairwise, MLFMM leaf, and periodic
  Ewald kernels by replacing repeated powers and separate phase trigonometry
  with shared or recurrent factors.
- Fused CuPy MLFMM selected leaf receive contractions so streamed applies avoid
  receive-side dense leaf translation scratch.

## [0.4.0] - 2026-05-17

### Added
- Added a CuPy dense direct-solve path using cuSOLVER-backed LU factorization,
  with repeated-RHS reuse through cached GPU LU payloads on the `Simulation`
  instance, mirroring the existing NumPy direct-solve cache pattern.
- Added a native CuPy restarted GMRES implementation that keeps Arnoldi/Givens
  work on device and exposes inner-iteration progress callbacks, with optional
  final true-residual verification through the shared solver controls.
- Added native CuPy FGMRES, LGMRES, and BiCGSTAB iterative solver paths,
  including `solve_linear_system(..., backend="cupy",
  method="fgmres"|"lgmres"|"bicgstab")`, with monitor-channel reporting aligned
  to the native GMRES result contract.
- Added native CuPy block-GMRES support for iterative multi-RHS solves
  (`B.shape == (n, nrhs)`) with:
  - block-aware operator/preconditioner adapters,
  - optional RHS batching and Gram-matrix deflation controls,
  - explicit block/per-RHS residual diagnostics in `LinearSolveResult`.
- Added a native NumPy high-frequency MLFMM coupling backend for sphere-cluster
  workflows:
  - `SimulationConfig(coupling_backend="mlfmm")`
  - exact leaf-near interactions with direct pairwise fallback for shallow
    hierarchies
  - single-level and multilevel directional far coupling with relative-offset
    batching and shared interpolation/transform caches
  - structured resolved-plan metadata on the prepared coupling operator
- Added a CuPy MLFMM repeated-apply backend for the existing CPU-built
  high-frequency MLFMM plan (`operator_backend="cupy", coupling_backend="mlfmm"`):
  - CPU plan/build remains the source of truth
  - exact-near and sampled far repeated applies run on device
  - direct-stage fallback remains the CuPy pairwise coupling path
  - sampled-far MLFMM interactions stay on `complex128` for both NumPy and CuPy
- Added absolute local absorbed-power diagnostics for dipole-source runs:
  `run.power["P_abs_local"]` and `run.power["P_abs_local_particles"]`.
- Added homogeneous rectangular-lattice periodic workflows:
  - `RectangularLattice2D`, `PeriodicSpec`, and `PeriodicOptions`
  - plane-wave `k_parallel` validation and periodic overlap checks
  - NumPy/CuPy periodic Ewald coupling, plus a direct-sum oracle for small
    checks
  - diffraction-order payloads on `SimulationResult.periodic`, including
    reflected/transmitted order amplitudes and `R/T/A` totals
  - periodic near-field slices with exterior Rayleigh-order evaluation and
    in-slab local-SVWF evaluation for homogeneous spheres

### Removed
- The built-in regular-grid block preconditioner from the high-level
  simulation API. The generic low-level linear-solver `preconditioner=...`
  callable hook remains available for custom experiments.

### Changed
- Source extensibility now includes a `LocalExpansionSource` capability for
  local outgoing-SVWF emitters, and the far-field / initial-field local-emitter
  paths now use that protocol instead of hard-coded dipole-source class checks.
- Dense/direct NumPy solves now use fast pairwise dense assembly when available
  and otherwise fall back to generic dense assembly through repeated
  matrix-free applies.
- Periodic dense direct validation now assembles `A = I - T W` directly from
  cached periodic Ewald blocks for diagonal particle-local `T` operators,
  avoiding column-by-column matrix-free assembly on sphere-like systems.
- Periodic solve logging now reports analytic/local source projection when no
  angular source quadrature is used, and periodic postprocessing reports the
  resolved diffraction-order count and `R/T/A` totals.
- CuPy periodic cache-off matvecs now batch source particles by the temporary
  structural-table memory budget instead of a fixed source-count cap.
- `solve_linear_system(..., backend="cupy", method="gmres")` now routes through
  pyceles' native CuPy GMRES path (instead of delegating to CuPy built-in
  GMRES), improving convergence observability within restart cycles.
- CuPy GMRES now routes 2D RHS inputs to the native block-GMRES iterative path
  (single-RHS behavior remains on the existing native GMRES path).
- Prepared CuPy direct-operator/preconditioner paths now accept true 2D RHS
  inputs directly (`(n, nrhs)`) and expose adapter-use diagnostics in block
  solve metadata when legacy 1D callables are wrapped column-wise.
- Native CuPy block-GMRES now batches block-Arnoldi orthogonalization and
  correction assembly through packed GEMM-style updates, substantially reducing
  dense-kernel overhead on measured dense multi-RHS workloads.
- CuPy MLFMM exact-near coupling now uses a memory-aware device-evaluated path
  (directed near-pair indices + compact translation tables) instead of
  pre-uploaded dense near block tensors, substantially reducing large-case GPU
  prepared-data footprint.
- CuPy MLFMM exact-near apply now uses direct complex input/output in the
  device kernel, removing per-iteration real/imag marshaling and reassembly on
  the host-side orchestration path.
- CuPy direct pairwise coupling now uses direct complex input/output in its
  fused RawKernel apply path, removing per-call real/imag split-repack and
  RHS-major transpose marshaling in the Python launch path.
- CuPy MLFMM grouped far-offset accumulation now uses weighted device kernels
  in single-level and multilevel sampled far passes, reducing intermediate
  tensor traffic in repeated applies.
- CuPy MLFMM grouped far/transfer batches now enforce source/destination
  uniqueness at upload time and fail fast on violations, while keeping
  non-atomic accumulation in the repeated-apply path.
- CuPy MLFMM directional forward/inverse transforms now use packed batched-GEMM
  forms with reflection pre-folded into uploaded operators, replacing the
  previous multi-einsum runtime path.
- CuPy MLFMM multilevel transfer stages now use fused map+phase+accumulate
  kernels for packed-stencil and sparse transfer maps, avoiding intermediate
  shifted/mapped directional tensors in repeated applies.
- CuPy MLFMM sampled-far apply now reuses per-RHS workspaces (outgoing/incoming
  hierarchy buffers plus leaf/incoming-box intermediates) across iterations to
  reduce allocation churn in iterative solves.
- MLFMM precision policy now accepts `compute_dtype="complex64"` on both NumPy
  and CuPy backends; this request is applied to exact-near interactions while
  sampled-far interactions remain `complex128`.
- CuPy MLFMM prepared-cache serialization now stores compact host payloads and
  rebuilds device prepared data on load, reducing cache footprint and avoiding
  device-graph pickling.
- CuPy MLFMM exact-near cache/runtime payload now uses directed leaf-pair
  schedules plus leaf particle tables instead of expanded directed particle-pair
  lists, improving memory scalability on large clusters.
- Prepared NumPy MLFMM operators now expose canonical plan, hierarchy, and
  memory diagnostics for backend-comparable stage and storage reporting.
- NumPy MLFMM repeated apply now uses grouped dense leaf operators as the
  default CPU path, while keeping a low-persistent-memory on-the-fly leaf mode
  available for reference/debug runs.

### Fixed
- Native CuPy GMRES / FGMRES / LGMRES now honor the shared
  `compute_final_residual` policy consistently with robust defaults:
  `compute_final_residual=True` performs true-residual checks at restart
  boundaries, while `False` disables true-residual verification for
  profiling-focused runs.
- Native CuPy block-GMRES now enforces per-RHS true-residual tolerance checks
  with strict per-column acceptance and now performs an in-cycle true-residual
  gate when the block proxy first reaches target, avoiding restart-boundary
  overshoot on large restart values.
- CuPy MLFMM exact-near now honors `compute_dtype="complex64"` in the device
  kernel and lookup-table uploads (instead of only casting the final near
  output), restoring the intended mixed-precision speedup while keeping
  sampled-far interactions on `complex128`.
- Plane-wave cross sections now report physical local dissipation by default:
  `cross_sections["C_abs"]` is now `C_abs_local`, while
  `C_abs_raw_diff = C_ext_raw - C_sca_raw` and `Delta_closure` remain exposed
  as explicit numerical diagnostics.
- Low-level `plane_wave_cross_sections(...)` semantics are now strict:
  callers must provide `local_absorption` explicitly, while legacy
  raw-difference behavior requires explicit `allow_raw_diff_fallback=True`.
- Finite-power diagnostics now expose explicit local/closure terms alongside
  existing `T/R` outputs:
  `P_abs_raw_diff`, `A_raw_diff`, `P_abs_local`, `A_local`, and
  `Delta_power_closure`, plus per-particle local vectors
  (`P_abs_local_particles`, `A_local_particles`).
- Finite-power local absorbed-power normalization now uses the same
  medium-dependent incident-intensity convention as the cross-section path,
  restoring expected `P_abs_raw_diff -> P_abs_local` quadrature convergence
  (including `n_medium != 1` cases).
- CuPy MLFMM repeated apply now memoizes grouped leaf receive adjoints per
  runtime object, removing repeated grouped `swapaxes(...).conj()` work in the
  hot loop.
- Multilevel MLFMM directional interpolation now follows the validated angular
  ordering and transfer conventions, avoiding severe GMRES stagnation on
  benchmark-scale multilevel runs.
- Rectangular interior translation LUT reuse is now cached correctly, removing
  repeated rebuilds during MLFMM preparation.
- Host-side directional MLFMM transforms now use the canonical normalized
  Legendre recurrence, avoiding non-finite values in shallow/high-order plans.

## [0.3.0] - 2026-03-12

### Added
- Added a CuPy backend for diagonal sphere/layered-sphere clusters:
  - `SimulationConfig(operator_backend="cupy")`
  - CuPy GMRES solver wrapper
  - diagonal-only GPU single-body `T` path
  - fused direct pairwise `W·x` RawKernel backend for `complex64` and `complex128`
    with GPU-resident translation precompute tables.
- Added inherited postprocessing backend selection via
  `SimulationConfig(postprocessing_backend="inherit" | "numpy" | "cupy")`.
- Added a first CuPy postprocessing slice for scattered far-field SVWF-to-PWP
  assembly while keeping the public far-field payloads NumPy-shaped.
- Added CuPy near-field postprocessing for the scattered field, the dominant
  Gaussian/general initial-field paths, and homogeneous-sphere internal fields,
  while keeping mixed non-spherical internal-field cases on the reference
  implementation.
- Added a reusable source-compliance test helper and contract tests to ensure
  built-in/new source classes expose required `Source` protocol methods and
  capability metadata.
- Added `BesselBeam` as an exact non-paraxial cone-ring angular-spectrum source
  with OAM order (`order_m`), axis tilt control
  (`polar_angle`/`azimuthal_angle`), and TE/TM Jones compatibility.
- Added `SLMSource` (angular-spectrum wrapper source) for complex
  phase/amplitude modulation of propagating TE/TM beams on `(alpha, beta)`
  grids.
- Added Maxwellian Laguerre-Gaussian source families:
  - `LaguerreGaussianBeam` (collimated exact angular-spectrum LG),
  - `FocusedLaguerreGaussianBeam` (Debye/aplanatic finite-NA focused LG).
- Added canonical SVWF/PVWF conversion helpers:
  `pwp_to_svwf_regular`, `angular_spectrum_to_svwf_regular`,
  `svwf_regular_to_pwp`, and `svwf_outgoing_to_pwp`.
- Local electric-source models:
  - `DipoleSource` for one point electric dipole
  - `DipoleCollection` for multiple dipoles
- Dipole excitation now projects to particle RHS coefficients through outgoing
  `l=1` SVWF coefficients translated to each sphere center (SMUTHI-aligned
  normalization, reusing pyceles translation kernels).
- Near-field initial-field evaluation now supports dipole sources directly and
  masks exact dipole-center samples as `NaN` to avoid singular-point artifacts.
- Dipole source convenience helpers:
  - Cartesian orientation triplet generation via
    `DipoleSource.cartesian_basis_sources(...)`
  - homogeneous-background dissipated-power helpers for single dipoles and
    dipole collections.
- Added dipole power/LDOS helpers:
  - `compute_dipole_power_ldos(...)`
  - `compute_dipole_ldos_enhancement(...)`
  evaluating particle-scattered fields at dipole positions without direct
  self-field sampling at `r=0`.
- Added layered-sphere support in the spherical-particle backend:
  - multilayer Mie/T-entry helpers (`layered_mie_ab`,
    `layered_sphere_T_diagonal`, `layered_internal_ab_ratios`),
  - mixed particle descriptor geometry via `Simulation(config, particles=[...])`,
  - mixed `Sphere` + `LayeredSphere` operator preparation path.
- Added `n_particles` convenience properties on `Simulation` and
  `SimulationResult` for explicit particle counts independent of geometry input
  style.
- Added spherical-basis homogeneous spheroid support:
  - dense particle-local `Spheroid` `T` blocks in CELES ordering,
  - SVWF rotation of aligned spheroid `T` blocks into the lab frame,
  - spheroid internal-field evaluation in particle-dispatch near-field workflows.
- Added isolated-spheroid regression coverage against local SMUTHI-generated
  cross-section oracles.
- Added sphere regression coverage against `miepython` for differential
  scattering and selected exterior total-field samples.
- Added explicit cache-clearing helpers for interactive workflows:
  - `pyceles.core.clear_caches()`
  - `pyceles.postprocessing.nearfield.clear_caches()`
  so long-lived sessions can release process-global precompute tables on demand.

### Changed
- `solve_linear_system(..., backend="cupy")` uses GMRES for the GPU iterative
  solve path; direct solves remain NumPy-only.
- `operator_backend="cupy"` uses the fused direct raw-kernel coupling path for
  GPU pairwise matvecs.
- `examples/profile_pyceles_phases.py` now accepts
  `--operator-backend {numpy,cupy}` and
  `--postprocessing-backend {inherit,numpy,cupy}`, and synchronizes GPU work
  when timing the solver phase so CuPy wall times are meaningful.
- Source capability contract now includes
  `Source.has_finite_incident_power()`, used as the canonical policy gate for
  finite-beam-only diagnostics across simulation and far-field workflows.
- Infinite-power diagnostics restrictions are now centralized through
  shared source-policy helpers instead of duplicated per-module checks.
- `SimulationConfig.azimuthal_angles` default now uses periodic sampling on
  `[0, 2*pi)` (`endpoint=False`) to keep periodic azimuth fast paths active by
  default.
- `SimulationConfig` shared default angular grids now use
  `uniform_polar_grid(1801)` and `uniform_periodic_azimuth_grid(360)`, which
  are lighter than the previous defaults while still signaling the distinct
  polar/azimuth quadrature roles through their counts.
- Angular-grid setup now uses canonical helpers:
  `core.uniform_polar_grid(n)` and
  `core.uniform_periodic_azimuth_grid(n)`.
- Public naming now uses `k0` for vacuum wavenumber (`2*pi/lambda`) in
  simulation/far-field APIs (instead of `omega`) to reduce ambiguity with true
  angular frequency in future dipole/LDOS workflows.
- Dipole far-field outputs now include direct (`initial`) and coherent total
  (`initial + scattered`) PWPs in addition to particle-scattered PWPs.
- `SimulationResult.polarization_jones` is now reserved for propagating TE/TM
  sources and set to `None` for local dipole sources to avoid misleading
  placeholder metadata.
- Total scattering cross-section integration now enforces periodic azimuth
  closure on endpoint-excluded `alpha in [0, 2*pi)` grids, matching the
  physically correct composite trapezoid rule on the circle and sphere/Mie
  benchmarks.
- Near-field internal-field replacement now treats exactly index-matched
  particles as transparent so the total field remains equal to the incident
  field everywhere in that case.
- Breaking: introduced explicit phase split for multi-source workflows:
  - `Simulation.solve_sources(...)` is now the canonical solve-only API,
  - `Simulation.postprocess_sources(...)` performs optional channel postprocessing
    (far field, power, cross sections),
  - `Simulation.run_multi_sources(...)` was removed.
- Breaking: `Simulation.solve_sources(...)` now accepts only
  mapping-style channel input (`{label: source}`); sequence + `labels=...`
  call patterns were removed to keep one explicit multi-source API surface.
- `solve_polarization_basis=True` now acts as a convenience `run()` wrapper
  built on top of `solve_sources(...)` + `postprocess_sources(...)` (TE/TM channels + Jones mixed
  recombination + unpolarized diagnostics).
- Source handling across solve/far-field/plotting workflows now follows source
  capabilities (`Source`/`AngularSpectrumSource` + Jones metadata) instead of
  hard-coded concrete-class checks, so wrapper propagating sources (for example
  `SLMSource`) participate in the same TE/TM basis workflows.
- Near-field internal-field evaluation is now particle-dispatch aware through
  the canonical `compute_internal_field(...)` entry point, with explicit support
  for mixed `Sphere` and `LayeredSphere` lists while keeping the homogeneous
  sphere kernel as the canonical implementation path.
- Breaking: removed `Simulation.from_particles(...)`; particle-descriptor
  construction now goes through `Simulation(config, particles=[...])` only.
- Breaking: removed legacy `Simulation(...)` array-geometry kwargs
  (`positions`, `radii`, `n_particle`); use explicit particle descriptors
  (`particles=[...]`) in all workflows.
- Breaking: removed `SimulationResult.n_particle`; exact particle definitions
  are exposed via `SimulationResult.particles`.
- Breaking: `core.matvec.prepare_matvec(...)` and
  `core.matvec.precompute_T_diagonal(...)` now use particle descriptors as the
  canonical API (`particles=[...]`), and thin `*_from_particles` aliases were
  removed.
- Breaking: prepared-operator ownership now lives under
  `pyceles.core.operators`; the old monolithic `core/matvec.py` module was
  split into package modules (`base`, `groups`, `single_body`,
  `coupling_pairwise`, `coupling_dense`, `prepare`) with no compatibility
  facade kept.
- Added canonical particle-construction helpers
  (`spheres_from_arrays`, `layered_spheres_from_arrays`,
  `spheroids_from_arrays`).
- Breaking: renamed previous `Ellipsoid` placeholder references to
  `Spheroid` across code/docs/tests (no compatibility aliases kept).
- Breaking: HDF5 geometry snapshots no longer persist compatibility
  `geometry/n_particle`; geometry payloads are now strictly particle-native
  (`geometry/particles`) with no persisted generic `positions`/`radii` tables.
- Breaking: near-field internal-field APIs are now particle-native:
  `compute_internal_field(...)`, `compute_near_field_components(...)`, and
  `compute_total_field(...)` now take geometry from `particles=[...]` only.
  Legacy array-style geometry inputs are removed from these paths.
- Breaking: `compute_near_field_slice(...)` now uses only
  `axis_0_min/axis_0_max/axis_1_min/axis_1_max` bounds; legacy
  `x_min/x_max/z_min/z_max` aliases were removed.
- Breaking: the near-field public package surface is now
  `pyceles.postprocessing.nearfield`; the temporary
  `pyceles.postprocessing.workflows` import path was removed during the
  near-field package split.
- Breaking: the far-field public package surface is now
  `pyceles.postprocessing.farfield`; the old monolithic
  `postprocessing/farfield.py` implementation was split into package modules
  (`patterns`, `power`, `cross_sections`, `common`) with `__init__.py` as the
  public entry point.
- Breaking: near-field plotting overlays are now particle-native:
  `plot_spheres(...)`, `plot_nearfield_panels(...)`,
  `plot_nearfield_panels_channels(...)`, and
  `plot_nearfield_poynting_overlay(...)` consume `particles=[...]` geometry
  instead of parallel `positions`/`radii` inputs.
- Internal simulation workflow ownership now lives under the
  `pyceles.simulation` package (`config`, `results`, `solve`, `postprocess`,
  `workflow`) instead of one monolithic `simulation.py` owner, while keeping
  the public `from pyceles.simulation import ...` package surface stable.
- `SimulationResult` now stores only canonical `particles`; convenience
  `positions`/`circumscribing_radii` views are derived properties rather than
  duplicated stored payloads.
- Breaking: `io.load_geometry_h5(...)` now returns only canonical
  `particles` + metadata attrs; the derived `positions` convenience payload was
  removed to keep one geometry representation in loaded snapshots.
- Breaking: `Simulation` / `SimulationResult` now expose
  `circumscribing_radii` as the canonical per-particle radius view; ambiguous
  `radii` shorthand was removed.
- Breaking: `DipoleSource.cartesian_basis_sources(...)` now returns fixed
  labels (`px`, `py`, `pz`); custom label overrides were removed so basis
  naming is consistent with other fixed-channel interfaces.
- README/examples now present source-only runs as `particles=[]` and no longer
  document array-geometry constructor patterns.
- Direct-solver repeated solves on the same `Simulation` instance now reuse a
  cached LU factorization of the dense operator in addition to reusing the
  dense matrix assembly.
- Added `solver_compute_final_residual` control (config + per-call override in
  `solve_sources(...)`) so repeated direct multi-source workflows can skip
  final residual diagnostics when speed is preferred.
- Direct solver runs now print an explicit solve-phase status line (setup mode,
  RHS count, residual-check mode) so verbose output no longer jumps from dense
  assembly directly to postprocessing logs.
- `examples/minimal_pyceles_demo.py` now includes a dipole-collection run on
  the same 4-particle geometry and uses autoscaled near-field panel limits for
  that dipole case.
- `examples/minimal_pyceles_demo.py` now also computes a coarse dipole LDOS
  enhancement map (`px`, `py`, `pz`, averaged) on the same y-slice.

### Fixed
- Dtype parsing for `compute_dtype`/`accum_dtype` now accepts generic NumPy
  dtype-like inputs (for example `np.complex64`, `np.dtype("complex64")`) in
  simulation and near-field workflows.
- Simulation config validation now warns when azimuthal grids include both
  `0` and `2*pi`, since this duplicated periodic endpoint usually wastes work
  and may disable periodic fast-path detection.
- IO/postprocessing channel helper guidance now consistently covers both
  `solve_polarization_basis=True` runs and per-channel
  `postprocess_sources(solve_sources(...))` results (near-field channel selection and
  far-field intensity convenience helper).
- `solve_polarization_basis=True` remains undefined for local dipole sources,
  and is now validated by TE/TM-source capabilities instead of concrete type
  names.
- Conservative LUT-radius helper logic is now centralized in
  `core.geometry_bounds`, removing duplicate cross-set bound implementations.
- Dipole-source documentation/error messages now explicitly note that the
  current real-`n_medium` requirement is a legacy beam-era solver policy, not
  a fundamental local-source physics limitation.
- README and source docstrings now document practical dipole-moment magnitude
  scaling (`|p| ~ k0^-3`) for readable near-field amplitudes in example units.
- Dipole-source validation now warns when dipole centers are placed inside
  particle circumscribing spheres (currently untested in the homogeneous-host
  dipole formulation and potentially unreliable).

## [0.2.0] - 2026-02-28

### Changed
- Breaking: renamed finite-beam power diagnostics helpers in `postprocessing.farfield`:
  - `initial_power_wavebundle_normal_incidence` -> `incident_power_from_pwp`
  - `transmitted_reflected_power` -> `finite_beam_power_fractions`
- Breaking: plane-wave cross-section semantics now follow SMUTHI-style cluster
  scattering:
  - `C_sca` in `plane_wave_cross_sections` is now far-field integrated,
  - `scattered_pwp_te` and `scattered_pwp_tm` are now required inputs,
  - redundant `C_sca_farfield` output key was removed,
  - `total_scattering_cross_section_from_coefficients` was removed from the
    public far-field API.
- `absorption_cross_section` no longer has a coefficient-only fallback path and
  now requires scattered TE/TM PWPs for SMUTHI-style cluster absorption
  (`C_abs = C_ext - C_sca`).
- Finite-beam transmitted/reflected fractions are now normalized from the
  provided initial TE/TM PWP (solid-angle integration) instead of a
  normal-incidence Gaussian closed form, making tilted-beam diagnostics
  physically consistent with the computed source spectrum.
- Package import is now silent by default; the ASCII logo is shown once per process at the start of verbose simulation runs.
- HDF5 far-field payloads are now stored in compact form (`alpha`, `beta`, `coeff`) without persisting redundant `kx/ky/kz` arrays.
- Simulation workflow diagnostics now persist source-vs-farfield angular-grid metadata in a compact form (shared grid stored once when equal).
- `examples/minimal_pyceles_demo.py` now defaults to mixed precision (`compute_dtype=complex64`, `accum_dtype=complex128`) to better reflect recommended usage.
- Plotting labels/titles were improved with math-style formatting for near-field components and far-field hemisphere captions.
- `solve_polarization_basis=True` workflow is explicitly documented as TE/TM-only solve plus Jones-channel recombination (no redundant third mixed solve).
- Core/postprocessing architecture was reorganized into responsibility-based modules:
  - source models and source-side helpers in `core.sources`,
  - SVWF projection kernels in `core.projection`,
  - near-field implementation under the `postprocessing.nearfield` package
    (`classification`, `components`, `slice`, `workflows`, and kernel modules),
  while keeping `core.fields` as the remaining public compatibility facade.
- LUT max-radius inference is now conservative and cheap:
  - translation `RadialLUT` sizing switched from pairwise `O(N^2)` distance scan to `O(N)` geometry bound,
  - near-field radial LUT sizing switched from exact `O(N*M)` sphere-point scan to an `O(N+M)` conservative bound.

### Fixed
- Translation `RadialLUT` now guards against the spherical-Hankel singularity at `kr=0`, preventing NaN/inf contamination for small `r`.
- Near-field scattered-field evaluation now skips interior-particle points and sanitizes inside scattered components, avoiding divergence artifacts at sphere centers and preventing overflow-related warnings.
- Near-field general initial-field integration now uses periodic azimuth weights on
  `endpoint=False` angular grids, consistent with RHS source projection.

## [0.1.0] - 2026-02-25

### Added
- Initial public Python package scaffold (`src/` layout, packaging metadata, license, examples, notebook).
- CELES/SMUTHI-compatible core SVWF indexing and translation/T-matrix building blocks.
- Matrix-free many-sphere solve path for `(I - T W) x = T b` with SciPy iterative/direct solver support.
- Source models for `PlaneWave` and `GaussianBeam`, including Jones-style TE/TM polarization inputs.
- Near-field decomposition APIs returning `initial`, `scattered`, `internal`, and `total` components.
- Far-field APIs for scattered/initial/total plane-wave patterns, power diagnostics, and plane-wave cross sections.
- Built-in regular-grid block preconditioner option (`grid_block`) and warm-start/preconditioner solver hooks.
- Precision policy with explicit `compute_dtype` and `accum_dtype` plumbing.
- User-facing geometry validity check for circumscribing-sphere overlap.
- Source-only (no scatterers) simulation support via explicit empty geometry arrays.
- Benchmark/profiling workflows and CELES-main replication notebook.
- Development tooling baseline with `ruff`, `mypy`, `pre-commit`, and notebook output stripping (`nbstripout`).

### Changed
- Angular-grid API now preserves a conceptual split between source-projection quadrature and far-field output bins, while keeping one shared CELES-style default grid.
