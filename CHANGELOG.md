# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Added `examples/run_smuthi_dipole_diagnostic.py` for SMUTHI-vs-pyceles dipole
  and dipole-collection cross-checks with JSON reference-value output.

### Changed
- `SimulationConfig.azimuthal_angles` default now uses periodic sampling on
  `[0, 2*pi)` (`endpoint=False`) to keep periodic azimuth fast paths active by
  default.
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
- Added `Simulation.run_multi_sources(...)` as the general multi-channel API:
  any labeled source set now runs through one shared-operator multi-RHS solve.
- `solve_polarization_basis=True` now acts as a convenience `run()` wrapper
  built on top of `run_multi_sources(...)` (TE/TM channels + Jones mixed
  recombination + unpolarized diagnostics).

### Fixed
- Dtype parsing for `compute_dtype`/`accum_dtype` now accepts generic NumPy
  dtype-like inputs (for example `np.complex64`, `np.dtype("complex64")`) in
  simulation and near-field workflows.
- Simulation config validation now warns when azimuthal grids include both
  `0` and `2*pi`, since this duplicated periodic endpoint usually wastes work
  and may disable periodic fast-path detection.
- IO/postprocessing channel helper guidance now consistently covers both
  `solve_polarization_basis=True` runs and per-channel
  `run_multi_sources(...)` results (near-field channel selection and
  far-field intensity convenience helper).
- `solve_polarization_basis=True` is now explicitly restricted to propagating
  TE/TM sources (`PlaneWave`/`GaussianBeam`), and raises for dipole sources.
- Conservative LUT-radius helper logic is now centralized in
  `core.geometry_bounds`, removing duplicate cross-set bound implementations.
- Dipole-source documentation/error messages now explicitly note that the
  current real-`n_medium` requirement is a legacy beam-era solver policy, not
  a fundamental local-source physics limitation.

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
  - near-field kernels in `postprocessing.nearfield_kernels`,
  - near-field orchestration in `postprocessing.nearfield_workflows`,
  while keeping `core.fields` and `postprocessing.nearfield` as public API facades.
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
