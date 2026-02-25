# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Regression tests for:
  - translation `RadialLUT` finiteness near `r=0`,
  - near-field inside-mask behavior for scattered-field components.

### Changed
- Package import is now silent by default; the ASCII logo is shown once per process at the start of verbose simulation runs.
- HDF5 far-field payloads are now stored in compact form (`alpha`, `beta`, `coeff`) without persisting redundant `kx/ky/kz` arrays.
- Simulation workflow diagnostics now persist source-vs-farfield angular-grid metadata in a compact form (shared grid stored once when equal).
- `examples/minimal_pyceles_demo.py` now defaults to mixed precision (`compute_dtype=complex64`, `accum_dtype=complex128`) to better reflect recommended usage.
- Plotting labels/titles were improved with math-style formatting for near-field components and far-field hemisphere captions.

### Fixed
- Translation `RadialLUT` now guards against the spherical-Hankel singularity at `kr=0`, preventing NaN/inf contamination for small `r`.
- Near-field scattered-field evaluation now skips interior-particle points and sanitizes inside scattered components, avoiding divergence artifacts at sphere centers and preventing overflow-related warnings.

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
