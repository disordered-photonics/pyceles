# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

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

