# pyceles

**pyceles** is a Python reimplementation of the MATLAB **CELES** package for the electromagnetic
simulation of large ensemble of particles using the T-matrix method.
The code starts as a pure NumPy + SciPy reference implementation, with accelerator backends planned.

This repository focuses on:
- Correctness first (CELES conventions, reproducible notebooks/tests)
- A clean NumPy + SciPy reference implementation
- Performance via vectorization + CELES-style caching/LUTs (no MEX build)
- A future optional GPU backend (CuPy, or possibly Numba and/or PETSc) without duplicating code paths
- A CELES-style precision policy (`compute_dtype`, `accum_dtype`) for CPU/GPU portability

## Acknowledgment

This project would not be possible without the frameworks established by [CELES](https://github.com/disordered-photonics/celes) and [SMUTHI](https://gitlab.com/AmosEgel/smuthi), and the work of their authors and contributors.
Users of pyceles are referred to the publications listed in the CELES and SMUTHI repositories, and those papers can be cited when pyceles is used in scientific work.

## What works today

- CELES-compatible VSWF indexing (tau blocks contiguous, m=-l..l)
- Wigner-3j implementation (cached)
- CELES translation coefficients (a5/b5 table) + normalized Legendre recurrences
- CELES wavebundle Gaussian incident field:
  - optimized normal-incidence kernels (ported from CELES)
  - tilted incidence via angular-spectrum projection
  - near-field tilted-beam acceleration via hidden rotated-frame evaluation
    (with general alpha-beta fallback via `force_general_initial_field=True`)
- CELES plane-wave incident coefficients
- O(N^2) reference matvec for (I - T W) x = T b
- SciPy GMRES wrapper with tqdm progress
- Near-field evaluation with CELES formulas:
  - scattered field
  - initial field (plane-wave-pattern integral)
  - total field with internal-field replacement inside spheres (CELES behavior)
  - canonical helper returning `initial/scattered/internal/total` components in one call
  - geometry-agnostic helper (`compute_near_field`) plus planar convenience wrapper (`compute_near_field_slice`)
- Far-field plane-wave pattern + forward/backward power flux (CELES formulas)
  - canonical helper returning `initial/scattered/total` PWPs in one call
  - plane-wave cross sections:
    - differential scattering cross section (`dC_sca/dOmega`)
    - total scattering cross section (PWP-integrated, SMUTHI-style cluster definition)
    - extinction cross section (coefficient-based)
    - absorption cross section (`C_abs = C_ext - C_sca`)
    - no coefficient-only cluster `C_sca` helper is exposed
- Physical source checks:
  - finite-beam power fractions are normalized by integrating the initial TE/TM
    plane-wave spectrum (works for normal and tilted Gaussian beams)
  - plane-wave excitation raises when requesting power fractions
- Geometry sanity check:
  - by default, `Simulation` enforces disjoint particle circumscribing spheres
    (required by T-matrix superposition), can be disabled via `check_circumscribing_sphere_overlap=False`
- Explicit no-scatterer (source-only) simulations:
  - pass explicit empty geometry arrays to run a beam-only simulation
    (`positions=np.zeros((0,3))`, `radii=np.zeros((0,))`, `n_particle=np.zeros((0,), complex)`)
  - `None` geometry inputs are rejected to avoid accidental empty runs
- Channel-aware polarization workflow:
  - source polarization accepts CELES-style `"TE"`, `"TM"` or Jones weights `(a_te, a_tm)`
  - optional dual-basis solve (`solve_polarization_basis=True`) computes both
    pure TE/TM channels in one simulation run (could be faster with block-Krylov solvers in PETSc)
  - mixed outputs are combined from Jones weights
  - optional unpolarized diagnostics are provided when basis channels are available
- Solver API extensions:
  - matrix-matrix solves (`A @ X`) with multiple RHS columns
  - warm start vectors/matrices (`solver_warm_start`)
  - preconditioner hook (`solver_preconditioner`)
  - built-in CELES-style regular-grid block-diagonal preconditioner
    (`solver_preconditioner_kind='grid_block'`)
- Angular-grid API:
  - one shared CELES-style default grid (`polar_angles`, `azimuthal_angles`)
  - optional split grids for source projection and far-field outputs
    (`source_*`, `farfield_*`)

## Angular-grid policy (shared defaults, split API)

pyceles keeps a single shared default angular grid to stay close to CELES usage
and avoid unnecessary complexity in standard workflows.

At the same time, the API keeps source and far-field grids distinct in principle:
- `source_polar_angles`, `source_azimuthal_angles`:
  quadrature nodes used to project the source onto SVWF coefficients (RHS).
- `farfield_polar_angles`, `farfield_azimuthal_angles`:
  output bins used for scattered/initial/total far-field PWPs.

Physics/numerics implications:
- The linear operator `(I - T W)` is independent of these angular grids.
- For finite-width beams (Gaussian/angular-spectrum sources), source-grid quality
  controls RHS fidelity. A finer far-field grid cannot recover source angular
  content that was undersampled during source projection.
- Plane-wave source projection is analytic, so this source-grid issue does not
  apply to plane-wave excitation.

Relation to CELES and SMUTHI:
- CELES typically uses one shared numerics grid for both source-related
  quadratures and far-field postprocessing.
- SMUTHI exposes more independent angular controls (especially natural in layered
  media workflows). pyceles does not currently target planar interfaces, but keeps
  this separation available for future flexibility and expert use.

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

## Install

From the repo root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

## Run the notebook replicating the original CELES_MAIN.m script

Open:

- `notebooks/01_celes_main_replication.ipynb`

The notebook resolves the repository root automatically and can run from a source checkout
without editable install, as long as runtime dependencies are available in the active environment.

The notebook reproduces the CELES_MAIN.m workflow:
1) load original 500-particle `examples/sphere_parameters.txt` from CELES
2) solve with GMRES (with progress)
3) print transmitted/reflected power
4) plot the near-field cross-cut (8 subplots) + Poynting vector
5) plot the far-field in both hemispheres

The notebook also exposes precision knobs (CELES used single precision for maximum performance):
- `compute_dtype`: hot-path compute dtype (`complex64` or `complex128`)
- `accum_dtype`: accumulation/reduction dtype (at least as precise as compute)

## pyceles-vs-MSTM benchmark script

Use:
- `examples/run_mstm_pyceles_cluster_benchmark.py`

Purpose:
- run the same tilted Gaussian-beam cluster setup in pyceles and MSTM v4.0,
- generate near-field TE/TM component maps, far-field hemisphere maps from MSTM `scattering_map_model=1`, and semilogy `S11(theta)` curves from MSTM `scattering_map_model=0`,
- keep pyceles `S11` outputs in their native normalization and show MSTM curves/maps as fixed-factor rescaled overlays,
- write comparison metrics (RMSE, relative RMSE, Pearson correlation) to a JSON summary.

Requirements:
- you must have a compiled MSTM 4.0 executable available locally,
- pass its path with `--mstm-exe`.

Example:

```bash
python examples/run_mstm_pyceles_cluster_benchmark.py \
  --mstm-exe <path-to-mstm-executable> \
  --output-prefix mstm_pyceles_500_tilted_l4_dense \
  --n-particles 500 \
  --polar-angle 0.43 \
  --azimuthal-angle 0.37 \
  --lmax 4 \
  --epsilon 1e-6 \
  --py-solver-method gmres \
  --n-beta 1801 \
  --n-alpha 720 \
  --nf-min -4000 0 -3000 \
  --nf-max 4000 0 5000 \
  --nf-step 25 \
  --mstm-scattering-map-dimension 181
```

Outputs are written in:
- `outputs/mstm` (MSTM input/output files),
- `outputs/mstm_diagnostics` (plots and `<prefix>_summary.json`).

## Notes on performance

CELES is fast because it:
- runs translation/matvec on GPU
- reuses cached translation tables + radial Hankel LUTs
- uses a block-diagonal preconditioner

pyceles currently matches the equations/conventions, but the matvec is still a
straight O(N^2) NumPy implementation. Next performance milestones will focus on
- optional CuPy (or Numba, PETSc) backend for the hot paths
- O(N log N) matvec via fast multipole method or FFT-based matvec for grid-snapped particles

## Recent CPU benchmark snapshot

Measured on a laptop with:
- CPU: `13th Gen Intel(R) Core(TM) i7-13850HX`
- Python: `3.12.10`
- NumPy/SciPy: `2.4.2 / 1.17.1`
- script: `examples/profile_pyceles_phases.py`

Common benchmark parameters:
- geometry: `N=500` spheres from `examples/sphere_parameters.txt`, `lmax=3`
- source: Gaussian beam (`wavelength=550`, `n_medium=1.0`, `beam_width=2000`, `TE`, normal incidence)
- angular grids: `n_beta=3601`, `n_alpha=180`
- near-field slice: plane `y=0`, `x=[-4000, 4000]`, `z=[-3000, 5000]`, `dx=40` (201 x 201 points)
- solver: `gmres`, `rtol=1e-4`, `restart=100`, `maxiter=1000`

Reproduce:
```bash
python examples/profile_pyceles_phases.py --n-particles 500 --lmax 3 --dx 40 --compute-dtype complex128 --accum-dtype complex128 --out-dir outputs/profiling_py312_c128a128 --quiet
python examples/profile_pyceles_phases.py --n-particles 500 --lmax 3 --dx 40 --compute-dtype complex64 --accum-dtype complex128 --out-dir outputs/profiling_py312_c64a128 --quiet
```

Phase wall times:
- `complex128/complex128`:
  - Solver (translation block cache OFF): `367.7 s`
  - Solver (translation block cache ON): `30.5 s`
  - Far-field postprocessing: `21.7 s`
  - Near-field postprocessing: `171.0 s`
- `complex64/complex128`:
  - Solver (translation block cache OFF): `314.4 s` (about `-14.5%`)
  - Solver (translation block cache ON): `25.6 s` (about `-16.2%`)
  - Far-field postprocessing: `18.7 s` (about `-14.0%`)
  - Near-field postprocessing: `138.0 s` (about `-19.3%`)
  
## Cumulative Optimization Notes

The current performance is the cumulative result of several tweaks.
Key improvements include:

- Scalar-Legendre translation path (`legendre_normalized_trigon_scalar`) in `translation_block`
  to reduce overhead in the matrix-free pair-block assembly hot path.
- `matmul`-based mode contraction in near-field kernels after benchmarking against
  `einsum` and `tensordot` (similar at very low `lmax`, better scaling for larger `lmax`).
- Translation table/LUT reuse (ab5 cache + radial Hankel LUT + optional exact `W_ij` block cache).
- End-to-end precision policy (`compute_dtype`, `accum_dtype`) enabling CELES-style
  mixed precision on CPU today and portable behavior for a future GPU backend.
- Rotated-frame initial beam near-field initial-field fast path:
  - for regular (non-periodic boundary conditions) workflows, pyceles rotates points to the beam-local frame,
    evaluates the beam initial field with analytic azimuth integration, then rotates
    E/H vectors back to the user frame.
  - the general alpha-beta quadrature path is still kept as fallback for flexible/periodic workflows.

Benchmarked but intentionally *not* kept as defaults:
- full `z*kz` phase precompute for near-field initial evaluation: modest speedup with large memory payload.
- giant all-alpha batch strategies: no consistent end-to-end gains.
- multiprocessing near-field initial evaluation: good speedups but high RAM/process overhead and extra complexity.

Important for CELES users: these gains preserve the matrix-free iterative workflow by default.
The solver does not require assembling/storing a global dense matrix; optional block caching
is an explicit tradeoff for systems where RAM is plentiful.

## Beyond CELES (current pyceles extras)

- Tilted Gaussian beams can use a rotated-frame near-field initial-field fast path
  while keeping output fields in the user frame.
- The general alpha-beta quadrature kernel is still available for periodic/flexible
  workflows and can be forced explicitly (`force_general_initial_field=True`).
- Jones-polarization input and optional dual-basis TE/TM solves.
- Basis-level outputs and unpolarized diagnostics for far-field/cross-section workflows.

## Polarization API (new)

All propagating sources support:
- `polarization="TE"` or `"TM"`
- `polarization=(a_te, a_tm)` with complex Jones-like amplitudes

Example:

```python
source = pcl.PlaneWave(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    polarization=(1.0 + 0j, 1.0j),  # circular-like mixture
    polar_angle=0.3,
    azimuthal_angle=0.1,
)
```

## Dual-Basis Solve + Unpolarized Diagnostics

Set `solve_polarization_basis=True` to compute and store TE/TM basis channels:

```python
cfg = pcl.SimulationConfig(
    source=source,
    solve_polarization_basis=True,
    solver_method="gmres",
)
run = pcl.Simulation(cfg, positions=pos, radii=rad, n_particle=n_part).run()

# Mixed (user-requested Jones state):
mixed_coeffs = run.coeffs

# Basis channels:
te_coeffs = run.coeffs_basis["te"]
tm_coeffs = run.coeffs_basis["tm"]

# Unpolarized far-field diagnostics (incoherent TE/TM average):
print(run.unpolarized)
```

Near-field evaluation can target mixed or basis channels:

```python
nf_mixed = pcl.compute_near_field_slice(run, channel="mixed")
nf_te = pcl.compute_near_field_slice(run, channel="te")
nf_tm = pcl.compute_near_field_slice(run, channel="tm")
```

For far-field plotting, an unpolarized intensity convenience helper is available:

```python
I_u = pcl.io.far_field_intensity_from_result(run, channel="unpolarized")
```

## Source-Only Run (No Scatterers)

For beam inspection/debugging, run a simulation with no particles by passing
explicit empty geometry arrays:

```python
run = pcl.Simulation(
    cfg,
    positions=np.zeros((0, 3), dtype=float),
    radii=np.zeros((0,), dtype=float),
    n_particle=np.zeros((0,), dtype=np.complex128),
).run()
```

Notes:
- this yields zero scattered coefficients/fields and preserves incident-field outputs
- use explicit empty arrays; `None` is intentionally rejected

## Warm Start and Preconditioner Hook

Example:

```python
cfg = pcl.SimulationConfig(
    source=source,
    solver_method="gmres",
    solver_warm_start=x0,              # shape (unknowns,) or (unknowns, nrhs)
    solver_preconditioner=M_inv_mv,    # callable M^{-1}(v)
)
```

Built-in regular-grid block preconditioner:

```python
cfg = pcl.SimulationConfig(
    source=source,
    solver_method="gmres",
    solver_preconditioner_kind="grid_block",
    solver_preconditioner_subdivisions=2,   # or (nx, ny, nz)
    solver_preconditioner_cubic_bbox=True,  # cube (True) or bbox (False)
)
```

Notes:
- `direct_max_n` (or `max_n` in low-level direct solver) limits the matrix size `n`
  of the linear system, not the number of RHS columns.
- For many-sphere systems, `n = N_spheres * n_modes(lmax)`.
- `SimulationConfig.solver_direct_max_n` is the high-level knob passed to the
  low-level solver's `direct_max_n/max_n` guard.
- If `solver_preconditioner` (custom callable) is set, keep
  `solver_preconditioner_kind="none"` to avoid ambiguous configuration.

## HDF5 Output With Basis Channels

`pcl.io.save_simulation_h5(...)` now stores:
- mixed solution/far-field groups (as before)
- optional basis groups when available:
  - `solution_basis/te`, `solution_basis/tm`
  - `far_field_basis/te`, `far_field_basis/tm`
- diagnostics under `diagnostics`:
  - mixed and basis power/cross-sections/decompositions
  - unpolarized diagnostics
  - Jones weights

Loading helpers are available via `pcl.io`:
- `load_geometry_h5`, `load_solution_h5`, `load_far_field_h5`
- `load_near_field_components_h5`, `load_mapping_h5`
- `load_simulation_h5` (workflow-level convenience loader)

## License

MIT
