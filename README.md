# pyceles

pyceles is a Python reimplementation of the MATLAB CELES package for electromagnetic
simulation of large particle ensembles with the T-matrix method.
The code starts as a pure NumPy + SciPy reference implementation, with accelerator backends planned.

This repository focuses on:
- Correctness first (CELES conventions, reproducible examples/notebooks)
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
  - finite-beam-only diagnostics use source capability
    `has_finite_incident_power()`
  - built-in `PlaneWave` and plane-wave-limit beams (`beam_width=0/inf`) are
    treated as infinite-power excitation and rejected for beam-power fractions
  - ideal infinite-power sources (for example `BesselBeam`) are likewise
    excluded from beam-power fractions by source capability policy
  - finite-beam power fractions are normalized by integrating the initial TE/TM
    plane-wave spectrum (works for normal and tilted Gaussian beams)
- Geometry sanity check:
  - by default, `Simulation` enforces disjoint particle circumscribing spheres
    (required by T-matrix superposition), can be disabled via `check_circumscribing_sphere_overlap=False`
- Explicit no-scatterer (source-only) simulations:
  - pass explicit empty geometry arrays to run a beam-only simulation
    (`positions=np.zeros((0,3))`, `radii=np.zeros((0,))`, `n_particle=np.zeros((0,), complex)`)
  - `None` geometry inputs are rejected to avoid accidental empty runs
- Channel-aware polarization workflow:
  - source polarization accepts CELES-style `"TE"`, `"TM"` or Jones weights `(a_te, a_tm)`
  - `SLMSource(base_source, modulation)` wrapper for angular-spectrum complex modulation
    (phase/amplitude masks on TE/TM plane-wave amplitudes)
  - `BesselBeam` exact non-paraxial cone-ring angular-spectrum source
    (`order_m` OAM phase, TE/TM Jones-compatible)
  - `Simulation.solve_sources(...)` is the canonical solve-only API for any
    labeled source set (shared operator, multi-RHS solve)
  - `Simulation.postprocess_sources(...)` turns solved channels into
    per-channel `SimulationResult` outputs (optionally with far-field diagnostics)
  - optional dual-basis convenience mode (`solve_polarization_basis=True`) still
    provides one mixed+basis+unpolarized `SimulationResult`
  - mixed outputs are combined from Jones weights
  - optional unpolarized diagnostics are provided when basis channels are available
- Local electric dipole sources:
  - `pcl.DipoleSource` and `pcl.DipoleCollection`
  - dipole RHS assembly uses outgoing `l=1` SVWF coefficients translated to each
    sphere center (SMUTHI-style concept, pyceles translation kernels)
  - near-field initial field supports dipoles and masks exact dipole-center
    grid hits as `NaN`
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

## Source capability contract

New source classes should satisfy the internal `Source` protocol in
`pyceles.core.sources` and explicitly implement:
- `incident_coeffs(...)`
- `with_polarization(...)` and `jones_coefficients()` for TE/TM propagating sources
- `has_finite_incident_power()` for finite-power diagnostics policy

Finite-beam-only diagnostics (`finite_beam_power_fractions`,
`pwp_power_decomposition` when source-aware, solver-side T/R reporting) are
enabled only when `has_finite_incident_power()` returns `True`.
This keeps policy centralized and avoids class-name-specific special cases as
new source wrappers/classes are added.

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
# optional: compare no-preconditioner vs grid_block in the same run
python examples/profile_pyceles_phases.py --n-particles 500 --lmax 3 --dx 40 --preconditioner-mode both --preconditioner-subdivisions 2 --cache-mode off --out-dir outputs/profiling_py312_precond_compare --quiet
```

Phase wall times:
- `complex128/complex128`:
  - Solver (translation block cache OFF): `366.5 s`
  - Solver (translation block cache ON): `30.3 s`
  - Far-field postprocessing: `22.2 s`
  - Near-field postprocessing: `171.3 s`
- `complex64/complex128`:
  - Solver (translation block cache OFF): `313.7 s` (about `-14.4%`)
  - Solver (translation block cache ON): `25.6 s` (about `-15.6%`)
  - Far-field postprocessing: `19.7 s` (about `-11.6%`)
  - Near-field postprocessing: `136.3 s` (about `-20.4%`)
- `grid_block` preconditioner (cache OFF, `complex128/complex128`, subdivisions=2):
  - Solver without preconditioner: `362.3 s`
  - Solver with `grid_block`: `241.8 s` (about `-33.3%`, `~1.50x` faster)
  
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

## Local Dipole Sources

Use dipole moments/orientations directly (no TE/TM polarization basis):

```python
source = pcl.DipoleSource(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    position=(0.0, 0.0, 0.0),
    dipole_moment=(1.0 + 0j, 0.0 + 0j, 0.0 + 0j),  # x-oriented
)
```

`dipole_moment` is a complex 3-vector. You can treat it as
`dipole_moment = amplitude * direction`, where `direction` can itself be
complex (relative component phases / elliptical source orientation).

For multiple dipoles in one source channel:

```python
source = pcl.DipoleCollection(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    positions=np.array([[0, 0, 0], [200, 0, 0]], dtype=float),
    dipole_moments=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.complex128),
)
```

Current scope/limits:
- homogeneous medium is still restricted to real `n_medium` in this solver path.
  This is inherited from pyceles' original beam-only (CELES-like) workflow and
  is not a fundamental dipole-physics limitation.
- `solve_polarization_basis=True` is not defined for dipole sources
- dipole homogeneous-background dissipated-power helpers are available:
  - `DipoleSource.dissipated_power_homogeneous_background()`
  - `DipoleCollection.dissipated_power_homogeneous_background()`
  - `DipoleCollection.dissipated_power_homogeneous_background_per_dipole()`
- dipole far-field helper provides direct (`initial`), particle-scattered, and
  coherent total (`initial + scattered`) PWPs
- dipole `SimulationResult` objects do not carry TE/TM Jones metadata
  (`polarization_jones=None`)
- dipole centers are expected in the homogeneous host medium (outside particle
  circumscribing spheres). Interior-embedded dipoles are currently untested and
  may be unreliable in the present solver path.
- dipole moments are interpreted in the same length-unit convention used by
  geometry and wavelength. A practical reference scale is
  `|p| ~ k0^-3 = (wavelength / (2*pi))^3`. For `wavelength=550` (nm units),
  this is about `6.7e5`; values around `1e6` to `1e7` are often convenient in
  examples when you want near-field magnitudes around `O(1)`.

Power/LDOS helpers (single dipole or dipole collections):

```python
res = pcl.compute_dipole_power_ldos(run)
print(res.power_total, res.power_homogeneous, res.enhancement)  # P, P0, P/P0

purcell = pcl.compute_dipole_ldos_enhancement(run)  # scalar for one dipole
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
solved = sim.solve_sources(dip.cartesian_basis_sources())  # px, py, pz
multi = sim.postprocess_sources(solved, include_farfield=False)
```

## Multi-Source Solve (Recommended)

Use `solve_sources(...)` when you need multiple channels (for example TE/TM
basis, dipole x/y/z, or SLM pattern sweeps) on the same geometry.
Then call `postprocess_sources(...)` when you need channel-level far-field or
power diagnostics:

```python
sim = pcl.Simulation(cfg, positions=pos, radii=rad, n_particle=n_part)
solved = sim.solve_sources(
    {
        "te": source.with_polarization("TE"),
        "tm": source.with_polarization("TM"),
    }
)
multi = sim.postprocess_sources(solved)

run_te = multi["te"]
run_tm = multi["tm"]
print(solved.solver_result.rhs_count)  # 2
```

## Dual-Basis Convenience Run

For mixed+basis+unpolarized outputs in one `SimulationResult`, keep using the
high-level convenience flag:

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

For `postprocess_sources(solve_sources(...))` outputs, each channel run is already pure. Use
`channel="mixed"` on that channel result:

```python
nf_te = pcl.compute_near_field_slice(multi["te"], channel="mixed")
nf_tm = pcl.compute_near_field_slice(multi["tm"], channel="mixed")
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
- Repeated direct solves on the same `Simulation` instance (for changed RHS/source)
  reuse both dense `A` and its LU factorization.
- `SimulationConfig.solver_compute_final_residual` controls whether pyceles
  computes final true residual diagnostics `||Ax-b||/||b||` after each solve.
  Keep it `True` by default; set it `False` for high-throughput repeated direct
  solves (for example LDOS maps) when you want maximum speed.
- `Simulation.solve_sources(..., solver_compute_final_residual=...)` can override
  the above per call.
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
- for `postprocess_sources(solve_sources(...))`, save each channel run separately (for example
  `save_simulation_h5(multi["te"], ...)`, `save_simulation_h5(multi["tm"], ...)`)

Loading helpers are available via `pcl.io`:
- `load_geometry_h5`, `load_solution_h5`, `load_far_field_h5`
- `load_near_field_components_h5`, `load_mapping_h5`
- `load_simulation_h5` (workflow-level convenience loader)

## License

MIT
