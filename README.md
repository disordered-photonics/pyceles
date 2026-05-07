# pyceles

pyceles is a Python reimplementation of the MATLAB CELES package for electromagnetic
simulation of large particle ensembles with the T-matrix method.
The code keeps a NumPy + SciPy reference implementation and now also ships an
optional CuPy backend for the direct many-body solve path, plus a CuPy-backed
MLFMM repeated-apply path for the current high-frequency sphere-cluster regime.

This repository focuses on:
- Correctness first (CELES conventions, reproducible examples/notebooks)
- A clean NumPy + SciPy reference implementation
- Performance via vectorization + CELES-style caching/LUTs (no MEX build)
- An optional CuPy backend for direct GPU pairwise solves, without duplicating
  the physical operator structure
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
- NumPy matrix-free MLFMM coupling backend for large sphere clusters:
  - `coupling_backend="mlfmm"` on the NumPy operator path
  - exact near interactions on the resolved leaf partition
  - automatic stage selection between direct pairwise fallback, single-level HF,
    and multilevel HF
  - one uniform-depth occupied-box hierarchy (non-adaptive octree)
  - high-frequency-only formulation: no low-frequency/static regime handling is
    included in the current backend
  - grouped dense leaf operators are the default repeated-apply path on CPU,
    while a lower-persistent-memory on-the-fly leaf mode remains available as a
    lean reference/debug option
  - relative-offset batching, interior radial-LUT reuse, and shared directional
    interpolation/transforms across occupied boxes
  - structured resolved-plan, hierarchy, and memory diagnostics through the
    prepared operator
  - precision policy: `compute_dtype=complex64` affects exact-near MLFMM work,
    while sampled far interactions remain `complex128` on both NumPy and CuPy
- CuPy matrix-free MLFMM repeated-apply backend for the same high-frequency
  hierarchy plan:
  - `operator_backend="cupy", coupling_backend="mlfmm"`
  - one-time hierarchy/build preparation stays on the validated CPU reference
    implementation, while exact-near and sampled-far repeated applies run on
    device
  - compact host-cache/prepared-cache payloads remain the source of truth and
    device-resident state is rebuilt from them as needed
  - streamed-far diagnostics report chunk/frontier decisions in backend-native
    units as well as byte-oriented summaries
- CuPy direct backend for the same `A = I - T W` operator:
  - fused RawKernel pairwise coupling `W·x` for `complex64` and `complex128`
  - GPU single-body `T` support for diagonal groups and explicit dense spherical-basis blocks
  - native CuPy GMRES / FGMRES / LGMRES / BiCGSTAB solve paths
  - native CuPy block-GMRES for multi-RHS GMRES runs
  - inherited CuPy far-field scattered-PWP postprocessing path
  - inherited CuPy near-field postprocessing path for:
    - scattered field
    - dominant Gaussian/general initial-field paths
    - homogeneous-sphere internal fields
- SciPy GMRES wrapper with tqdm progress
- Near-field evaluation with CELES formulas:
  - scattered field
  - initial field (plane-wave-pattern integral)
  - total field with internal-field replacement inside particles
    (`Sphere`, `LayeredSphere`, and `Spheroid`)
  - canonical helper returning `initial/scattered/internal/total` components in one call
  - geometry-agnostic helper (`compute_near_field`) plus planar convenience wrapper (`compute_near_field_slice`)
- Far-field plane-wave pattern + forward/backward power flux (CELES formulas)
  - canonical helper returning `initial/scattered/total` PWPs in one call
  - plane-wave cross sections:
    - differential scattering cross section (`dC_sca/dOmega`)
    - total scattering cross section (PWP-integrated, SMUTHI-style cluster definition)
    - extinction cross section (coefficient-based)
    - local absorption cross section (`C_abs = C_abs_local`)
    - explicit raw/closure diagnostics:
      - `C_abs_raw_diff = C_ext_raw - C_sca_raw`
      - `Delta_closure = C_abs_raw_diff - C_abs_local`
    - low-level helper contract:
      - `plane_wave_cross_sections(...)` requires explicit `local_absorption`
      - legacy raw-difference fallback is opt-in via `allow_raw_diff_fallback=True`
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
  - finite-power diagnostics now expose both raw missing-energy and local
    dissipation terms:
    - `P_abs_raw_diff = P_initial - P_transmitted - P_reflected`
    - `P_abs_local` from the generic local exciting-field route (`e = b + W x`)
    - per-particle local diagnostics:
      - `P_abs_local_particles`
      - `A_local_particles`
    - `A_raw_diff = P_abs_raw_diff / P_initial`
    - `A_local = P_abs_local / P_initial`
    - `Delta_power_closure = P_abs_raw_diff - P_abs_local`
- Geometry sanity check:
  - by default, `Simulation` enforces disjoint particle circumscribing spheres
    (required by T-matrix superposition), can be disabled via `check_circumscribing_sphere_overlap=False`
- Explicit particle descriptors:
  - `Simulation(config, particles=[...])` accepts mixed supported particle families
    (`Sphere`, `LayeredSphere`, `Spheroid`) in one geometry
  - `Simulation.n_particles` / `SimulationResult.n_particles` provide canonical
    particle counts across all supported particle descriptors
- Axisymmetric spheroid support:
  - homogeneous `Spheroid` particle-local scattering blocks in the spherical SVWF basis
  - aligned and rotated spheroid `T`-matrix support in CELES ordering
  - internal-field evaluation inside spheroids
  - regression coverage against isolated-particle SMUTHI and ScatterPy references
  - current limitation: near-field evaluation remains unreliable for points
    outside a spheroid but inside its circumscribing sphere, because the main
    implementation still uses the outgoing spherical SVWF expansion there
  - exploratory surface-integral, arbitrary-precision, and first spheroidal-shell
    postprocessing variants were investigated, but none is ready to replace the
    default path yet
- Explicit no-scatterer (source-only) simulations:
  - pass `particles=[]` to run a beam-only simulation
- Channel-aware polarization workflow:
  - source polarization accepts CELES-style `"TE"`, `"TM"` or Jones weights `(a_te, a_tm)`
  - `SLMSource(base_source, modulation)` wrapper for angular-spectrum complex modulation
    (phase/amplitude masks on TE/TM plane-wave amplitudes)
  - `BesselBeam` exact non-paraxial cone-ring angular-spectrum source
    (`order_m` OAM phase, tiltable axis via `polar_angle`/`azimuthal_angle`,
    TE/TM Jones-compatible)
  - canonical conversion helpers:
    `pwp_to_svwf_regular`, `angular_spectrum_to_svwf_regular`,
    `svwf_regular_to_pwp`, `svwf_outgoing_to_pwp`
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
  - simulation power outputs now include absolute local absorbed-power diagnostics
    for dipole-driven runs:
    - `P_abs_local`
    - `P_abs_local_particles`
  - near-field initial field supports dipoles and masks exact dipole-center
    grid hits as `NaN`
- Solver API extensions:
  - matrix-matrix solves (`A @ X`) with multiple RHS columns
  - warm start vectors/matrices (`solver_warm_start`)
  - preconditioner hook (`solver_preconditioner`)
  - inherited postprocessing backend policy via
    `SimulationConfig(postprocessing_backend="inherit" | "numpy" | "cupy")`
- Angular-grid API:
  - one shared CELES-style default grid (`polar_angles`, `azimuthal_angles`)
  - optional split grids for source projection and far-field outputs
    (`source_*`, `farfield_*`)

## Current limits / not yet landed

- Embedded dipoles inside `Sphere` or `LayeredSphere` particles are not
  supported yet in the main solver path. Current dipole workflows assume dipole
  centers remain in the homogeneous host medium.
- Complex host refractive indices are not supported in the current workflow.
  This is required for beam/plane-wave sources, but it is not a fundamental
  limitation for local dipole sources and may be revisited in the future.
- T-matrix superposition still requires disjoint circumscribing spheres for all
  particle families. In practice this remains especially restrictive for close
  configurations of elongated spheroids, where alternative coupling schemes can
  be implemented.
- Periodic boundary conditions is still work in progress.
- At the moment, particles in a simulation need to share the same `lmax`.
- Exterior near-field evaluation for spheroids remains unreliable at points
  lying inside the circumscribing sphere but outside the physical particle.

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

For contributors, install the development toolchain:

```bash
python -m pip install -e .[dev]
```

If you have a recent NVIDIA GPU and want the optional CuPy backend:

```bash
python -m pip install -e .[cupy]
```

The extra is named `cupy` rather than `gpu` so future accelerator extras can
remain explicit (`cupy`, `pyopencl`, `numba`, ...), instead of collapsing
different backends into one generic label.

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
- uses solver/preconditioner tuning around the matrix-free operator

pyceles now ships two direct backends for the same many-body operator:
- a NumPy/SciPy reference path,
- a CuPy path using a fused RawKernel for the direct pairwise coupling matvec.

For larger dilute clusters on the NumPy path, pyceles also ships a matrix-free
high-frequency MLFMM coupling backend selected through
`SimulationConfig(coupling_backend="mlfmm")`. This keeps near interactions
exact while accelerating the far coupling through single-level or multilevel
directional box operators, depending on the resolved hierarchy depth.

Current implementation choices:
- the hierarchy is uniform-depth, not adaptive: pyceles first builds one root
  cube for the full particle set, then subdivides it to one shared depth and
  keeps only the occupied boxes on each level
- stage policy is depth-based:
  - depth `0-1`: direct pairwise fallback
  - depth `2`: single-level HF MLFMM on the occupied leaf level
  - depth `>=3`: multilevel HF MLFMM with upward/downward transfer between
    occupied levels
- the current backend is intentionally high-frequency only: here "HF" means
  the far coupling is represented through directional sampled translators on
  boxes that are large enough for that asymptotic formulation to be effective
- low-frequency stabilization/switching is intentionally out of scope for the
  current implementation, so pyceles does not try to blend this backend into a
  separate LF regime today

For moderate dense systems that fit in memory, both backends also support a
direct dense solve with cached LU reuse for repeated RHS workflows (for example
multi-source sweeps or local LDOS probes). On the CuPy path this uses
CuPy/cuSOLVER rather than a custom fused solve kernel.

For matrix-free iterative solves, CuPy GMRES also supports true multi-RHS
inputs: `solve_linear_system(..., backend="cupy", method="gmres", b.shape==(n, nrhs))`
routes to a native block-GMRES path, and `Simulation.solve_sources(...)` uses
that path automatically on labeled multi-channel runs.

For single-RHS iterative solves on the CuPy backend, pyceles also ships native
`fgmres[cupy]`, `lgmres[cupy]`, and `bicgstab[cupy]` paths. These keep the
main Krylov state on device. For restarted GMRES-family methods, pyceles now
uses restart-boundary true-residual checks by default (`compute_final_residual=True`)
for robust stopping decisions; setting `compute_final_residual=False` disables
those checks for profiling-focused runs.

Current MLFMM scope/limits:
- matrix-free MLFMM stages are available on both NumPy and CuPy operator
  backends, with sampled-far interactions fixed to `complex128`
- on the NumPy path, grouped dense leaf operators are the default repeated-apply
  shape; the compact on-the-fly leaf mode is kept as a lower-persistent-memory
  reference/debug path rather than the main CPU fast path
- prepared NumPy MLFMM operators expose canonical plan, hierarchy, and memory
  diagnostics to make backend comparisons and storage tradeoffs explicit
- on the CuPy path, one-time hierarchy/build preparation stays on the validated
  CPU reference implementation, while repeated MLFMM applies run on device
- CuPy exact-near evaluation is memory-aware: near interactions are applied
  from directed pair indices plus compact translation tables on device, rather
  than from a pre-uploaded dense near-block tensor
- CuPy grouped far-offset accumulation uses weighted device kernels (for both
  single-level and multilevel sampled far passes) to reduce intermediate tensor
  traffic in the repeated-apply path
- grouped far/transfer accumulation uses non-atomic kernels under a strict
  grouped-batch uniqueness contract; preparation fails fast if an unexpected
  non-unique grouped schedule is encountered
- CuPy MLFMM prepared-cache payloads are compact host payloads; device-resident
  prepared data is rebuilt on load instead of being pickled directly
- matrix-free iterative solves for true MLFMM stages
- dense/direct solves use fast pairwise assembly when available and otherwise
  fall back to generic dense assembly through repeated matrix-free applies
- the current implementation targets the high-frequency regime only
- the octree policy is uniform-depth rather than adaptive

Practical CuPy notes for dense systems:
- CuPy GMRES in pyceles now uses a native implementation by default for
  `solve_linear_system(..., backend="cupy", method="gmres")`.
- CuPy GMRES multi-RHS runs (`nrhs > 1`) use a native block-GMRES path and
  report per-RHS final true residuals in `LinearSolveResult`.
- Block-GMRES convergence requires every RHS column to satisfy the requested
  tolerance; a low aggregate/block residual alone is not accepted as converged.
- Inner block iterations still use a cheap aggregate residual proxy for
  monitoring, but pyceles now runs an immediate per-RHS true-residual gate when
  that proxy reaches target (plus restart-boundary checks), so large restart
  values can still stop promptly without waiting for cycle end.
- Motivation for this native path:
  - built-in CuPy GMRES callback/convergence visibility is restart-cycle based,
    so convergence behavior between restarts is opaque
  - this can force awkward restart trade-offs: large restart may hide early
    convergence and overshoot iterations; very small restart can degrade
    convergence quality
  - native GMRES keeps the Krylov work on device while exposing per-inner-step
    progress (`pr_rel_res`) and verifying true residuals at restart boundaries
    for robust stop decisions
- pyceles does not expose a public runtime toggle to swap back to built-in
  CuPy GMRES in the simulation API
- CuPy block-GMRES now runs with native 2D operator applies on prepared CuPy
  paths and keeps generic low-level preconditioner plumbing available for
  experimental solver calls.
- Block-GMRES speedup remains workload-dependent. Dense multi-RHS cases can now
  outperform sequential solves, while small or weakly coupled RHS sets may stay
  near parity or slightly slower; profile on your workload before assuming gains.
- `LinearSolveResult.block_metadata` reports whether block adapters were used
  (`operator_block_adapter_used`, `preconditioner_block_adapter_used`) so users
  can verify the active execution path.
- Check your GPU's `singleToDoublePrecisionPerfRatio` before assuming
  `complex128` is close to a `2x` cost over `complex64`. On many consumer/laptop
  parts the ratio is very high, and end-to-end `complex128` slowdowns can be
  an order of magnitude larger than `2x` for transcendental-heavy kernels.
- Use `examples/benchmark_cupy_precision_ratio.py` to measure machine-local
  FP64/FP32 penalty quickly before choosing production dtypes.
- On dense/non-normal systems, restarted GMRES can be very restart-sensitive.
  Small restart values may stagnate; larger restart values can recover
  convergence.

In internal solver sweeps on representative pairwise CuPy cases, restarted
GMRES-family methods remained restart-sensitive at small restart budgets,
while native `bicgstab[cupy]` was consistently a strong all-rounder with a
much smaller solver-state footprint than restarted GMRES-family methods.
For memory-aware large-scale CuPy runs, `bicgstab` is therefore a sensible
first solver choice unless a specific geometry shows better behavior with a
restarted method.
Postprocessing follows the solve backend by default through
`postprocessing_backend="inherit"`. The current CuPy postprocessing slices are:
- scattered far-field SVWF-to-PWP assembly,
- scattered near-field,
- the dominant Gaussian/general initial-field paths,
- homogeneous-sphere internal fields.

Mixed non-spherical internal-field cases still fall back to the NumPy
reference kernels today.

Current CuPy feature-parity gaps relative to the NumPy reference path include:
- layered-sphere internal near-field kernels,
- spheroid internal near-field kernels,
- mixed non-spherical internal-field subsets,
- callback-only axisymmetric single-body GPU wrappers.
Current performance milestones after the first direct and MLFMM GPU backends are:
- continue improving the CuPy raw-kernel path and native solver stack for larger low-`lmax` clusters,
- continue improving the streamed CuPy MLFMM backend for larger high-frequency clusters,
- extend GPU acceleration further into postprocessing-heavy workflows and remaining backend gaps.
A previous CELES experiment with rotation-translation-rotation (RTR) coupling idea
was explored as a possible alternative translation backend. After matching the RTR
block formulas to the shipped CELES-compatible translation conventions, the experimental implementation
reproduced translation blocks accurately but remained much slower than the current
reference block builder on the public `500`-particle benchmark (`~26x` slower at `lmax=3`,
`~38x` slower at `lmax=4`). RTR therefore remains an interesting mathematical
direction, but not competitive against pyceles' low-`lmax` brute-force path.

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
- solver: `gmres`, `rtol=1e-4`, `restart=25`, `maxiter=100`

Reproduce:
```bash
python examples/profile_pyceles_phases.py --n-particles 500 --lmax 3 --dx 40 --operator-backend numpy --compute-dtype complex128 --accum-dtype complex128 --cache-mode off --solver-restart 25 --solver-maxiter 100 --out-dir outputs/profile_matrix_cpu_c128_none --quiet
python examples/profile_pyceles_phases.py --n-particles 500 --lmax 3 --dx 40 --operator-backend numpy --compute-dtype complex64 --accum-dtype complex128 --cache-mode off --solver-restart 25 --solver-maxiter 100 --out-dir outputs/profile_matrix_cpu_c64_none --quiet
python examples/profile_pyceles_phases.py --n-particles 500 --lmax 3 --dx 40 --operator-backend cupy --postprocessing-backend inherit --compute-dtype complex128 --accum-dtype complex128 --cache-mode off --solver-restart 25 --solver-maxiter 100 --out-dir outputs/profile_matrix_cupy_c128_none --quiet
python examples/profile_pyceles_phases.py --n-particles 500 --lmax 3 --dx 40 --operator-backend cupy --postprocessing-backend inherit --compute-dtype complex64 --accum-dtype complex128 --cache-mode off --solver-restart 25 --solver-maxiter 100 --out-dir outputs/profile_matrix_cupy_c64_none --quiet
```

Phase wall times:
- NumPy, `complex128/complex128`, no preconditioner:
  - Solver: `389.8 s`
  - Far-field: `21.9 s`
  - Near-field: `181.2 s`
- NumPy, `complex64/complex128`, no preconditioner:
  - Solver: `324.9 s`
  - Far-field: `19.6 s`
  - Near-field: `146.7 s`
- CuPy, `complex128/complex128`, no preconditioner:
  - Solver: `11.4 s`
  - Far-field: `1.9 s`
  - Near-field: `14.6 s`
- CuPy, `complex64/complex128`, no preconditioner:
  - Solver: `1.16 s`
  - Far-field: `0.89 s`
  - Near-field: `4.53 s`

On this benchmark, native CuPy GMRES reduced restart overshoot on the
no-preconditioner runs (from `40` to `22` iterations at `rtol=1e-4`,
`restart=25`), which is where the largest runtime gain appears for
`complex128`.

## Cumulative Optimization Notes

The current performance is the cumulative result of several tweaks.
Key improvements include:

- Scalar-Legendre translation path (`legendre_normalized_trigon_scalar`) in `translation_block`
  to reduce overhead in the matrix-free pair-block assembly hot path.
- Translation table/LUT reuse (ab5 cache + radial Hankel LUT + optional exact `W_ij` block cache).
- End-to-end precision policy (`compute_dtype`, `accum_dtype`) enabling CELES-style
  mixed precision on CPU today and portable behavior for a future GPU backend.
- Rotated-frame initial beam near-field initial-field fast path:
  - for regular (non-periodic boundary conditions) workflows, pyceles rotates points to the beam-local frame,
    evaluates the beam initial field with analytic azimuth integration, then rotates
    E/H vectors back to the user frame.
  - the general alpha-beta quadrature path is still kept as fallback for flexible/periodic workflows.
- CuPy near-field scattered-field rewrite that contracts over the mode index
  before assembling Cartesian components, reducing temporary tensor payloads
  while preserving the reference formulas.

Benchmarked but intentionally *not* kept as defaults:
- full `z*kz` phase precompute for near-field initial evaluation: modest speedup with large memory payload.
- giant all-alpha batch strategies: no consistent end-to-end gains.
- multiprocessing near-field initial evaluation: good speedups but high RAM/process overhead and extra complexity.
- hardware-aware scattered-field chunk heuristics: no robust win and regressions on dense near-field canvases.
- shared `matmul` rewrite for CuPy scattered-field contractions: cleaner algebra, but slightly slower than the current `einsum` path on the profiled `lmax=3/4` cases.

Important for CELES users: these gains preserve the matrix-free iterative workflow by default.
The solver does not require assembling/storing a global dense matrix; optional block caching
is an explicit tradeoff for systems where RAM is plentiful.

For long interactive sessions or large exploratory sweeps, process-global
precompute caches can be cleared explicitly to release memory:

```python
import pyceles as pcl

pcl.core.clear_caches()
pcl.postprocessing.nearfield.clear_caches()
```

pyceles does not call these automatically between runs because warm caches are
often beneficial when repeating solves in the same process.

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
  circumscribing spheres). Interior-embedded dipoles are not supported yet in
  the present solver path.
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
print(solved.solver_result.rhs_count)  # 2
print(solved.solver_result.method)     # "gmres[cupy-block]" on CuPy iterative multi-RHS
```

## Mixed Particle Descriptors

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

Notes:
- Layered spheres are treated as one particle each (one center, one solved
  outgoing-coefficient block).

Array-to-particle helper for homogeneous spheres:

```python
particles = pcl.spheres_from_arrays(
    positions=positions,          # shape (N, 3)
    radii=radii,                  # shape (N,)
    refractive_indices=n_particle # scalar or shape (N,)
)
sim = pcl.Simulation(cfg, particles=particles)
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
run = pcl.Simulation(cfg, particles=particles).run()

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

For beam inspection/debugging, run a simulation with no particles via
`particles=[]`:

```python
run = pcl.Simulation(
    cfg,
    particles=[],
).run()
```

Notes:
- this yields zero scattered coefficients/fields and preserves incident-field outputs

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

Notes:
- `direct_max_n` (or `max_n` in low-level direct solver) limits the matrix size `n`
  of the linear system, not the number of RHS columns.
- For many-sphere systems, `n = N_spheres * n_modes(lmax)`.
- `SimulationConfig.solver_direct_max_n` is the high-level knob passed to the
  low-level solver's `direct_max_n/max_n` guard.
- Repeated direct solves on the same `Simulation` instance (for changed RHS/source)
  reuse both dense `A` and its LU factorization.
- `SimulationConfig.solver_compute_final_residual` controls true-residual
  verification/diagnostics policy:
  - for native CuPy restarted GMRES/FGMRES/LGMRES, `True` enables restart-boundary
    true-residual checks (robust default), while `False` disables them;
  - for other solver paths, it controls whether final `||Ax-b||/||b||`
    diagnostics are computed after the solve.
  Keep it `True` by default; set it `False` for profiling or high-throughput
  repeated solves when you want maximum speed and can accept less residual
  verification.
- `Simulation.solve_sources(..., solver_compute_final_residual=...)` can override
  the above per call.
- SciPy GMRES progress reports SciPy's cheap preconditioned residual
  (`pr_rel_res`). SciPy BiCGSTAB, LGMRES, and GCROTMK callbacks do not expose a
  cheap residual scalar, so pyceles reports iteration-only progress for those
  methods instead of spending an extra matrix-vector product per callback just
  for display. Final true-residual diagnostics are still controlled by
  `solver_compute_final_residual`.
- `solver_preconditioner` is a custom callable hook. pyceles no longer ships a
  built-in grid-block preconditioner in the high-level simulation API.

## HDF5 Output With Basis Channels

`pcl.io.save_simulation_h5(...)` now stores:
- mixed solution/far-field groups (as before)
- particle-native geometry under `geometry/particles` (no persisted generic
  `positions`/`radii` tables)
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

## Related Works

pyceles is an independent Python implementation, but it draws heavily on the
multiple-scattering literature and on ideas demonstrated in earlier implementations.
Some relevant references are grouped by how directly they shape the current code or
the near-term roadmap.

Direct predecessors:

- Egel et al., *CELES: CUDA-accelerated simulation of electromagnetic scattering by large ensembles of spheres*, JQSRT 199 (2017) 103-110. https://doi.org/10.1016/j.jqsrt.2017.05.010
- Egel et al., *SMUTHI: A Python package for the simulation of light scattering by multiple particles near or between planar interfaces*, JQSRT 273 (2021) 107846. https://doi.org/10.1016/j.jqsrt.2021.107846

Related references for validation of present and future features:

- Auguie et al., *SMARTIES: User-friendly codes for fast and accurate calculations of light scattering by spheroids*, JQSRT 174 (2016) 39-55. https://doi.org/10.1016/j.jqsrt.2016.01.005
- Pena-Rodriguez et al., *Near- and far-field Mie scattering calculations for a multilayered sphere*, CPC 180 (2009) 2348-2354. https://doi.org/10.1016/j.cpc.2009.07.010
- Rasskazov et al., *STRATIFY: a comprehensive and versatile MATLAB code for a multilayered sphere*, OSAC 3 (2020) 2290-2306. https://doi.org/10.1364/OSAC.399979
- Dufva et al., *Unified derivation of the translational addition theorems for the spherical scalar and vector wave functions* Progress In Electromagnetics Research B 4 (2008) 79-99 http://dx.doi.org/10.2528/PIERB07121203
- Martin, *Another look at addition theorems for vector spherical wavefunctions* Mathematical Methods in the Applied Sciences 47.16 (2024) 12443-12459. https://doi.org/10.1002/mma.9987
- Mun et al., *Multipole decomposition for interactions between structured optical fields and meta-atoms*, OE 28 (2020) 36756-36770. https://doi.org/10.1364/OE.409775
- Gumerov and Duraiswami, *Computation of scattering from clusters of spheres using the fast multipole method* JASA 117 (2005) 1744-1761. https://doi.org/10.1121/1.1853017
- Theobald et al., *Simulation of light scattering in large, disordered nanostructures using a periodic T-matrix method*, JQSRT 272 (2021) 107802. https://doi.org/10.1016/j.jqsrt.2021.107802
- Nečada and Törmä, *Multiple-Scattering T-matrix Simulations for Nanophotonics: Symmetries and Periodic Lattices*. 30.2 (2021) 357-395. https://doi.org/10.4208/cicp.OA-2020-0136
- Mackowski and Kolokolova, *Application of the multiple sphere superposition solution to large-scale systems of spheres via an accelerated algorithm*, JQSRT 287 (2022) 108221. https://doi.org/10.1016/j.jqsrt.2022.108221
- Mackowski, *Extension of the Multiple Sphere T-Matrix code to include multiple plane boundaries and 2-D periodic systems*, JQSRT 290 (2022) 108292. https://doi.org/10.1016/j.jqsrt.2022.108292
- Markkanen and Yuffa, *Fast superposition T-matrix solution for clusters with arbitrarily shaped constituent particles*, JQSRT 189 (2017) 181-188. https://doi.org/10.1016/j.jqsrt.2016.11.004
- Stilgoe et al., *Computational toolbox for scattering of focused light from flattened or elongated particles using spheroidal wavefunctions*, JQSRT 331 (2025) 109267. https://doi.org/10.1016/j.jqsrt.2024.109267
- Gumerov and Duraiswami, *Fast Multipole Methods on Graphics Processors* Journal of Computational Physics 227.18 (2008) 8290-8313. https://doi.org/10.1016/j.jcp.2008.05.023
- Beutel et al., *Unified lattice sums accommodating multiple sublattices for solutions of the Helmholtz equation in two and three dimensions* Phys Rev A  107 (2023) 013508. https://doi.org/10.1103/PhysRevA.107.013508
- Beutel et al., *treams – a T-matrix-based scattering code for nanophotonics* Computer Physics Communications 297 (2024) 109076. https://doi.org/10.1016/j.cpc.2023.109076

## License

MIT
