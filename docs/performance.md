# Performance notes

pyceles keeps a NumPy/SciPy reference path and adds accelerated backends where
they preserve the same operator structure. This page records the main
performance assumptions, benchmark snapshots, and optimization decisions that
are too detailed for the root README.

## Solver and backend structure

CELES is fast because it runs translation/matvec work on GPU, reuses cached
translation tables and radial Hankel LUTs, and tunes solver/preconditioner
choices around a matrix-free operator.

pyceles currently ships:

- a NumPy/SciPy reference path,
- a CuPy path using fused RawKernels for direct pairwise coupling,
- a high-frequency MLFMM backend for larger dilute sphere clusters,
- experimental periodic Ewald paths on NumPy and CuPy.

For moderate dense systems that fit in memory, both backends support direct
dense solves with cached LU reuse for repeated RHS workflows. On the CuPy path
this uses CuPy/cuSOLVER rather than a custom fused solve kernel.

For matrix-free iterative solves, CuPy GMRES supports true multi-RHS inputs:
`solve_linear_system(..., backend="cupy", method="gmres", b.shape==(n, nrhs))`
routes to a native block-GMRES path, and `Simulation.solve_sources(...)` uses
that path automatically on labeled multi-channel runs.

For single-RHS iterative solves on the CuPy backend, pyceles also ships native
`fgmres[cupy]`, `lgmres[cupy]`, and `bicgstab[cupy]` paths. These keep the main
Krylov state on device. Restarted GMRES-family methods use restart-boundary
true-residual checks by default for robust stopping decisions.

## Practical CuPy notes

- CuPy GMRES in pyceles uses a native implementation by default.
- CuPy GMRES multi-RHS runs use native block-GMRES and report per-RHS final true
  residuals.
- Block-GMRES convergence requires every RHS column to satisfy the requested
  tolerance.
- Inner block iterations use a cheap aggregate residual proxy for monitoring,
  followed by per-RHS true-residual checks when the proxy reaches target.
- pyceles does not expose a public runtime toggle to swap back to built-in CuPy
  GMRES in the simulation API.
- `LinearSolveResult.block_metadata` reports whether block adapters were used.
- Check your GPU's `singleToDoublePrecisionPerfRatio` before assuming
  `complex128` is close to a `2x` cost over `complex64`. On many consumer or
  laptop GPUs the ratio is high, and `complex128` slowdowns can be much larger
  than `2x` for transcendental-heavy kernels.
- Measure machine-local FP64/FP32 penalty before choosing production dtypes for
  large CuPy runs.
- Restarted GMRES can be restart-sensitive on dense/non-normal systems; small
  restart values may stagnate.

Internal solver sweeps on representative pairwise CuPy cases showed that
restarted GMRES-family methods remained restart-sensitive at small restart
budgets, while native `bicgstab[cupy]` was consistently a strong all-rounder
with a smaller solver-state footprint. For memory-aware large-scale CuPy runs,
`bicgstab` is a sensible first solver choice unless a specific geometry shows
better behavior with a restarted method.

## MLFMM implementation notes

The MLFMM backend currently targets the high-frequency far-coupling regime.
Close particle-pair interactions are handled exactly in the near field, so a
separate low-frequency far-coupling branch has not yet been a practical
performance bottleneck for the finite-size particle ensembles pyceles targets.

Sampled-far MLFMM interactions remain `complex128` because high-order
directional translations are more sensitive to phase/interpolation error than
the exact-near path.

Current implementation details:

- matrix-free MLFMM stages are available on both NumPy and CuPy operator
  backends,
- NumPy grouped dense leaf operators are the default repeated-apply shape,
- compact on-the-fly leaf mode is kept as a lower-persistent-memory
  reference/debug path,
- prepared NumPy MLFMM operators expose plan, hierarchy, and memory diagnostics,
- CuPy exact-near evaluation is memory-aware and avoids a pre-uploaded dense
  near-block tensor,
- CuPy grouped far-offset accumulation uses device kernels that multiply each
  grouped interaction by its precomputed interpolation/translation weight during
  accumulation. Doing this in one pass avoids materializing large intermediate
  weighted tensors for every Krylov apply.
- grouped far/transfer accumulation uses non-atomic kernels under a strict
  grouped-batch uniqueness contract,
- dense/direct solves use fast pairwise assembly when available and otherwise
  fall back to generic dense assembly through repeated matrix-free applies,
- the octree policy is uniform-depth rather than adaptive.

A previous rotation-translation-rotation coupling idea was explored as an
alternative translation backend. After matching the formulas to the shipped
CELES-compatible conventions, the experimental implementation reproduced
translation blocks accurately but was much slower than the current reference
block builder on the public 500-particle benchmark: about `26x` slower at
`lmax=3` and `38x` slower at `lmax=4`.

## Recent finite-cluster benchmark snapshot

Measured on a laptop with:

- CPU: `13th Gen Intel(R) Core(TM) i7-13850HX`,
- GPU: `NVIDIA RTX 2000 Ada Generation Laptop GPU`,
- Python: `3.12.10`,
- NumPy/SciPy: `2.4.2 / 1.17.1`,
- CuPy: `14.0.1`,
- script: `examples/profile_pyceles_phases.py`.

Common benchmark parameters:

- geometry: `N=500` spheres from `examples/sphere_parameters.txt`, `lmax=3`,
- source: Gaussian beam, `wavelength=550`, `n_medium=1.0`,
  `beam_width=2000`, TE, normal incidence,
- angular grids: `n_beta=3601`, `n_alpha=180`,
- near-field slice: plane `y=0`, `x=[-4000, 4000]`,
  `z=[-3000, 5000]`, `dx=40`,
- solver: `bicgstab`, `rtol=1e-4`.

Representative phase wall times:

- NumPy, `complex128/complex128`, no preconditioner:
  - Solver: `369.5 s`,
  - Far-field: `21.9 s`,
  - Near-field: `180.7 s`.
- NumPy, `complex64/complex128`, no preconditioner:
  - Solver: `319.3 s`,
  - Far-field: `19.5 s`,
  - Near-field: `146.7 s`.
- CuPy, `complex128/complex128`, no preconditioner:
  - Solver: `5.69 s`,
  - Far-field: `2.04 s`,
  - Near-field: `14.06 s`.
- CuPy, `complex64/complex128`, no preconditioner:
  - Solver: `0.46 s`,
  - Far-field: `0.70 s`,
  - Near-field: `3.12 s`.


The "no preconditioner" wording refers to a regular-grid block preconditioner
that was previously shipped with pyceles, directly inspired by the CELES
implementation. This option was later removed due to the lack of robust speedups
in the measured workloads, especially with the current CuPy implementation.
The low-level linear-solver `preconditioner=...` callable hook remains available
for custom experiments.

## Pairwise optimization notes

Kept improvements:

- scalar-Legendre translation path in `translation_block`,
- translation table/LUT reuse,
- optional exact `W_ij` block cache,
- end-to-end `compute_dtype`/`accum_dtype` precision policy,
- rotated-frame tilted-beam initial near-field fast path,
- CuPy scattered near-field contraction over mode index before Cartesian
  assembly.

Benchmarked but intentionally not kept as defaults:

- full `z*kz` phase precompute for near-field initial evaluation,
- giant all-alpha batch strategies,
- multiprocessing near-field initial evaluation,
- hardware-aware scattered-field chunk heuristics,
- shared `matmul` rewrite for CuPy scattered-field contractions,
- the built-in regular-grid block preconditioner, which was useful as an
  experiment but not compelling enough to keep as public API.

These gains preserve the matrix-free iterative workflow by default. Optional
block caching is an explicit tradeoff for systems where RAM is plentiful.

For long interactive sessions or large exploratory sweeps, process-global
precompute caches can be cleared explicitly:

```python
import pyceles as pcl

pcl.core.clear_caches()
pcl.postprocessing.nearfield.clear_caches()
```

pyceles does not call these automatically between runs because warm caches are
often beneficial when repeating solves in the same process.

## Periodic benchmark snapshot

Use `examples/profile_pyceles_periodic_phases.py` for the rectangular-cell
periodic profile. The script places the 500 prototype spheres from
`examples/sphere_parameters.txt` into a non-overlapping 3000 nm square periodic
cell, solves one normally incident plane-wave RHS, and optionally computes
periodic xy/xz near-field slices.

Common benchmark parameters:

- geometry: `N=500`, `lmax=3`, homogeneous medium, 3000 nm periodic square cell,
- source: plane wave, `wavelength=550`, `n_medium=1.0`, TE, normal incidence,
- source projection: analytic plane-wave RHS,
- solver: `bicgstab`, `rtol=1e-4`, `maxiter=800`,
- periodic options: automatic `eta`, adaptive shell counts,
  `shell_tolerance=1e-10`,
- precision: currently `complex128/complex128`.

Measured phase wall times on the same laptop/GPU:

- CuPy, explicit periodic W cache:
  - W-cache population: `11.8 s`,
  - BiCGSTAB solve: `12.9 s` for 111 iterations,
  - Full solve phase: `42.3 s`,
  - Near field: xy `1.4 s`, xz `112 s`.
- CuPy, no periodic W cache:
  - BiCGSTAB solve: `817 s` for 115 iterations, about `7.1 s/iteration`,
  - Full solve phase: `828 s`,
  - Near field: xy `1.4 s`, xz `112 s`.
- NumPy, explicit periodic W cache:
  - W-cache population: `120 s`,
  - BiCGSTAB solve: `833 s` for 115 iterations,
  - Full solve phase: `966 s`,
  - Near field: xy `1.4 s`, xz `1443 s`.
- NumPy, no periodic W cache:
  - not run in the current snapshot: even one uncached source sweep is about
    `120 s`, while BiCGSTAB generally needs two operator applications per full
    iteration. A 115-iteration solve would therefore take well over 7 h before
    postprocessing.
- CuPy direct dense validation, with near field and final residual check skipped:
  - periodic W-block generation: `8.4 s`,
  - source-streamed dense `A` assembly: `1.70 s`,
  - dense LU factorization: `43.4 s`,
  - full solve phase: `65.5 s`.
- NumPy direct dense validation, with near field and final residual check skipped:
  - periodic W-block generation: `123 s`,
  - source-streamed dense `A` assembly: `3.85 s`,
  - dense LU factorization: `21.8 s`,
  - full solve phase: `162 s`.

The direct dense rows are validation paths, not the intended scaling route. The
source-streamed assembler discards each periodic W-block batch after writing
its columns into `A`, avoiding a second dense-matrix-sized temporary cache. The
cache-off periodic path is memory-light, but it recomputes periodic Ewald work
on every Krylov matvec. For large periodic runs, explicit W-block caching is
currently the practical path when memory permits. Profiling shows that the
cache-off cost is dominated by shifted-reciprocal Ewald arithmetic rather than
the final tensor contraction, so direct-to-output fusion is not presently a
compelling option.

## Periodic optimization notes

- Periodic many-body solves remain particle-local and operator/Krylov oriented.
  pyceles applies `A = I - T W` through the prepared operator and does not build
  a dense cluster T-matrix as a production workflow.
- Adaptive Ewald shell selection has a propagating-order reciprocal guard,
  clear non-convergence errors, and reusable non-pair shell workspaces.
  `real_shells=None` or `reciprocal_shells=None` means adaptive accumulation;
  explicit integers force fixed shell counts, while `max_shells` is a safety cap
  rather than an accuracy target.
- The CuPy periodic Ewald path evaluates real-space and shifted reciprocal
  structural sums with fused device kernels, handles same-plane pairs on
  device, and batches source particles by temporary-memory budget.
- Explicit periodic W-block caching is an opt-in memory/runtime tradeoff.
- Dense direct periodic solves are validation paths. For diagonal particle-local
  `T` operators, dense `A` is assembled directly from cached periodic W blocks.
- Periodic near-field evaluation uses Rayleigh orders above/below the particle
  slab and local periodic SVWF evaluation inside the slab.

### Periodic output basis

Periodic observable output uses an explicit diffraction-order basis.
Periodic output-order selection is not the same thing as Ewald shell selection.
Ewald `real_shells` and `reciprocal_shells` control lattice-sum convergence in
the operator. By contrast, `output_bmax` and `field_bmax` are reciprocal-space
radii for diffraction/Rayleigh orders: all integer orders satisfying
`|k_parallel + m*b1 + n*b2| <= bmax` are included.

"Evanescent orders" refers to evanescent diffraction/Rayleigh orders in that
output table.

For `R/T/A`, `output_bmax=None` is enough because only propagating orders
contribute to flux. Exterior periodic near-field maps often need evanescent
orders, so the production near-field API requires an explicit `field_bmax` or a
configured `PeriodicOptions.output_bmax`.

The public periodic profile and MSTM-comparison scripts provide a convenience
heuristic when their near-field `bmax` CLI option is omitted:

```text
bmax = sqrt(k^2 + (D / d)^2)
```

Here `d` is a characteristic exterior distance from the particle slab and `D` is
a target evanescent decay, with the scripts currently defaulting to `D = 8`
(`exp(-8)` amplitude decay over `d`). Increasing the value includes more
evanescent orders and can be useful for stricter near-field map convergence
checks. For production studies, prefer an explicit `field_bmax` sweep and inspect
stability of the resulting field maps.
