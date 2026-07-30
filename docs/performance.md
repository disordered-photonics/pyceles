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

The high-level defaults follow the same split: finite clusters default to
BiCGSTAB, while periodic systems default to restarted GMRES because periodic
BiCGSTAB can stagnate when the corresponding GMRES solve progresses.

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

### Periodized MLFMM (experimental)

NumPy and CuPy can periodize the MLFMM hierarchy for rectangular
two-dimensional cells:

```python
import pyceles as pcl

config = pcl.SimulationConfig(
    periodic=pcl.PeriodicSpec(
        lattice=pcl.RectangularLattice2D(ax, ay),
        options=pcl.PeriodicOptions(method="ewald"),
    ),
    coupling_backend="mlfmm",
    operator_backend="cupy",  # or "numpy"
)
```

Ewald prepares the lattice closure once; repeated applies reuse the ordinary
MLFMM hierarchy and finite image corrections. CuPy follows the same precision
policy as finite MLFMM: sampled box translations remain complex128, while
central and explicit boundary-image leaf interactions use the requested
compute dtype through the same fused exact-leaf kernel and radial LUT. The GPU
path does not evaluate Ewald sums during Krylov applications and does not
retain particle-pair periodic translation blocks. The exact image Hankel LUT
may span several cell lengths, but the separate regular-wave leaf-center LUT
is deliberately sliced to the occupied leaf radius rather than duplicating
that long range on host and device.

The path remains experimental. Tune MLFMM accuracy and periodic Ewald options
together when comparing it with pairwise periodic Ewald, and inspect the
periodization diagnostics for the number of exact residual image pairs in
tight cells.

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

For a reproducible refresh of the supported finite and periodic cases, run the
public suite orchestrator from a clean Python process:

```powershell
python examples/profile_pyceles_benchmark_suite.py --suite all
```

Each case gets its own output directory and subprocess, and the root
`suite_manifest.json` records the exact commands. Use `--suite finite` or
`--suite periodic` for one family, `--skip-postprocessing` when only
solve/preparation data is wanted, and `--skip-periodic-cache-off` when only
cache-on periodic rows are needed. Every CuPy case is run to convergence. The
slow NumPy periodic cache-off, Rayleigh, and periodized-MLFMM cases are
one-iteration reference probes whose linear-solve time can be extrapolated
from equivalent converged runs. `--reuse-existing` resumes a refresh without
rerunning completed profile summaries.

Common benchmark parameters:

- geometry: `N=500` spheres from `examples/sphere_parameters.txt`, `lmax=3`,
- source: Gaussian beam, `wavelength=550`, `n_medium=1.0`,
  `beam_width=2000`, TE, normal incidence,
- angular grids: `n_beta=3601`, `n_alpha=180`,
- near-field slice: plane `y=0`, `x=[-4000, 4000]`,
  `z=[-3000, 5000]`, `dx=40`,
- solver: `bicgstab`, `rtol=1e-4`.

Solve/preparation wall times from the default suite are:

| backend and dtype | solver phase |
| --- | ---: |
| NumPy, `complex128/complex128` | `428.2 s` |
| NumPy, `complex64/complex128` | `357.4 s` |
| CuPy, `complex128/complex128` | `5.52 s` |
| CuPy, `complex64/complex128` | `0.45 s` |

Postprocessing is measured only on representative paths. These measurements
use the same profile geometry and angular/field grids; they are not repeated
for every solver row because the field kernels do not depend on the solver
iteration count.

| finite backend and dtype | far field | near field |
| --- | ---: | ---: |
| NumPy, `complex128/complex128` | `20.9 s` | `41.8 s` |
| NumPy, `complex64/complex128` | `18.1 s` | `33.8 s` |
| CuPy, `complex128/complex128` | `1.81 s` | `4.13 s` |
| CuPy, `complex64/complex128` | `0.72 s` | `2.23 s` |


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
- solver: `gmres`, `rtol=1e-4`, `maxiter=800`,
- periodic options: automatic `eta`, adaptive shell counts,
  `shell_tolerance=1e-10`,
- precision: currently `complex128/complex128`.

Measured phase wall times on the same laptop/GPU are grouped by the work they
represent. `Solve` is the complete solve phase; `prep` includes operator
preparation and, where applicable, W-cache generation. The one-iteration rows
are marked explicitly and are not presented as converged solves.

| coupling and backend | mode | prep | linear solve | solve | iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| Pairwise, NumPy | W cache on | `120.2 s` | `598.3 s` | `718.7 s` | 160 |
| Pairwise, CuPy | W cache on | `10.0 s` | `5.5 s` | `15.7 s` | 160 |
| Pairwise, CuPy | W cache off | `0.03 s` | `580.8 s` | `581.0 s` | 160 |
| Pairwise, NumPy | W cache off | `0.03 s` | `247.0 s` | `247.2 s` | 1* |
| Periodized MLFMM, CuPy | cache off | `7.4 s` | `125.1 s` | `132.7 s` | 160 |
| Periodized MLFMM, NumPy | cache off | `186.4 s` | `13.0 s` | `199.6 s` | 1* |
| Rayleigh, CuPy | cache off | `0.03 s` | `47.3 s` | `47.5 s` | 160 |
| Rayleigh, NumPy | cache off | `0.03 s` | `44.5 s` | `44.6 s` | 1* |

The direct validation rows, with field work and the final residual check
skipped, were:

| backend | W-block generation | assembly | factorization | solve phase |
| --- | ---: | ---: | ---: | ---: |
| NumPy | `121.4 s` | `3.51 s` | `22.9 s` | `148.3 s` |
| CuPy | `8.0 s` | `1.86 s` | `43.7 s` | `53.7 s` |

`*` One-iteration reference probe. Extrapolating its linear-solve time is
useful for rough planning, but preparation and convergence behavior still need
to be measured separately.

Periodic postprocessing is also kept separate from the solve table. The
periodic profiles measured the following representative pairwise W-cache-on
field maps:

| periodic backend and dtype | xy slice | xz slice |
| --- | ---: | ---: |
| NumPy, `complex128/complex128` | `1.5 s` | `1318.9 s` |
| CuPy, `complex128/complex128` | `1.5 s` | `115.7 s` |


The direct dense rows are validation paths, not the intended scaling route. The
source-streamed assembler discards each periodic W-block batch after writing
its columns into `A`, avoiding a second dense-matrix-sized temporary cache. The
cache-off periodic path is memory-light, but it recomputes periodic Ewald work
on every Krylov matvec. For large periodic runs, explicit W-block caching is
currently the practical path when memory permits. On CuPy, the cache-on path
now stores the accepted dense cache as one contiguous device matrix and applies
it with a single GEMV/GEMM per matvec instead of repeating source-block
contractions. The NumPy cache-on path retains its source-block cache because
converting its Python block dictionary would temporarily duplicate a
dense-matrix-sized allocation. Profiling shows that the
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
  structural sums with fused device kernels, classifies same-plane pairs
  inside each device batch, and batches source particles by temporary-memory
  budget. Same-plane pair indices are not retained across batches, so the
  matrix-free path does not acquire a hidden quadratic index cache.
- `PeriodicOptions(method="rayleigh")` is an opt-in hybrid alternative for
  vertically extended cells. Exact Ewald work is restricted to the periodic
  self block and non-self pairs satisfying `|delta_z| <= rayleigh_z_cut`; the
  remaining reciprocal coupling is applied by two z-sorted semiseparable scans.
  For `N` particles, `Q` retained reciprocal orders, and `K` directed non-self
  near pairs, the repeated apply scales as `O(N * Q * Nm + K * Nm^2)` and the
  exact-near cache as `O(K * (2*lmax+1)^2)`, instead of recomputing Ewald sums for all
  `N^2` pairs or storing all dense W blocks.
- The default Rayleigh half-band is one medium wavelength in `|delta_z|`,
  enlarged to at least twice the largest particle circumscribing radius. This
  is deliberately conservative: evanescent reciprocal orders have already
  decayed substantially before a pair enters the scan. An explicit
  `rayleigh_z_cut` may be used for convergence studies but cannot be smaller
  than the particle-safe diameter bound.
- `rayleigh_reciprocal_shells` fixes the square reciprocal half-width. With the
  default `None`, pyceles selects a half-width from a conservative evanescent
  envelope controlled by the same `shell_tolerance` and `max_shells` settings
  used by Ewald accumulation. This is a truncation heuristic, not an error
  proof; production studies should compare representative results against a
  converged Ewald configuration and sweep the band/half-width.
- Reciprocal work chunks are selected automatically from a fixed temporary-
  memory budget. This avoids a small public chunk cap that would leave the GPU
  under-occupied and add avoidable Python/einsum loop overhead on the CPU.
- Preparation builds the reciprocal projection plan, the shared exact self
  block, and the sparse exact-near cache. `cache_translation_blocks=False`
  disables the dense periodic W cache, but the hybrid method still retains its
  `O(N Q)` lateral phase table and `O(K (2*lmax+1)^2)` compact exact-near
  structural cache. These are part of the Rayleigh repeated-apply design, not a
  leaked dense W matrix. Structural sums are accumulated in complex128 and then
  stored once in the selected compute dtype; a complex64 cache therefore uses
  half the persistent bytes of a complex128 cache without changing the current
  complex64 matvec arithmetic. Time preparation separately from warmed
  matvecs: the first direct `apply()` includes any preparation that has not
  already been requested through `populate_coupling()`.
- Before allocating a CuPy exact-near cache, pyceles estimates its compact byte
  size together with the remaining Rayleigh tables and bounded apply workspace.
  The automatic policy uses the same guarded CuPy pool ceiling as streamed
  MLFMM, so Windows/WDDM shared-memory spill is not treated as available device
  memory. Small caches stay resident on the GPU. Oversized caches remain in
  ordinary host memory and are copied synchronously through one reusable bounded
  device staging buffer during each matvec. This fallback trades PCIe traffic
  for GPU residency; it does not remove the underlying `K` scaling.
- CuPy contracts the compact exact-near cache in bounded pair batches. The
  batching budget includes the possible dense `(Nm, Nm)` block intermediate
  and its library workspace, rather than only the final `(Nm,)` contribution.
  This keeps dense bands from creating a hidden `O(K Nm^2)` temporary spanning
  every near pair at once.
- When measuring Rayleigh error against Ewald, pin a demonstrably converged
  Ewald `eta` and shell configuration (or sweep them). A difference against an
  automatically selected, insufficiently converged Ewald reference is not a
  Rayleigh truncation estimate and can dominate the comparison in large cells.
- The hybrid method is most useful when the vertical band is sparse. A dense
  same-height layer still has `K = O(N^2)` and therefore receives little memory
  or setup benefit. Exact grazing diffraction orders (Wood anomalies) are
  rejected by the Rayleigh operator; use exact Ewald or move away from the
  anomaly.
- The periodic phase profiler accepts `--periodic-method rayleigh` together
  with the `--rayleigh-*` convergence controls. Hybrid profiling requires
  `--cache-mode off`, because its sparse near cache replaces dense W caching.
- Explicit periodic W-block caching is an opt-in memory/runtime tradeoff.
- Dense direct periodic solves are validation paths. For diagonal particle-local
  `T` operators, dense `A` is assembled directly from cached periodic W blocks.
- Periodic near-field evaluation uses Rayleigh orders above/below the particle
  slab. For in-slab points, `method="rayleigh"` applies the same vertical split
  as the coupling operator: reciprocal z scans accumulate vertically distant
  particle sources, while source-point pairs inside `rayleigh_z_cut` retain the
  exact Ewald local-SVWF projection. For `P` field points and `Kp` exact-near
  source-point pairs, this changes the dominant shape from all `P*N` Ewald
  evaluations to approximately `O((P + N) * Q * Nm + Kp * Nm)` plus structural
  Ewald work for the near pairs. Dense same-height field maps can therefore
  remain expensive even when the solve itself benefits strongly from Rayleigh.

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
