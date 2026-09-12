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
`fgmres[cupy]`, `lgmres[cupy]`, `bicgstab[cupy]`, harmonic `gcro[cupy]`, and
`lsqr[cupy]` and reference `lsqr` paths. The native path keeps the main Krylov
state on device; the NumPy path delegates to SciPy. GCRO retains a bounded
harmonic recycle space controlled by `solver_gcro_recycle_dim`; LSQR keeps only its
bidiagonal recurrence but requires an exact Hermitian-adjoint action and
currently supports one RHS. Multi-RHS workflows solve columns independently.
Restarted GMRES-family methods use
restart-boundary true-residual checks by default for robust stopping decisions.

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
- `LinearSolveResult.block_metadata` reports block batches and residual
  summaries; native block operators are required to accept `(n, nrhs)` inputs
  directly.
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

`SimulationConfig` defaults to restarted GMRES for a predictable, broadly
applicable baseline. Users can select BiCGSTAB or another supported method
explicitly when it better suits a particular workload; periodic systems in
particular can be sensitive to solver choice.

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

A previous rotation-translation-rotation coupling idea was also explored as an
alternative translation backend. After matching the formulas to the shipped
CELES-compatible conventions, the experimental implementation reproduced
translation blocks accurately but was much slower than the current reference
block builder on the public 500-particle benchmark: about `26x` slower at
`lmax=3` and `38x` slower at `lmax=4`. It is therefore not part of the active
translation path.

## Recent finite-cluster benchmark snapshot

Measured on a laptop with:

- CPU: `13th Gen Intel(R) Core(TM) i7-13850HX`,
- GPU: `NVIDIA RTX 2000 Ada Generation Laptop GPU`,
- Python: `3.12.10`,
- NumPy/SciPy: `2.5.1 / 1.18.0`,
- CuPy: `14.0.1`,
- script: `examples/profile_pyceles_phases.py`.

For a reproducible refresh of the supported finite and periodic cases, run the
public suite orchestrator from a clean Python process:

```powershell
python examples/profile_pyceles_benchmark_suite.py --suite all
```

Each case gets its own output directory and subprocess, and the root
`suite_manifest.json` records the exact commands. The finite suite runs all
four backend/precision rows with one far-field and near-field postprocessing
pass per row. If `--finite-cache-mode both` is selected, each finite profile
also measures cache-on and cache-off solves while reusing the primary solved
result for postprocessing. The pairwise periodic cache-on cases provide the
documented NumPy and CuPy xy, interior-xy, and xz field maps; cache-off, direct,
and Rayleigh cases remain solve-focused. Use `--suite finite` or
`--suite periodic` for one family, `--skip-postprocessing` when only
solve/preparation data is wanted, and `--skip-periodic-cache-off` when only
cache-on periodic rows are needed. Use `--backends cupy` to restrict a refresh
to CuPy rows when the NumPy reference rows are already available; the default
remains both backends. Every CuPy case is run to convergence. The slow NumPy
periodic cache-off and Rayleigh cases are one-iteration reference
probes whose linear-solve time can be extrapolated
from equivalent converged runs. `--reuse-existing` resumes a refresh and
reruns a case if its existing summary is missing a required field phase.

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
| NumPy, `complex128/complex128` | `383.3 s` |
| NumPy, `complex64/complex128` | `321.3 s` |
| CuPy, `complex128/complex128` | `5.66 s` |
| CuPy, `complex64/complex128` | `0.51 s` |

Postprocessing is measured once for every finite backend/precision row. These
measurements use the same profile geometry and angular/field grids; cache-mode
and solver comparisons do not repeat the field kernels because they do not
depend on the solver iteration count.

| finite backend and dtype | far field | near field |
| --- | ---: | ---: |
| NumPy, `complex128/complex128` | `20.57 s` | `165.96 s` |
| NumPy, `complex64/complex128` | `19.35 s` | `132.67 s` |
| CuPy, `complex128/complex128` | `1.87 s` | `0.89 s` |
| CuPy, `complex64/complex128` | `0.70 s` | `0.65 s` |

The low-level linear-solver `preconditioner=...` callable hook remains available
for custom experiments; no built-in preconditioner is selected automatically.

## Pairwise optimization notes

The current implementation emphasizes:

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

Use `examples/profile_pyceles_benchmark_suite.py --suite periodic` to refresh
this complete supported snapshot.

To refresh a different precision policy, pass
`--periodic-compute-dtype` and `--periodic-accum-dtype`; keep the resulting rows
together as one complete suite snapshot rather than mixing runs.

```powershell
python examples/profile_pyceles_benchmark_suite.py --suite periodic --periodic-maxiter 800
```

- 500 prototype spheres from `examples/sphere_parameters.txt` in a 3000 nm
  square cell, `lmax=3`, wavelength 550 nm, homogeneous medium `n=1.0`;
- normally incident TE plane wave with analytic source projection;
- GMRES `rtol=1e-4`, `maxiter=800`, automatic eta and adaptive Ewald shells;
- CuPy postprocessing at `dx=30 nm` for exterior-xy, slab-interior-xy, and
  vertical xz maps.

`prep` includes method-specific preparation (including W-cache or Rayleigh
construction), and `solve` is the complete profiled solve phase. One-iteration
rows are reference probes, not converged solves.

| backend and dtype | coupling mode | prep | linear solve | solve | iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| NumPy, `complex128/complex128` | Pairwise, W cache on | `147.18 s` | `520.74 s` | `671.34 s` | 160 |
| CuPy, `complex128/complex128` | Pairwise, W cache on | `7.41 s` | `4.66 s` | `12.25 s` | 160 |
| CuPy, `complex128/complex128` | Pairwise, W cache off | `0.06 s` | `536.65 s` | `540.19 s` | 160 |
| NumPy, `complex128/complex128` | Pairwise, W cache off | `0.03 s` | `285.93 s` | `429.74 s` | 1* |
| CuPy, `complex128/complex128` | Rayleigh, cache off | `2.58 s` | `11.38 s` | `14.30 s` | 160 |
| NumPy, `complex128/complex128` | Rayleigh, cache off | `58.72 s` | `1.44 s` | `61.08 s` | 1* |
| NumPy, `complex64/complex128` | Pairwise, W cache on | `148.90 s` | `537.60 s` | `689.76 s` | 160 |
| CuPy, `complex64/complex128` | Pairwise, W cache on | `6.57 s` | `3.60 s` | `10.37 s` | 160 |
| CuPy, `complex64/complex128` | Pairwise, W cache off | `0.06 s` | `530.97 s` | `534.46 s` | 160 |
| NumPy, `complex64/complex128` | Pairwise, W cache off | `0.03 s` | `285.82 s` | `430.03 s` | 1* |
| CuPy, `complex64/complex128` | Rayleigh, cache off | `2.53 s` | `4.59 s` | `7.38 s` | 160 |
| NumPy, `complex64/complex128` | Rayleigh, cache off | `57.74 s` | `1.09 s` | `59.53 s` | 1* |

All 160-step GMRES rows reached approximately `9.35e-5`; the two precision
policies agree to the displayed residual and power-balance precision. Here
`complex64/complex128` stores periodic translation data and performs its
contractions in complex64, while the cancellation-sensitive scalar Ewald sum
remains complex128. That scalar work dominates NumPy Ewald preparation and the
cache-off pairwise probe, so those phases should not be expected to speed up
and can be slightly slower than `complex128/complex128` because of ordinary
host/runtime variability. Compute-dtype-sensitive phases can still benefit once
that structural work is amortized: in this snapshot the complex64 direct
assembly/factorization and Rayleigh repeated apply are faster. This differs from
the finite pairwise case, where more of the hot path changes dtype; CuPy still
benefits where c64 device kernels dominate. The rows are a paired snapshot, not
a promise of ordering for short or host-bound phases.

Direct dense validation (field work and final residual check skipped):

| backend and dtype | W-block generation | assembly | factorization | solve phase |
| --- | ---: | ---: | ---: | ---: |
| NumPy, `complex128/complex128` | `147.23 s` | `3.51 s` | `20.07 s` | `318.34 s` |
| CuPy, `complex128/complex128` | `5.45 s` | `2.14 s` | `43.68 s` | `55.13 s` |
| NumPy, `complex64/complex128` | `147.22 s` | `2.09 s` | `10.59 s` | `305.05 s` |
| CuPy, `complex64/complex128` | `4.57 s` | `0.26 s` | `1.49 s` | `10.38 s` |

`*` One-iteration reference probe. Extrapolating its linear-solve time is
useful for rough planning, but preparation and convergence behavior still need
to be measured separately.

The pairwise cache-on postprocessing phases use the primary solved result and
are independent of Krylov convergence:

| backend and dtype | exterior xy | interior xy | vertical xz |
| --- | ---: | ---: | ---: |
| NumPy, `complex128/complex128` | `1.40 s` | `1515.71 s` | `1624.09 s` |
| CuPy, `complex128/complex128` | `1.47 s` | `7.57 s` | `9.86 s` |
| NumPy, `complex64/complex128` | `1.38 s` | `1481.91 s` | `1600.28 s` |
| CuPy, `complex64/complex128` | `1.46 s` | `7.57 s` | `9.86 s` |

The interior plane is the occupied-slab midpoint (`z=1488.49 nm`; 872 of
10,201 pixels are inside particles for this seed) and is saved as
`nearfield_xy_interior_total.npz`. The direct rows are validation paths rather
than a production scaling route; cache-off remains memory-light but recomputes
periodic Ewald work on every Krylov matvec.

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
  contracted and stored once in the selected compute dtype; a complex64 cache
  therefore uses half the persistent bytes of a complex128 cache and performs
  its repeated contractions in complex64. Time preparation separately from warmed
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

For repeated horizontal destination planes, the CuPy in-slab evaluator also
recognizes shared z coordinates and reuses the source-side reciprocal
calculation before applying the destination phases. This is an automatic,
exact dispatch decision; there is no user-facing switch. Mixed-z arrays (for
example, a vertical cross-section) are grouped internally when they contain
enough points per plane, while small or irregular groups use the generic exact
pair path. When several planes are needed, pass the complete point cloud to
one `compute_periodic_near_field` call whenever practical. Calling the helper
once per line or stripe repeats setup and transfer work and can be much slower,
even when each individual stripe qualifies for the plane optimization. The
optimization applies to in-slab periodic near fields; it does not change the
periodic solver matvec or the exterior Rayleigh-order evaluator.

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
