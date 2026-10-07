# Performance and scaling

pyceles keeps a NumPy/SciPy path as a reference implementation and adds
accelerated backends where they preserve the same prepared operator

```text
A = I - T W.
```

Performance depends on particle count, `lmax`, geometry, solver, precision,
and requested postprocessing. The phase profilers described below are the
public way to measure and compare those costs across different systems.
Their output is deliberately phase-resolved so that a regression in preparation,
solve, or field evaluation is visible instead of being hidden in one wall-time
number.

## Choosing a backend and solver

- **NumPy/SciPy** is the portable reference path and is convenient only for
  small systems, debugging, and independent checks.
- **CuPy** keeps the hot operator and Krylov state on the GPU. It is the normal
  choice for large direct, MLFMM, and periodic runs when a compatible CUDA
  device is available.
- **Pairwise coupling** is the direct `O(N^2)` finite-cluster reference and is
  useful while the system is moderate enough to fit in memory.
- **MLFMM** is the matrix-free high-frequency path for larger finite sphere
  clusters. It keeps close interactions exact and accelerates well-separated
  interactions through directional translations.
- **Ewald** is the general periodic coupling method. The hybrid
  `method="rayleigh"` path keeps exact self and near-vertical interactions and
  uses reciprocal Rayleigh scans for vertically separated particles; it is
  most useful for sufficiently extended cells.

The high-level default solver is restarted GMRES. BiCGSTAB can be a good
memory-light choice for finite systems, while periodic systems are generally
best solved with GMRES. CuPy also provides FGMRES, LGMRES, GCRO-DR, and LSQR;
NumPy uses the corresponding SciPy/reference implementations where available.
Solver selection is explicit: pyceles does not automatically infer a method
from system size. LSQR is single-RHS and requires an exact Hermitian adjoint
of the prepared operator; imported dense T matrices are compatible with this
algebraic path, but that does not create shape or material derivatives.

## Precision and memory

`compute_dtype` controls compact operator and contraction arithmetic, while
`accum_dtype` controls wider reductions on paths that support it. A common GPU
policy is `complex64` computation with `complex128` accumulation. Higher
precision can be important for demanding residual tolerances, cancellation-
sensitive lattice sums, and final validation.

CuPy establishes a guarded device-memory ceiling below physical VRAM. Persistent
data used by every matvec remain device-resident; pyceles does not silently
stage a hot cache through host or shared memory. If an exact-near cache or
other persistent repeated-apply state cannot fit, preparation fails explicitly
instead of changing the runtime regime. Bounded chunking and recomputation of
cheap intermediates remain available within an explicitly selected algorithm.

Direct dense solves assemble and factorize the full operator. They are useful
for small reference cases and repeated right-hand sides, but the caller owns
the memory budget; there is no hidden size-based fallback.

CuPy GMRES supports native block solves for compatible multi-RHS operators.
Block convergence requires every right-hand-side column to satisfy the target;
the inexpensive aggregate residual is only a monitor, followed by true
per-column residual checks. Restarted Krylov methods perform true residual
checks at restart boundaries by default. Disabling final checks is suitable
only for low-level profiling and must not be interpreted as a converged solve.

## Finite-cluster MLFMM

MLFMM is intended for high-frequency finite sphere clusters. Its exact-near
partition handles close pairs, while the sampled far field uses directional
translation operators. The current hierarchy is uniform-depth rather than
adaptive. NumPy and CuPy share the validated hierarchy construction; repeated
CuPy applies run on device.

The implementation keeps sampled-far interactions in the numerically safer
precision used by the directional translations. CuPy exact-near evaluation is
batched so that temporary dense mode blocks do not grow into an unbounded
`O(N^2)` allocation. The finite MLFMM operator exposes an exact adjoint on both
backends, including the reverse sampled-far traversal needed by LSQR.

## Public phase profilers

`benchmarks/profiling/finite_phases.py` profiles one finite-cluster run and writes
a `profile_summary.json` containing preparation, solver, far-field, and
near-field phases. Its default reference case is the 500-sphere
the `examples/sphere_parameters.txt` geometry with `lmax=3`, wavelength 550, a
Gaussian beam of width 2000, TE polarization, and `rtol=1e-4`.

The most relevant finite options are:

- `--operator-backend {numpy,cupy}` and `--postprocessing-backend {inherit,numpy,cupy}`;
- `--compute-dtype` and `--accum-dtype`;
- `--solver`, `--solver-rtol`, `--solver-restart`, and `--solver-maxiter`;
- `--cache-mode {off,on,both}`;
- `--skip-farfield`, `--skip-nearfield`, and `--cuda-profiler-api`.

For a complete finite/periodic suite, use the public orchestrator:

```powershell
python benchmarks/profiling/benchmark_suite.py --suite all
```

Use `--suite finite` or `--suite periodic` to select one family, `--backends
cupy` to restrict a refresh, `--skip-postprocessing` for solve-only runs, and
`--reuse-existing` to resume completed cases. The suite creates one output
directory per case and a manifest containing the exact command line.

### Finite reference phases

The following reference measurements use the default 500-sphere geometry and
the standard finite profiler settings. They are a baseline for regression
checks, not a promise of machine-independent runtime. The phase split is:

- **solver**: preparation plus the iterative/direct solve as reported by the
  profile;
- **far field**: conversion to scattered/initial/total plane-wave patterns;
- **near field**: evaluation of the configured Cartesian slice and components.

The reference environment was a 13th-generation Intel Core i7-13850HX with an
NVIDIA RTX 2000 Ada Laptop GPU, Python 3.12, NumPy/SciPy 2.5/1.18, and CuPy
14.0.1.

| backend and dtype | solver | far field | near field |
| --- | ---: | ---: | ---: |
| NumPy, `complex128/complex128` | 383.3 s | 20.57 s | 165.96 s |
| NumPy, `complex64/complex128` | 321.3 s | 19.35 s | 132.67 s |
| CuPy, `complex128/complex128` | 1.15 s | 1.79 s | 0.86 s |
| CuPy, `complex64/complex128` | 0.23 s | 0.70 s | 0.62 s |

The reference field slice uses `y=0`, `x=[-4000,4000]`, `z=[-3000,5000]`,
and `dx=40`. The benchmark is most useful when refreshed with the same
geometry, angular grids, precision, and solver; otherwise compare phase trends
rather than absolute seconds.

The finite scaling harness measures the separate pairwise and MLFMM solve
path:

```bash
python benchmarks/scaling/finite_pairwise_mlfmm.py \
  --couplings pairwise,mlfmm \
  --powers-of-two-range 10,20
```

Use a shorter range or `--n-values` for a smoke test. Its default MLFMM leaf
size is 32 particles; lower-level hierarchy and accuracy controls remain
available when the default is not appropriate. Pairwise coupling becomes
memory-bound well before the largest MLFMM cases.

### Periodic reference phases

`benchmarks/profiling/periodic_phases.py` profiles periodic preparation,
the complete solve, and optional exterior/interior `xy` and `xz` near-field
maps. It uses the 500-sphere prototype geometry in a 3000-unit square cell,
`lmax=3`, wavelength 550, normal-incidence TE illumination, GMRES with
`rtol=1e-4`, and adaptive Ewald shells by default. The most relevant options
are:

- `--coupling-backend {pairwise,mlfmm}` and `--periodic-method {ewald,rayleigh}`;
- `--operator-backend {numpy,cupy}`, precision, and `--cache-mode`;
- `--solver`, `--solver-rtol`, `--solver-restart`, and `--solver-maxiter`;
- `--field-bmax`, `--output-bmax`, `--rayleigh-z-cut`, and
  `--rayleigh-reciprocal-shells`;
- `--skip-nearfield` and `--cuda-profiler-api`.

Preparation includes method-specific work such as Ewald W-cache or Rayleigh
plan construction. The solve phase includes the complete Krylov run. Near-field
maps are reported separately because their cost depends strongly on the number
and placement of points.

The suite's standard periodic reference rows are:

| backend and dtype | coupling | preparation | linear solve | complete solve | iterations |
| --- | --- | ---: | ---: | ---: | ---: |
| NumPy, `complex128/complex128` | pairwise, cache on | 147.18 s | 520.74 s | 671.34 s | 160 |
| CuPy, `complex128/complex128` | pairwise, cache on | 7.28 s | 3.62 s | 11.10 s | 160 |
| CuPy, `complex128/complex128` | pairwise, cache off | 0.06 s | 514.21 s | 517.55 s | 160 |
| NumPy, `complex128/complex128` | Rayleigh, cache off* | 58.72 s* | 1.44 s* | 61.08 s* | 1* |
| CuPy, `complex128/complex128` | Rayleigh, cache off | 2.58 s | 9.46 s | 12.26 s | 160 |
| NumPy, `complex64/complex128` | pairwise, cache on | 148.90 s | 537.60 s | 689.76 s | 160 |
| CuPy, `complex64/complex128` | pairwise, cache on | 6.60 s | 2.60 s | 9.39 s | 160 |
| CuPy, `complex64/complex128` | pairwise, cache off | 0.06 s | 509.37 s | 512.68 s | 160 |
| NumPy, `complex64/complex128` | Rayleigh, cache off* | 57.74 s* | 1.09 s* | 59.53 s* | 1* |
| CuPy, `complex64/complex128` | Rayleigh, cache off | 2.52 s | 3.18 s | 5.89 s | 160 |

The asterisked NumPy rows are preparation/first-step probes, not converged
solutions and must not be compared to complete solve times. The estimated time
to convergence in those cases is of the order of the number of iterations (160)
times the measured time for a single iteration. Ewald's scalar lattice sums
remain `complex128` for cancellation safety, so changing compact compute
precision does not accelerate every preparation phase.

The same periodic profile can be refreshed directly:

```powershell
python benchmarks/profiling/benchmark_suite.py --suite periodic
```

For a small smoke run, pass `--periodic-maxiter` explicitly and use
`--skip-postprocessing`. Keep rows from one precision and geometry policy
together when making a new comparison.

## Finite implementation choices

The finite path emphasizes scalar-Legendre translation evaluation, reusable
translation tables and radial LUTs, explicit compute/accumulation precision,
the source-parallel CuPy pairwise kernel, the rotated-frame tilted-beam
initial-field fast path, and mode-index contraction for CuPy scattered fields.
The direct CuPy kernel has exact forward and Hermitian-adjoint actions,
supports block right-hand sides, and uses one scalar output mode per CUDA
block. That fixed mode decomposition is intentional: exploratory multi-mode
tiles repeated pair geometry and were slower on the measured workloads.

Several alternatives remain deliberately out of the default API because they
do not improve the general workload enough to justify their memory or
maintenance cost: full `z*kz` precomputation for near fields, giant all-alpha
batches, multiprocessing inside near-field evaluation, hardware-specific
chunk heuristics, and a regular-grid block preconditioner. A
rotation-translation-rotation traversal was likewise tested and eventually
not retained as a second translation backend.

The low-level `solver_preconditioner` hook is still available for applications
with a problem-specific approximate inverse. pyceles does not select one
automatically.

## Periodic scaling and Rayleigh safety

The standalone periodic scaling harness performs a complete preflight before
any solve and writes resumable JSON records:

```bash
python benchmarks/scaling/periodic_ewald_rayleigh.py
```

Rayleigh/Wood anomalies occur when a reciprocal order becomes grazing,

```text
|k_parallel + m b1 + n b2| = k_host.
```

They are singular channels of the artificial periodic lattice, not Ewald
solver failures. Use `pyceles.core.periodic.rayleigh_report` or
`suggest_safe_period_scales` before selecting a period. The geometry-only
report identifies grazing orders but does not predict the minimum of a full
coupling-norm scan. Production comparisons should vary shell and Rayleigh
truncation controls and compare representative cases with a converged Ewald
calculation.

For `method="rayleigh"`, the exact-near cache scales with the number of directed
non-self pairs inside the vertical band. A dense same-height layer can still be
quadratic. Reciprocal work is chunked from a bounded temporary-memory budget;
the persistent phase table and compact near cache remain part of the chosen
algorithm. The default near-band and reciprocal truncation are conservative
heuristics, not formal error bounds.

Periodic output-order selection is separate from Ewald shell selection. Ewald
`real_shells` and `reciprocal_shells` control lattice-sum convergence, whereas
`output_bmax` and `field_bmax` select propagating or evanescent diffraction
orders for output and near-field evaluation. For exterior near fields, choose
`field_bmax` explicitly and check map stability as it is increased.

## Periodic implementation notes

- Periodic many-body solves remain particle-local and operator/Krylov oriented;
  production workflows do not build a dense cluster T matrix.
- Adaptive Ewald shell selection includes a propagating-order reciprocal guard.
  `max_shells` is a safety cap, not an accuracy target.
- The CuPy Ewald path batches structural sums and same-plane classification
  under a temporary-memory budget; it does not retain a hidden quadratic index
  cache in the matrix-free path.
- Rayleigh preparation builds a reciprocal projection plan, one exact periodic
  self block, and a sparse exact-near cache. Its repeated apply scales with
  retained reciprocal orders and directed near pairs rather than all dense
  pairwise blocks, but a dense same-height layer remains effectively
  quadratic.
- The compact repeated-apply cache must fit the guarded device-memory budget;
  preparation fails instead of silently switching hot matvecs to host staging.
- Near-field evaluation uses the same vertical split as the Rayleigh solve for
  in-slab points. For repeated horizontal destination planes, CuPy reuses
  source-side reciprocal work; pass the complete point cloud in one call when
  practical.

Periodic observable output uses an explicit diffraction-order basis. Ewald
shell controls (`real_shells`, `reciprocal_shells`) are not output-order
controls: `output_bmax` and `field_bmax` select propagating and evanescent
orders. `output_bmax=None` is sufficient for `R/T/A`; exterior near fields
usually require an explicit `field_bmax` and a stability check.

## Cache management

Process-global precomputation caches are useful when repeating compatible
simulations. They can be cleared explicitly between independent workloads:

```python
import pyceles as pcl

pcl.core.clear_caches()
pcl.postprocessing.nearfield.clear_caches()
```

The library does not clear them automatically because retaining warm tables is
often the fastest choice in an interactive workflow.
