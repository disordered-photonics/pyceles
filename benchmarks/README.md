# Benchmarks and validation drivers

The programs in this tree are reproducibility and performance tools.
As such, they are intentionally more configurable than the short workflows
in `examples/`.

## Scaling

- `scaling/finite_pairwise_mlfmm.py` compares finite pairwise and MLFMM
  coupling on a deterministic particle ladder.
- `scaling/periodic_ewald_rayleigh.py` measures the lateral and vertical
  periodic growth families, including preflight checks and resumable JSON
  records.

Both drivers default to the documented CuPy precision and solver policies;
use their `--help` output to select a shorter smoke ladder or an explicitly
requested backend.

## Profiling

`profiling/finite_phases.py` and `profiling/periodic_phases.py` split
preparation, solve, and field-evaluation phases.  `profiling/benchmark_suite.py`
launches the public reference cases in isolated Python processes so allocator
state does not leak between measurements.

## Independent validation

`validation/mstm_finite.py` and `validation/mstm_periodic.py` drive the
external MSTM comparisons used for validation.  They require a local MSTM
installation and are not part of the basic installation test.
