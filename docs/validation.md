# Validation and reproducibility

pyceles is developed around reproducible numerical checks combining unit tests for
individual numerical kernels, physics-oriented invariants, regression tests against
independent references, and larger benchmark scripts.

## Automated tests

The test suite is organized by pytest markers:

- `unit`: isolated numerical and API checks,
- `physics`: physically meaningful end-to-end or invariant checks,
- `regression`: locked behavior and reference cases,
- `io`: HDF5, plotting, and workflow-oriented tests,
- `gpu`: tests requiring a real CuPy/CUDA runtime,
- `fake_gpu`: CuPy-dispatch tests that do not require a real CUDA device,
- `slow`: longer-running tests kept out of fast local gates,
- `filesystem` and `hdf5`: tests touching temporary files or HDF5 payloads,
- `reference`: comparisons against independent formulas, fixed oracle data, or external references,
- `api_contract`: public-shape and protocol checks.

Useful commands:

```bash
python -m pytest -q -m "not gpu and not slow"
python -m pytest -q -m "not gpu"
python -m pytest -q -m gpu
```

## Reference classes covered by tests

The current tests cover, among other areas:

- CELES-compatible VSWF indexing and mode metadata,
- Wigner-3j machinery,
- spherical angular functions and recurrences,
- Mie and layered-sphere T-matrix routines,
- translation coefficients and coupling operators,
- direct dense/pairwise matvec consistency,
- NumPy and CuPy backend agreement for selected operators,
- near-field and far-field helper contracts,
- energy/power consistency checks,
- local absorption and dipole diagnostics,
- HDF5 workflow persistence,
- periodic Ewald helper routines and CuPy parity checks,
- spheroid regression cases against external references.

## Public benchmark scripts

The repository includes benchmark scripts intended to produce human-inspectable
validation artifacts.

Finite-cluster MSTM comparison:

```bash
python examples/run_mstm_pyceles_cluster_benchmark.py \
  --mstm-exe <path-to-mstm-executable>
```

Periodic MSTM comparison:

```bash
python examples/run_mstm_pyceles_periodic_benchmark.py \
  --mstm-exe <path-to-mstm-executable>
```

Profiling entry points:

- `examples/profile_pyceles_phases.py`
- `examples/profile_pyceles_periodic_phases.py`

These scripts are intended to produce reproducible local timing and comparison
artifacts. Exact numbers depend strongly on CPU, GPU, CUDA, BLAS, and driver
versions.
