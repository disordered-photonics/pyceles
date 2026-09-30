# Validation and reproducibility

pyceles combines unit tests for numerical kernels, physics-oriented invariants,
API-contract tests, and comparisons with independent scattering references.
The CPU path is the correctness anchor; CuPy tests check backend agreement and
GPU-specific execution where a CUDA device is available.

## Automated tests

Tests use markers to make the required runtime explicit:

- `unit`, `physics`, and `regression` cover numerical and physical behavior;
- `api_contract` protects public shapes and protocols;
- `io`, `filesystem`, and `hdf5` cover persistence and plotting-facing paths;
- `reference` covers independent formulas or fixed oracle data;
- `gpu` requires CuPy and a working CUDA device;
- `fake_gpu` exercises CuPy dispatch without CUDA;
- `slow` marks longer-running tests.

Useful commands are:

```bash
python -m pytest -q -m "not gpu and not slow"
python -m pytest -q -m "not gpu"
python -m pytest -q -m fake_gpu
python -m pytest -q -m gpu
```

The complete suite is the strongest check before a release:

```bash
python -m pytest -q
```

## Covered numerical contracts

The test suite covers, among other areas:

- CELES-compatible VSWF indexing, Wigner-3j tables, angular functions, and
  translation coefficients;
- Mie and layered-sphere particle operators;
- direct pairwise and dense-operator consistency;
- NumPy/CuPy agreement for selected operators and postprocessing;
- translation, rotation, inversion-parity, particle-order, and reciprocity
  invariants;
- near-field, far-field, cross-section, power-balance, absorption, and dipole
  diagnostics;
- HDF5 persistence and standard spectral `.tmat.h5` basis conversion;
- periodic Ewald helpers, Rayleigh diagnostics, and selected periodic backend
  checks;
- spheroid regression cases against independent particle calculations.

## Reproducible benchmark drivers

The repository includes scripts for controlled external comparisons:

```bash
python benchmarks/validation/mstm_finite.py \
  --mstm-exe <path-to-mstm-executable>

python benchmarks/validation/mstm_periodic.py \
  --mstm-exe <path-to-mstm-executable>
```

The finite and periodic scaling harnesses write self-contained JSON records and
can resume an interrupted sweep:

```bash
python benchmarks/scaling/finite_pairwise_mlfmm.py \
  --couplings pairwise,mlfmm --powers-of-two-range 10,12

python benchmarks/scaling/periodic_ewald_rayleigh.py \
  --families lateral --n-values-lateral 114,228
```

These are measurements, not fixed performance guarantees. Record the backend,
precision, solver, geometry, and postprocessing settings with any result that
will be compared across machines.

For a loader-only check of a standard dense T-matrix file:

```bash
python examples/inspect_standard_tmatrix.py \
  --tmatrix <particle.tmat.h5> --wavelength 600
```

External programs such as MSTM, SMUTHI, or TREAMS require their own local
installations and conventions. When comparing fields, align wavelength units,
host medium, basis/polarization convention, angular or spatial sampling, and
solver residual before interpreting differences. The repository does not claim
that a single external discretization is a universal reference for every
postprocessing quantity.
