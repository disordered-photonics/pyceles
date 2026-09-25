# pyceles

pyceles is a NumPy/SciPy-first implementation of CELES-style multiple
scattering with the T-matrix method. It provides a correctness-oriented CPU
path and optional CuPy acceleration for direct coupling, MLFMM, periodic
operators, and selected postprocessing stages.

The central many-body system is written as

```text
A = I - T W
```

where `T` contains particle-local scattering and `W` contains inter-particle
coupling. The same prepared-operator interface is used by the reference and
accelerated backends, with explicit control over solver method and numerical
precision.

pyceles builds on the conventions and ideas established by
[CELES](https://github.com/disordered-photonics/celes) and
[SMUTHI](https://gitlab.com/AmosEgel/smuthi). Their publications are listed in
[`docs/references.md`](docs/references.md) and can be cited when pyceles is
used in scientific work.

## Supported workflows

- Homogeneous, layered, axisymmetric, and perfect-conductor particles.
- Standard dense `.tmat.h5` particle blocks, selected at an exact wavelength
  and converted to the CELES spherical-wave convention.
- Plane waves, Gaussian and Laguerre-Gaussian beams, Bessel beams, focused
  beams, angular-spectrum/SLM sources, local dipoles, and dipole collections.
- Direct pairwise coupling and high-frequency MLFMM for finite clusters.
- GMRES-family methods, BiCGSTAB, GCRO-DR, LSQR with exact adjoints, and
  direct dense solves where the caller has sufficient memory.
- Single- and multi-source workflows, including block solves for compatible
  CuPy GMRES runs.
- Component-resolved near fields, far-field plane-wave patterns, cross
  sections, power balances, local absorption, and dipole LDOS diagnostics.
- HDF5 persistence for particle-native geometry, solutions, fields, and typed
  diagnostics.
- Experimental rectangular 2D periodic workflows with Ewald coupling and the
  opt-in hybrid exact-near/Rayleigh-far operator.

The detailed capability inventory is in
[`docs/capabilities.md`](docs/capabilities.md). The curated public API map is
in [`docs/api.md`](docs/api.md), and runnable usage starts with
[`docs/quickstart.md`](docs/quickstart.md).

## Scope and current boundaries

pyceles is pre-1.0. Important boundaries are still present:

- one simulation currently uses one common `lmax` for all particles;
- the homogeneous host path currently requires a real positive refractive
  index;
- T-matrix superposition requires disjoint circumscribing spheres by default;
- opaque imported T matrices support exact operator adjoints, but cannot
  provide hidden internal fields or shape/material derivatives;
- embedded dipoles inside particles are not yet supported by the main solver path;
- periodic workflows are experimental and currently target homogeneous
  rectangular 2D lattices with plane-wave excitation;
- spheroid exterior fields require care inside a circumscribing sphere but
  outside the physical spheroid.

See [`docs/limitations.md`](docs/limitations.md) for the full boundary list.

## Installation

pyceles requires Python 3.12 or newer.

```bash
python -m pip install -U pip
python -m pip install pyceles
```

For a checkout in editable mode:

```bash
python -m pip install -e .
```

For GPU use, install the CuPy wheel matching the CUDA runtime before installing
pyceles, or use the package extra:

```bash
python -m pip install cupy-cuda12x   # or cupy-cuda13x
python -m pip install pyceles
```

```bash
python -m pip install pyceles[cupy]
```

The optional `dev` and `notebooks` extras are intended for contributors and
interactive examples; see [`docs/installation.md`](docs/installation.md).

## Minimal example

```python
import pyceles as pcl

particles = [
    pcl.Sphere(position=(-180.0, 0.0, 0.0), radius=90.0, refractive_index=1.5 + 0j),
    pcl.Sphere(position=(180.0, 0.0, 0.0), radius=90.0, refractive_index=1.5 + 0j),
]

source = pcl.PlaneWave(
    wavelength=550.0,
    medium_n=1.0 + 0j,
    polarization="TE",
)
config = pcl.SimulationConfig(
    wavelength=550.0,
    n_medium=1.0 + 0j,
    lmax=3,
    solver_method="gmres",
    solver_rtol=1e-6,
)

result = pcl.Simulation(config, particles=particles).run(source)
print(result.n_particles)
print(result.cross_sections)
```

For near-field slices, multi-source runs, imported T matrices, dipoles, and
HDF5 output, see [`docs/quickstart.md`](docs/quickstart.md) and
[`docs/workflows.md`](docs/workflows.md).

## Examples and scaling harnesses

The repository includes small examples and general-purpose benchmark drivers:

- `examples/minimal_pyceles_demo.py` demonstrates mixed particles, CuPy,
  near/far fields, HDF5 output, and dipole diagnostics.
- `examples/source_showcase_demo.py` collects source families.
- `notebooks/01_celes_main_replication.ipynb` reproduces the `CELES_main.m`-style
  workflow.
- `examples/run_pairwise_mlfmm_scaling_benchmark.py` measures finite-cluster
  pairwise and MLFMM scaling. Its default ladder is `2**10` through `2**20`;
  use `--powers-of-two-range` or `--n-values` for a smaller run. Pairwise
  coupling becomes memory-bound well before the largest MLFMM cases; the full
  ladder is intended for a GPU with at least 8GB of memory.
- `examples/run_ewald_rayleigh_scaling_benchmark.py` measures the two periodic
  growth families with a complete preflight and resumable JSON records.

Validation strategy and external-reference entry points are described in
[`docs/validation.md`](docs/validation.md). Performance guidance, memory
policy, and scaling interpretation are in
[`docs/performance.md`](docs/performance.md).

## AI-assisted development

OpenAI GPT-5.6 Luna was used for assistance with code implementation, refactoring,
debugging, testing, and documentation editing. The physical formulations, validation
criteria and benchmarking protocols are defined by the authors, who reviewed all
generated changes and remain responsible for the software, its numerical results,
and scientific claims.

## Acknowledgment and license

This project would not be possible without the frameworks established by CELES
and SMUTHI and the work of their authors and contributors. pyceles is released
under the MIT license.
