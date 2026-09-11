# pyceles

pyceles is a Python reimplementation of the MATLAB CELES package for
electromagnetic simulation of large particle ensembles with the T-matrix method.
It keeps a NumPy/SciPy reference implementation and adds optional CuPy
acceleration for selected direct, MLFMM, postprocessing, and experimental
periodic workflows.

The project focuses on:

- CELES-compatible conventions and reproducible numerical workflows,
- a clean NumPy/SciPy reference implementation,
- performance through vectorization, caching, and optional CuPy kernels,
- the same `A = I - T W` operator structure across CPU and GPU backends,
- explicit precision control through `compute_dtype` and `accum_dtype`.

## Acknowledgment

This project would not be possible without the frameworks established by
[CELES](https://github.com/disordered-photonics/celes) and
[SMUTHI](https://gitlab.com/AmosEgel/smuthi), and the work of their authors and
contributors. Users of pyceles are referred to the publications listed in the
CELES and SMUTHI repositories, and those papers can be cited when pyceles is
used in scientific work.

## Documentation

The root README is now the main project overview. Longer-form Markdown notes are
available under [`docs/`](docs/):

- [capabilities](docs/capabilities.md): detailed feature inventory,
- [quickstart](docs/quickstart.md): minimal end-to-end example,
- [API map](docs/api.md): curated public entry points,
- [workflows](docs/workflows.md): polarization, multi-source, dipole, HDF5, and solver notes,
- [performance](docs/performance.md): benchmark snapshots and optimization notes,
- [validation](docs/validation.md): tests and external-reference scripts,
- [limitations](docs/limitations.md): current technical boundaries,
- [references](docs/references.md): related literature.

The docs are plain Markdown for now. Sphinx/MyST can be added later if the
project adopts generated API pages or hosted HTML documentation.

## What works today

Current tested capabilities include:

- CELES-compatible VSWF indexing, Wigner-3j tables, translation coefficients,
  and plane-wave/Gaussian incident-field machinery,
- explicit particle descriptors for homogeneous, layered, axisymmetric, and
  perfect-conductor particles, plus uniform compact instance/archetype storage,
- plane waves, structured beams, angular-spectrum SLM wrappers, local dipoles,
  and dipole collections,
- direct pairwise coupling on NumPy and CuPy,
- high-frequency MLFMM coupling for large sphere clusters on NumPy and CuPy
  repeated-apply paths,
- native CuPy Krylov solvers, including GMRES-family methods, BiCGSTAB,
  harmonic GCRO-DR, and block-GMRES for multi-RHS runs,
- near-field and far-field postprocessing, including component-resolved
  `initial`, `scattered`, `internal`, and `total` near-field maps,
- local absorption and dipole power/LDOS diagnostics,
- HDF5 save/load workflows,
- experimental rectangular 2D periodic workflows with diffraction-order
  payloads, `R/T/A`, periodic near-field slices, and an opt-in hybrid
  exact-near/Rayleigh-far repeated-apply operator for vertically extended cells.

See [docs/capabilities.md](docs/capabilities.md) for the full feature inventory.

## Current limits

pyceles is pre-1.0, and some boundaries are still intentional:

- particles in one simulation currently share the same `lmax`,
- homogeneous-medium workflows currently assume a real host refractive index,
- T-matrix superposition requires disjoint particle circumscribing spheres by
  default,
- embedded dipoles inside particles are not supported in the main solver path,
- periodic workflows are experimental and currently limited to homogeneous
  rectangular 2D lattices with plane-wave excitation,
- spheroid exterior near fields remain unreliable for points inside the
  circumscribing sphere but outside the physical particle.

See [docs/limitations.md](docs/limitations.md) for more detail.

## Future nice-to-have features and workflows

These are reminders for future development, not a release commitment:

- import dense T-matrix data from external T-matrix databases,
- optical force and torque postprocessing,
- spectral-sweep workflows for dispersion studies and approximate time-domain
  reconstruction,
- nonlinear double-pass workflows, such as fundamental-field solves followed by
  second-harmonic source construction,
- basic layered-media support, starting from a single planar interface between
  two homogeneous half-spaces,
- a vendor-neutral accelerator backend, such as a PyOpenCL-style path, once the
  useful CuPy kernel boundaries are clearer.

## Install

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

For contributors:

```bash
python -m pip install -e .[dev]
```

For GPU work, install a CuPy wheel matching your CUDA runtime when possible, then
install pyceles:

```bash
python -m pip install cupy-cuda12x
python -m pip install -e .
```

or, on CUDA 13:

```bash
python -m pip install cupy-cuda13x
python -m pip install -e .
```

The repository also exposes a `cupy` extra:

```bash
python -m pip install -e .[cupy]
```

See [docs/installation.md](docs/installation.md) for installation and local
quality-check notes.

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
    polar_angle=0.0,
    azimuthal_angle=0.0,
)

config = pcl.SimulationConfig(
    wavelength=550.0,
    n_medium=1.0 + 0j,
    lmax=3,
    solver_method="bicgstab",
    solver_rtol=1e-6,
)

result = pcl.Simulation(config, particles=particles).run(source)
print(result.n_particles)
print(result.cross_sections)
```

See [docs/quickstart.md](docs/quickstart.md) for near-field and HDF5 examples.

## Examples and benchmarks

Useful entry points:

- `examples/minimal_pyceles_demo.py`: mixed particles, CuPy direct solve,
  near/far field, HDF5 output, and dipole LDOS map,
- `notebooks/01_celes_main_replication.ipynb`: replication of the original
  `CELES_MAIN.m` workflow,
- `examples/profile_pyceles_phases.py`: finite-cluster profiling,
- `examples/profile_pyceles_periodic_phases.py`: rectangular periodic profiling,
- `examples/run_mstm_pyceles_cluster_benchmark.py`: finite-cluster comparison
  against MSTM,
- `examples/run_mstm_pyceles_periodic_benchmark.py`: periodic comparison
  against MSTM.

Benchmark details and representative timings are kept in
[docs/performance.md](docs/performance.md) and
[docs/validation.md](docs/validation.md).

## License

MIT
