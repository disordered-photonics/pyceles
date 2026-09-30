# Quickstart

This page shows a minimal pyceles workflow: define particles, define a source, configure a simulation, solve, and optionally evaluate a near-field slice.

## A small sphere-cluster run

```python
import numpy as np
import pyceles as pcl

particles = [
    pcl.Sphere(
        position=(-180.0, 0.0, 0.0),
        radius=110.0,
        refractive_index=1.5 + 0.0j,
    ),
    pcl.Sphere(
        position=(180.0, 0.0, -60.0),
        radius=90.0,
        refractive_index=1.5 + 0.0j,
    ),
    pcl.Sphere(
        position=(20.0, 0.0, 110.0),
        radius=100.0,
        refractive_index=1.5 + 0.0j,
    ),
]

source = pcl.PlaneWave(
    wavelength=550.0,
    medium_n=1.0 + 0.0j,
    polarization="TE",
    polar_angle=0.0,
    azimuthal_angle=0.0,
    amplitude=1.0,
)

config = pcl.SimulationConfig(
    wavelength=550.0,
    n_medium=1.0 + 0.0j,
    lmax=3,
    solver_method="bicgstab",
    solver_rtol=1e-6,
    verbose=True,
)

simulation = pcl.Simulation(config, particles=particles)
result = simulation.run(source)

print(result.n_particles)
if result.cross_sections is not None:
    print(result.cross_sections.extinction, result.cross_sections.scattering)
```

Plane waves use cross-section diagnostics. Finite-beam power fractions are
reported separately when the source has finite incident power.

The same high-level model can be moved to the CuPy direct backend by changing the operator backend and, when useful, the precision policy:

```python
config_gpu = pcl.SimulationConfig(
    wavelength=550.0,
    n_medium=1.0 + 0.0j,
    lmax=3,
    operator_backend="cupy",
    coupling_backend="pairwise",
    solver_method="bicgstab",
    compute_dtype="complex64",
    accum_dtype="complex128",
    verbose=True,
)
```

Use this only in an environment where CuPy and a compatible CUDA runtime are available.

## Near-field slice

After solving, a planar slice can be evaluated with:

```python
near = pcl.compute_near_field_slice(
    result,
    axis_0_min=-300.0,
    axis_0_max=300.0,
    axis_1_min=-300.0,
    axis_1_max=300.0,
    dx=10.0,
    plane="y",
    plane_value=0.0,
    show_progress=True,
)

E_total, H_total = near.field_maps["total"]
print(E_total.shape, H_total.shape)
```

The `NearFieldSlice` object records the grid axes, the selected plane, and field maps for initial/scattered/internal/total components where available.
With `show_progress=True`, the high-level finite near-field API reports these
physical evaluation stages on one progress bar. Backend-specific low-level
kernels may use different internal work units; fused GPU launches are not split
solely to manufacture finer progress updates.

## HDF5 output

A solved run and a near-field slice can be saved with:

```python
from pathlib import Path

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
pcl.save_simulation_h5(result, near, out_dir / "quickstart.h5")
```

and loaded again with:

```python
loaded = pcl.load_simulation_h5(out_dir / "quickstart.h5")
```

## Existing richer examples

More complete examples can be found in:

- `examples/finite_cluster.py`: a compact 500-particle finite-cluster solve
  using MLFMM.
- `examples/periodic_cell.py`: a compact 500-particle periodic solve using
  Rayleigh coupling and diffraction-order power diagnostics.
- `examples/particles_demo.py`: the same 500-particle geometry represented by
  layered spheres, randomly oriented spheroids, and PEC spheres, with near-
  and far-field canvases for each representation.
- `examples/sources_demo.py`: a fixed source-family visualization workflow.
- `notebooks/01_celes_main_replication.ipynb`: CELES main example replication.

Treat these examples as a source of end-to-end usage patterns while the public API is still evolving.
