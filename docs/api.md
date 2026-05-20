# API map

This page is a curated map of the intended public API. It is not generated from
docstrings yet, and it intentionally does not list every importable internal
module.

## Simulation workflow

Most user workflows start from:

- `pyceles.SimulationConfig`
- `pyceles.Simulation`
- `pyceles.SimulationResult`
- `pyceles.MultiSourceSimulationResult`
- `pyceles.SolvedSourcesResult`

The high-level pattern is:

```python
import pyceles as pcl

config = pcl.SimulationConfig(...)
result = pcl.Simulation(config, particles=[...]).run()
```

## Particles

Particle descriptors are passed to `Simulation`:

- `pyceles.Sphere`
- `pyceles.LayeredSphere`
- `pyceles.Spheroid`
- `pyceles.spheres_from_arrays`
- `pyceles.layered_spheres_from_arrays`
- `pyceles.spheroids_from_arrays`

## Sources

Common source descriptors include:

- `pyceles.PlaneWave`
- `pyceles.GaussianBeam`
- `pyceles.LaguerreGaussianBeam`
- `pyceles.FocusedLaguerreGaussianBeam`
- `pyceles.BesselBeam`
- `pyceles.SLMSource`
- `pyceles.DipoleSource`
- `pyceles.DipoleCollection`

## Postprocessing

Near-field helpers:

- `pyceles.compute_near_field`
- `pyceles.compute_near_field_slice`
- `pyceles.compute_periodic_near_field`
- `pyceles.compute_periodic_near_field_slice`
- `pyceles.mix_near_field_components`
- `pyceles.mix_near_field_slices`

Dipole diagnostics:

- `pyceles.compute_dipole_power_ldos`
- `pyceles.compute_dipole_ldos_enhancement`

Periodic far-field results are exposed through:

- `pyceles.PeriodicFarFieldPayload`

## Periodic descriptors

Experimental periodic workflows use:

- `pyceles.RectangularLattice2D`
- `pyceles.PeriodicSpec`
- `pyceles.PeriodicOptions`

## I/O

Workflow-level HDF5 helpers:

- `pyceles.save_simulation_h5`
- `pyceles.load_simulation_h5`

## Implementation modules

Modules under `pyceles.core`, `pyceles.linear`, `pyceles.simulation`, and
`pyceles.postprocessing` are importable because pyceles is a Python package, but
not every internal module is a stable user-facing contract. In particular,
operator implementations, CuPy RawKernel helpers, cache objects, and low-level
periodic kernels should be treated as implementation details unless they are
documented here or in the root README.
