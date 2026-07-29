# Capabilities

This page keeps the detailed feature inventory out of the root README while
preserving the implementation notes that are useful for future documentation,
benchmark writeups, and paper drafting.

## Core conventions

- CELES-compatible VSWF indexing, with tau blocks contiguous and `m=-l..l`.
- Cached Wigner-3j implementation.
- CELES translation coefficients (`a5`/`b5` tables) with normalized Legendre
  recurrences.
- CELES plane-wave incident coefficients.
- O(N^2) reference matvec for `(I - T W) x = T b`.
- CELES-style precision policy through `compute_dtype` and `accum_dtype`.

## Sources

Current source support includes:

- plane waves,
- Gaussian and Laguerre-Gaussian beams,
- focused Laguerre-Gaussian beams,
- Bessel beams,
- SLM-modulated angular-spectrum sources,
- local electric dipoles and dipole collections.

Gaussian wavebundle support includes optimized normal-incidence kernels ported
from CELES, tilted-incidence source projection through angular spectra, and a
rotated-frame near-field initial-field fast path with a general alpha-beta
fallback through `force_general_initial_field=True`.

Physical source checks are capability based:

- finite-beam diagnostics call `has_finite_incident_power()`,
- built-in `PlaneWave` and plane-wave-limit beams are treated as infinite-power
  excitations and rejected for beam-power fractions,
- ideal infinite-power sources such as `BesselBeam` are excluded from
  beam-power fractions by the same policy,
- finite-beam power fractions are normalized by integrating the initial TE/TM
  plane-wave spectrum.

Finite-beam and periodic diagnostics share the immutable
`SimulationResult.power: PowerBalance` contract:

- `incident_power`, `reflected_power`, and `transmitted_power`,
- `local_absorbed_power` from the generic local exciting-field route
  `e = b + W x`, plus its per-particle decomposition,
- `flux_defect = incident_power - reflected_power - transmitted_power`,
- `closure_error = flux_defect - local_absorbed_power`,
- normalized `reflectance`, `transmittance`, `local_absorptance`,
  `flux_defect_fraction`, and `closure_error_fraction`.

The flux defect is not labeled absorption. For an inexact solve, the local
estimate includes the equation defect; a nonzero closure error then exposes
residual inconsistency between the local and far-field identities, including
operator approximation, quadrature, or
basis-truncation effects.

## Particles and geometry

`Simulation(config, particles=[...])` accepts explicit particle descriptors and
mixed supported particle families:

- `Sphere`,
- `PECSphere`,
- `LayeredSphere`,
- `Spheroid`.

`Simulation` normalizes these inputs into an immutable `ParticleCollection`.
`Simulation.n_particles` and `SimulationResult.n_particles` provide canonical
particle counts across descriptor and array-generated inputs.

By default, `Simulation` enforces disjoint circumscribing spheres, which is
required by the current T-matrix superposition workflow. The check can be
disabled with `check_circumscribing_sphere_overlap=False` for exploratory cases.

Array-generated geometries should use the canonical helpers:

- `spheres_from_arrays`,
- `pec_spheres_from_arrays`,
- `layered_spheres_from_arrays`,
- `spheroids_from_arrays`.

The helpers return `ParticleCollection` objects and can be combined with
`ParticleCollection.concatenate(...)`. Every particle family uses the same
instance/archetype storage contract: one position and compact archetype tag per
instance, plus one immutable descriptor per distinct archetype. Single-body
preparation retains one diagonal or dense operator per unique archetype rather
than per particle. See [particle_storage.md](particle_storage.md).

`PECSphere` uses the analytic perfect-conductor Mie limit instead of an
artificial large complex refractive index. Near-field points inside PEC spheres
are treated as particle-internal points with zero physical field.

## Spheroids

Axisymmetric spheroid support includes:

- homogeneous spheroid particle-local scattering blocks in the spherical SVWF
  basis,
- aligned and rotated spheroid T-matrix support in CELES ordering,
- internal-field evaluation inside spheroids,
- regression coverage against isolated-particle SMUTHI / NFMDS references.

For current spheroid near-field caveats, see [limitations.md](limitations.md).

## Coupling backends

The default many-body operator has the same top-level structure across backends:

```text
A = I - T W
```

where `T` is particle-local scattering and `W` is inter-particle coupling.

Supported coupling paths include:

- direct pairwise NumPy reference coupling,
- direct CuPy coupling with fused RawKernel `W @ x` for `complex64` and
  `complex128`,
- NumPy high-frequency MLFMM coupling for large sphere clusters,
- CuPy high-frequency MLFMM repeated apply for the same hierarchy plan,
- experimental NumPy and CuPy periodized MLFMM paths for rectangular 2D cells,
- experimental periodic Ewald coupling for rectangular 2D lattices.

## MLFMM scope

The current MLFMM backend is a high-frequency, matrix-free coupling path for
larger sphere clusters. It keeps exact near interactions on the resolved leaf
partition and accelerates well-separated far interactions through single-level
or multilevel directional operators. On the CuPy path, hierarchy preparation
still comes from the validated CPU reference plan while repeated applies run on
device.

See [performance.md](performance.md) for precision and runtime notes, and
[limitations.md](limitations.md) for current MLFMM boundaries.

## Near-field and far-field postprocessing

Near-field evaluation includes:

- scattered field,
- initial field,
- total field with internal-field replacement inside supported particles,
- a geometry-agnostic helper `compute_near_field`,
- a planar helper `compute_near_field_slice`,
- component maps for `initial`, `scattered`, `internal`, and `total`.

Far-field support includes:

- plane-wave pattern assembly,
- forward/backward power flux,
- canonical helpers returning `initial`, `scattered`, and `total` PWPs,
- differential scattering cross section `dC_sca/dOmega`,
- PWP-integrated cluster scattering cross section,
- extinction cross section,
- local absorption cross section `C_abs = C_abs_local`,
- explicit raw/closure diagnostics:
  - `C_abs_raw_diff = C_ext_raw - C_sca_raw`,
  - `Delta_closure = C_abs_raw_diff - C_abs_local`.

`plane_wave_cross_sections(...)` requires explicit `local_absorption`. The
legacy raw-difference fallback is opt-in through
`allow_raw_diff_fallback=True`. pyceles does not expose a coefficient-only
cluster `C_sca` helper.

## CuPy postprocessing coverage

When `postprocessing_backend="inherit"` and the solve backend is CuPy, pyceles
uses available CuPy postprocessing paths by default. Current CuPy slices include:

- scattered far-field SVWF-to-PWP assembly,
- scattered near-field,
- dominant Gaussian/general initial-field paths,
- homogeneous-sphere internal fields.

Layered-sphere and spheroid internal-field cases still fall back to the
NumPy reference implementation.

## Periodic workflows

Homogeneous rectangular 2D periodic-cell workflows currently include:

- `RectangularLattice2D`, `PeriodicSpec`, and `PeriodicOptions`,
- plane-wave Bloch source validation through incident `k_parallel`,
- periodic Ewald coupling with a direct-sum oracle for small checks,
- geometry-only Rayleigh/Wood anomaly diagnostics for artificial periodicity
  checks,
- periodic diffraction-order payloads on `SimulationResult.periodic`,
- reflected/transmitted diffraction-order amplitudes and `R/T/A` totals,
- periodic near-field slices:
  - exterior Rayleigh-order evaluation above/below the particle slab,
  - in-slab periodic local-SVWF evaluation outside/inside homogeneous spheres,
    including exact-near/Rayleigh-far acceleration when the solved periodic
    method is `"rayleigh"`,
- CuPy periodic Ewald coupling with optional explicit W-block caching,
- an opt-in NumPy/CuPy hybrid repeated-apply method through
  `PeriodicOptions(method="rayleigh")`: one shared exact periodic self block, a
  configurable exact-Ewald vertical near band, and reciprocal Rayleigh
  upward/downward scans for vertically separated pairs,
- experimental mesh-free NumPy and CuPy periodization of the existing MLFMM
  hierarchy through `coupling_backend="mlfmm"` with
  `PeriodicOptions(method="ewald")`. Ewald prepares coarse sampled lattice
  closures once on the CPU; finite nearby images are resolved by ordinary MLFMM
  descent and exact leaf interactions on the selected repeated-apply backend.

Periodic workflows remain experimental; current limits are documented in
[limitations.md](limitations.md), and performance/convergence knobs are discussed
in [performance.md](performance.md).
