# Current limitations

This page records current limitations.

## Physical and geometrical scope

- The main homogeneous-medium solver path currently assumes real positive host refractive index `n_medium`.
- All particles in one simulation currently share the same `lmax`.
- T-matrix superposition assumes disjoint particle circumscribing spheres. This is enforced by default through `Simulation` geometry checks.
- At the moment, this restriction is especially limiting for close spheroids because their circumscribing spheres can in principle overlap even when physical spheroids do not.

## Particle archetypes and scaling

- Geometry and prepared single-body storage scale with the number of unique
  archetypes, not only with particle count. Repeated layered spheres,
  spheroids, and custom dense T matrices therefore share storage and prepared
  operators just as repeated homogeneous spheres do.
- An orientation is currently part of the solver-facing spheroid archetype
  because the fallback stores a lab-frame dense block. A geometry with
  effectively unique orientations can therefore have as many prepared blocks
  as particles. A body-frame/block-sparse rotation representation is the
  intended extension for that workload.
- Fully unique arbitrary dense T matrices are inherently `O(N * Nm^2)` unless a
  future backend provides additional block-sparse, low-rank, or on-the-fly
  structure.

## Spheroids

- Homogeneous axisymmetric spheroids are supported with aligned/rotated spherical-basis T-matrix blocks.
- Internal-field evaluation inside spheroids is covered.
- Exterior near-field evaluation is still unreliable for points that lie inside a spheroid's circumscribing sphere but outside the physical spheroid, because the current default path uses the outgoing spherical SVWF expansion there.
- Exploratory surface-integral, arbitrary-precision, and spheroidal-shell postprocessing variants have been investigated but are not yet ready to replace the default path.

## Dipoles

- Local dipole sources are supported in the homogeneous host medium.
- Dipole-driven local absorbed-power diagnostics are available.
- Embedded dipoles inside particles are not supported in the main solver path yet.
- Dipole near-field initial fields mask exact dipole-center grid hits as `NaN`.

## Periodic workflows

Periodic boundary conditions remain experimental.

Current tested scope:

- homogeneous rectangular 2D lattices,
- plane-wave excitation,
- NumPy and CuPy `complex64`/`complex128` periodic Ewald paths,
- opt-in NumPy/CuPy hybrid exact-near/Rayleigh-far repeated applies for
  vertically extended cells,
- selected local-SVWF near-field workflows for homogeneous spheres,
- direct dense validation paths for controlled cases.

Not production-ready yet:

- non-rectangular lattices,
- reduced-cell local sources,
- mixed-precision periodic production runs,
- periodic MLFMM coupling.

The hybrid Rayleigh method is not a general cure for dense planar cells. Its
exact-near cache scales with the number of directed non-self pairs inside the
vertical band, which remains quadratic when most particles share nearly the
same height. CuPy can keep an oversized compact cache in host memory and stream
it through a bounded device buffer, but this only moves the GPU-residency wall;
it does not change the host-memory or per-matvec transfer scaling. Its automatic
reciprocal truncation is conservative but heuristic; scientific runs should
sweep `rayleigh_z_cut` or `rayleigh_reciprocal_shells` and compare
representative cases with exact Ewald.

Periodic in-slab near-field evaluation supports both `method="ewald"` and
`method="rayleigh"`. The hybrid path retains exact Ewald local-SVWF evaluation
for source-point pairs inside the vertical band and uses reciprocal Rayleigh
scans only for vertically distant sources. It therefore inherits the same
limitation as the hybrid solve: a dense same-height point/source population can
leave the exact-near work effectively quadratic.

Dense/direct periodic validation can remain memory-sensitive on small GPUs. Even when matrices are prepared in Fortran order for CuPy LU factorization, dense matrices, LU workspace, pivots, and cached W blocks can all be live at the same time.

## MLFMM backend

- The current MLFMM backend is high-frequency oriented; see
  [performance.md](performance.md) for the numerical rationale and benchmarks.
- The hierarchy is uniform-depth rather than adaptive.
- Low-frequency/static stabilization and switching are out of scope for the current implementation.
- CuPy MLFMM repeated apply reuses CPU-prepared hierarchy/build data and rebuilds device-resident state from compact host caches as needed.
- `compute_dtype="complex64"` affects exact-near work, while sampled-far MLFMM interactions currently remain `complex128` on both NumPy and CuPy paths.

## GPU backend

- The GPU backend is optional and depends on a working CuPy/CUDA environment.
- Not every postprocessing stage has a CuPy implementation; unsupported stages may fall back to NumPy reference paths.
- The direct CuPy backend is fast for moderate direct pairwise problems, but direct O(N^2) coupling is not the long-term large-N scaling path.
- CuPy/driver/library combinations can affect runtime behavior, memory use, and diagnostic-tool noise.

## Public API stability

pyceles is pre-1.0. The intended public surface is:

- `Simulation`, `SimulationConfig`, and result objects,
- particle descriptors such as `Sphere`, `PECSphere`, `LayeredSphere`, and
  `Spheroid`,
- source descriptors such as `PlaneWave`, Gaussian/Laguerre-Gaussian/Bessel sources, SLM wrappers, and dipole sources,
- HDF5 save/load helpers,
- near-field and far-field user-facing helpers.

Lower-level modules under `pyceles.core`, `pyceles.linear`, and implementation-specific operator modules may change without a long deprecation period until the API is declared stable.
