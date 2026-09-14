# Particle storage and single-body operators

pyceles separates **particle instances** from **particle archetypes**.

A particle instance contributes a center and a compact archetype index. An
archetype is an immutable, position-independent descriptor containing the shape,
material, orientation, and any other metadata needed to prepare its single-body
operator. `ParticleCollection` stores:

```text
positions[N, 3]
archetype_indices[N]
archetypes[U]
```

where `N` is the number of instances and `U` is the number of distinct
position-independent descriptors. Explicit descriptor lists and all built-in
array constructors normalize to this same representation.

This is a uniform public model, not a requirement that every particle use the
same numerical kernel. During operator preparation, instances are grouped by
single-body representation and each group stores one operator per unique
archetype plus a local instance-to-operator map. Current representations are:

- diagonal mode data for homogeneous, PEC, and layered spheres;
- shared spherical-basis dense blocks for the current axisymmetric fallback;
- shared dense blocks supplied by custom particle factories.

Consequently, persistent geometry storage is approximately `O(N + U metadata)`
and persistent prepared single-body storage is `O(U operator_size)`, rather than
`O(N operator_size)`, whenever instances reuse archetypes. The CuPy diagonal
path consumes the map in a fused gather/multiply kernel and does not materialize
an expanded per-instance diagonal table. Shared dense blocks are applied per
unique operator without retaining an expanded `(N, Nm, Nm)` array.

This distinction is important for scalability claims. A million instances of
one layered-sphere archetype or one imported dense T matrix should not pay for a
million Python descriptors or a million stored T matrices. Conversely, a
million genuinely distinct dense lab-frame T matrices remain intrinsically
large; no storage container can compress them without additional structure.
Future representations can add body-frame rotations, block sparsity, low-rank
forms, or on-the-fly operators behind the same instance/archetype contract.

## Imported dense T matrices

`load_tmatrix_h5` reads the published spectral `.tmat.h5` representation and
returns one wavelength-selected, immutable dense block in CELES ordering. A
selection is mandatory for multi-wavelength files and is exact within floating
point storage tolerance; no nearest-neighbour wavelength is chosen:

```python
import pyceles as pcl

config = pcl.SimulationConfig(wavelength=550.0, n_medium=1.0)
data = pcl.load_tmatrix_h5("particle.tmat.h5", wavelength=config.wavelength)
particle = data.as_particle(
    position=(0.0, 0.0, 0.0),
    radius=300.0,
    wavelength=config.wavelength,
    n_medium=config.n_medium,
)
simulation = pcl.Simulation(config, particles=[particle])
```

pyceles remains unit-agnostic here. Wavelength axes retain their file length
unit, reciprocal-length axes retain the corresponding length unit, and
frequency/angular-frequency axes are represented as vacuum wavelength in
metres for selection and context checking. No simulation coordinate, radius,
or wavelength is converted automatically. The caller is responsible for using
one consistent numerical length convention across the simulation and all
imported matrices.

The standard file's electric/magnetic parity or positive/negative helicity
convention is translated at the I/O boundary, so callers do not need to
reorder modes or perform the basis conversion themselves. The
`radius` argument is the particle's circumscribing radius for overlap checks
and plotting; the imported file is otherwise treated as an opaque scatterer.
Consequently its interior near field is reported as unavailable (`NaN`) while
its enclosing sphere can still be drawn. Helicity-basis files and separate
incident/scattered mode sets are handled differently: helicity is converted to
the square parity form, while separate mode sets remain unsupported. Multiple
copies share the immutable block; use
`pyceles.core.rotate_svwf_tmatrix_block` once per desired orientation.

The file's `embedding` metadata contains relative permittivity and permeability
when supplied by the producer. Use `TMatrixData.embedding_refractive_index`
and `TMatrixData.validate_context(...)` to check the selected matrix against
the simulation wavelength and host index. The stored `wavelength_unit` is
informational; there is deliberately no simulation-wide unit registry or
automatic conversion layer. pyceles does not interpolate a spectrum or
silently choose a nearest wavelength; a missing or incompatible numerical
context causes `as_particle` to fail. Direct `TMatrixParticle` construction is
the explicit escape hatch when the caller is managing provenance separately.

An imported block is opaque with respect to hidden geometry and internal-field
material data, but it is not forward-only: the prepared dense particle group
applies the exact conjugate transpose of the stored block. Consequently LSQR
and other adjoint-based operator algorithms remain available when the coupling
backend has a matching adjoint. The import alone cannot supply derivatives with
respect to an unknown shape, material, or wavelength-dependent model, and it
cannot reconstruct the particle's interior field.

## Construction

Use the built-in array helpers when their schemas fit. Use
`ParticleCollection.from_archetypes(...)` for a generic shared-archetype
geometry:

```python
import numpy as np
import pyceles as pcl

archetypes = (
    pcl.LayeredSphere(
        position=(0.0, 0.0, 0.0),
        layer_radii=(40.0, 80.0),
        layer_refractive_indices=(1.8 + 0j, 1.5 + 0.01j),
    ),
)
positions = np.asarray([[0.0, 0.0, 0.0]], dtype=float)
particles = pcl.ParticleCollection.from_archetypes(
    positions=positions,
    archetypes=archetypes,
    archetype_indices=np.zeros(len(positions), dtype=np.uint8),
)
```

Indexing or iterating a collection materializes ordinary particle descriptors
for convenience. Performance-sensitive code should consume the columnar
properties and archetype groups instead.

## Persistence

HDF5 geometry schema `pyceles.particles.v2` stores positions, compact archetype
indices, and one typed payload per archetype. The loader accepts only this
canonical schema; older descriptor-per-instance files must be regenerated.
