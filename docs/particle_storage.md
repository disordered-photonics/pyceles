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
indices, and one typed payload per archetype. The loader still accepts the
legacy descriptor-per-instance `v1` schema and normalizes it to a
`ParticleCollection`.
