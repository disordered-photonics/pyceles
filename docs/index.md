# pyceles documentation

pyceles is a Python implementation of CELES-style many-particle electromagnetic
scattering with the T-matrix method. The project is GPU-oriented: it keeps a
NumPy/SciPy reference implementation for portability and validation, while the
main scalable finite and periodic solver paths use CuPy acceleration.

The documentation in this folder is deliberately small. It collects stable,
user-facing material that would otherwise make the root README harder to scan.

For the current feature inventory, start with
[capabilities.md](capabilities.md). For technical boundaries, see
[limitations.md](limitations.md).

## Suggested reading order

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Capabilities](capabilities.md)
- [API map](api.md)
- [Workflow notes](workflows.md)
- [Particle storage](particle_storage.md)
- [Performance notes](performance.md)
- [Validation and reproducibility](validation.md)
- [Current limitations](limitations.md)
- [Related works](references.md)

These pages are plain Markdown and are intended to remain usable without a
documentation build tool.
