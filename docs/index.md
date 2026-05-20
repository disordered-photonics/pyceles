# pyceles documentation

pyceles is a Python implementation of CELES-style many-particle electromagnetic
scattering with the T-matrix method. The project keeps a NumPy/SciPy reference
implementation while adding optional CuPy acceleration for selected solver and
postprocessing paths.

The documentation in this folder is deliberately small. It collects stable,
user-facing material that would otherwise make the root README harder to scan,
without committing the project to a full documentation site (yet).

For the current feature inventory, start with
[capabilities.md](capabilities.md). For technical boundaries, see
[limitations.md](limitations.md).

## Suggested reading order

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Capabilities](capabilities.md)
- [API map](api.md)
- [Workflow notes](workflows.md)
- [Performance notes](performance.md)
- [Validation and reproducibility](validation.md)
- [Current limitations](limitations.md)
- [Related works](references.md)

These pages are plain Markdown. They can be converted to a Sphinx/MyST site
later without making Sphinx a runtime dependency of pyceles.
