# Installation

pyceles requires Python 3.12 or newer and uses a `src/`-layout package with
project metadata in `pyproject.toml`. The current release metadata lists
CPython 3.12, 3.13, and 3.14.

## Development checkout

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e .[dev]
```

This installs the code in editable mode and keeps imports synchronized with the
working tree.

## Optional notebooks

```bash
python -m pip install -e .[notebooks]
```

Then open:

```text
notebooks/01_celes_main_replication.ipynb
```

The notebook reproduces the original CELES main workflow using the bundled CELES
sphere-parameter example.

## Optional CuPy backend

pyceles exposes a `cupy` optional dependency, but practical GPU installations
often work best when CuPy is installed through the wheel matching the local CUDA
runtime.

Typical examples are:

```bash
python -m pip install cupy-cuda12x
python -m pip install -e .
```

or, on a CUDA 13 environment:

```bash
python -m pip install cupy-cuda13x
python -m pip install -e .
```

The repository metadata also contains:

```bash
python -m pip install -e .[cupy]
```

Use whichever route matches the CUDA runtime and CuPy packaging available on the
machine.

## Recommended local checks

For contributors, the most useful local checks are:

```bash
python -m pre_commit run --all-files
python -m pytest -q
```

GPU-specific tests require a working CUDA/CuPy installation:

```bash
python -m pytest -q -m gpu
```

Documentation is maintained as Markdown under `docs/` and does not require an
additional documentation tool.
