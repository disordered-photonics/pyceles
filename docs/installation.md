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

CuPy is optional and is not included in pyceles's runtime dependencies. The
correct CuPy package depends on the CUDA runtime available on the machine, so
install the matching pre-built wheel explicitly.

Typical examples are:

```bash
python -m pip install pyceles cupy-cuda12x
```

or, on a CUDA 13 environment:

```bash
python -m pip install pyceles cupy-cuda13x
```

For an editable checkout, install pyceles separately after installing the
matching CuPy wheel:

```bash
python -m pip install -e .
```

Do not install the generic source-build `cupy` package alongside one of these
CUDA-specific distributions; use exactly one CuPy distribution per environment.

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
