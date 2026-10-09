# Installation

pyceles requires Python 3.12 or newer and uses a `src/`-layout package with
project metadata in `pyproject.toml`. The current release metadata lists
CPython 3.12, 3.13, and 3.14.

## Published package

For a released version from PyPI:

```bash
python -m pip install -U pyceles
```

pyceles requires Python 3.12 or newer. For the GPU-first performance path,
install one CuPy wheel matching the CUDA major version available on the
machine; see [CuPy GPU backend](#cupy-gpu-backend) below.

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

## CuPy GPU backend

The NumPy/SciPy reference path is available without CUDA. The performance path
used for large finite, MLFMM, and periodic workloads is CuPy-based, so install
the matching pre-built wheel explicitly. Choose the wheel for the CUDA major
version available on the machine:

For CUDA 13.x:

```bash
python -m pip install pyceles cupy-cuda13x
```

For CUDA 12.x systems:

```bash
python -m pip install pyceles cupy-cuda12x
```

If the machine has an NVIDIA driver but no system-wide CUDA Toolkit, install
CuPy together with its matching CUDA component wheels through the `ctk` extra:

```bash
python -m pip install pyceles "cupy-cuda13x[ctk]"
# or: python -m pip install pyceles "cupy-cuda12x[ctk]"
```

This is a convenient fresh-environment setup; it does not remove the
requirement for a compatible NVIDIA driver. When a system CUDA Toolkit is
already installed, prefer the regular `cupy-cuda13x` or `cupy-cuda12x` wheel.
For the CUDA component compatibility matrix and optional libraries, see the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

For an editable checkout, install pyceles separately after installing the
matching CuPy wheel:

```bash
python -m pip install -e .
```

Do not install the generic source-build `cupy` package alongside one of these
CUDA-specific distributions; use exactly one CuPy distribution per environment.
The CuPy dependency is intentionally not bundled into pyceles because the
appropriate wheel is CUDA-generation-specific.

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
