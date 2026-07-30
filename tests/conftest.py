"""Shared pytest policy for the test suite."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from pyceles._optional import import_cupy

REPO_ROOT = Path(__file__).resolve().parent.parent

SUITE_MARKS = {
    "unit": pytest.mark.unit,
    "physics": pytest.mark.physics,
    "regression": pytest.mark.regression,
    "io": pytest.mark.io,
}


def _relative_test_path(item: pytest.Item) -> str:
    return item.path.resolve().relative_to(REPO_ROOT).as_posix()


@lru_cache(maxsize=1)
def _require_cupy_runtime() -> tuple[Any, Any]:
    try:
        cupy, cupyx_sparse_linalg = import_cupy()
    except RuntimeError as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"CuPy is unavailable: {exc}")
    try:
        _ = cupy.asarray([0.0], dtype=cupy.float32).sum().get()
        cupy.cuda.Stream.null.synchronize()
    except Exception as exc:  # pragma: no cover - optional runtime availability
        pytest.skip(f"CuPy/CUDA runtime is unavailable: {exc}")
    return cupy, cupyx_sparse_linalg


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    for item in items:
        rel_path = _relative_test_path(item)
        fixture_names = getattr(item, "fixturenames", ())
        suite = Path(rel_path).parts[1]
        suite_mark = SUITE_MARKS.get(suite)
        if suite_mark is not None:
            item.add_marker(suite_mark)

        if "tmp_path" in fixture_names:
            item.add_marker(pytest.mark.filesystem)
        if "gpu" in item.keywords:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def _gpu_compilation_cache_env(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    if not any("gpu" in item.keywords for item in request.session.items):
        yield
        return

    # Keep compilation artifacts inside one pytest-owned session root. CuPy
    # and CUDA caches must be shared across real-device tests; assigning fresh
    # cache directories per test forces repeated JIT work.
    root = tmp_path_factory.mktemp("cupy-compilation-cache")
    cupy_cache_dir = root / "cupy-cache"
    cuda_cache_dir = root / "cuda-cache"
    cupy_cache_dir.mkdir()
    cuda_cache_dir.mkdir()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("CUPY_CACHE_DIR", str(cupy_cache_dir))
    monkeypatch.setenv("CUDA_CACHE_PATH", str(cuda_cache_dir))

    yield
    monkeypatch.undo()


@pytest.fixture(autouse=True)
def _require_gpu_runtime_for_marked_tests(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    _gpu_compilation_cache_env: None,
) -> None:
    if "gpu" not in request.node.keywords:
        return

    # Scratch state remains isolated per test even though compiled kernels are
    # cached for the whole session.
    root = tmp_path_factory.mktemp("cupy-runtime")
    tmp_dir = root / "tmp"
    tmp_dir.mkdir()
    monkeypatch.setenv("TMP", str(tmp_dir))
    monkeypatch.setenv("TEMP", str(tmp_dir))
    monkeypatch.setenv("TMPDIR", str(tmp_dir))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_dir))
    _require_cupy_runtime()


@pytest.fixture(scope="session")
def cupy_runtime(_gpu_compilation_cache_env: None) -> tuple[Any, Any]:
    return _require_cupy_runtime()
