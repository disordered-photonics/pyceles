"""Tests for shared CuPy allocator policy without requiring a CUDA runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyceles._cupy_memory import cupy_allocator_snapshot, guarded_device_limit_bytes

_GIB = 1024**3
_MIB = 1024**2

pytestmark = pytest.mark.fake_gpu


class _FakeRuntime:
    def __init__(self, *, free_bytes: int, total_bytes: int) -> None:
        self.free_bytes = int(free_bytes)
        self.total_bytes = int(total_bytes)

    def memGetInfo(self) -> tuple[int, int]:
        return self.free_bytes, self.total_bytes


class _FakePool:
    def __init__(
        self,
        *,
        runtime: _FakeRuntime,
        used_bytes: int,
        total_bytes: int,
        limit_bytes: int,
    ) -> None:
        self.runtime = runtime
        self.used = int(used_bytes)
        self.total = int(total_bytes)
        self.limit = int(limit_bytes)
        self.set_calls = 0
        self.trim_calls = 0

    def used_bytes(self) -> int:
        return self.used

    def total_bytes(self) -> int:
        return self.total

    def get_limit(self) -> int:
        return self.limit

    def set_limit(self, *, size: int) -> None:
        self.limit = int(size)
        self.set_calls += 1

    def free_all_blocks(self) -> None:
        self.runtime.free_bytes += self.total - self.used
        self.total = self.used
        self.trim_calls += 1


class _FakeCuPy:
    def __init__(self, runtime: _FakeRuntime, pool: _FakePool) -> None:
        self.cuda = SimpleNamespace(runtime=runtime)
        self._pool = pool

    def get_default_memory_pool(self) -> _FakePool:
        return self._pool


def test_guarded_device_limit_uses_bounded_physical_memory_reserve() -> None:
    assert guarded_device_limit_bytes(1 * _GIB) == 1 * _GIB - 256 * _MIB
    assert guarded_device_limit_bytes(8 * _GIB) == 8 * _GIB - 512 * _MIB
    assert guarded_device_limit_bytes(32 * _GIB) == 31 * _GIB


def test_cupy_allocator_preserves_smaller_existing_pool_limit() -> None:
    runtime = _FakeRuntime(free_bytes=5 * _GIB, total_bytes=8 * _GIB)
    pool = _FakePool(
        runtime=runtime,
        used_bytes=1 * _GIB,
        total_bytes=2 * _GIB,
        limit_bytes=6 * _GIB,
    )

    snapshot = cupy_allocator_snapshot(_FakeCuPy(runtime, pool), apply_pool_limit=True)

    assert snapshot.effective_device_limit_bytes == 6 * _GIB
    assert snapshot.pool_limit_applied is False
    assert pool.set_calls == 0
    assert pool.trim_calls == 0


def test_cupy_allocator_trims_fragmented_cache_for_large_fresh_block() -> None:
    runtime = _FakeRuntime(free_bytes=200 * _MIB, total_bytes=8 * _GIB)
    pool = _FakePool(
        runtime=runtime,
        used_bytes=4 * _GIB,
        total_bytes=7 * _GIB + 384 * _MIB,
        limit_bytes=7 * _GIB + 512 * _MIB,
    )

    snapshot = cupy_allocator_snapshot(
        _FakeCuPy(runtime, pool),
        apply_pool_limit=True,
        required_fresh_allocation_bytes=300 * _MIB,
    )

    assert pool.trim_calls == 1
    assert snapshot.pool_trimmed_for_fragmentation is True
    assert snapshot.pool_trimmed_to_limit is False
    assert snapshot.guaranteed_fresh_allocation_bytes >= 300 * _MIB
