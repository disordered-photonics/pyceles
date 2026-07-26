"""Shared CuPy allocator and guarded-device-memory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MIB = 1024**2
_DEVICE_GUARD_FRACTION = 1.0 / 16.0
_MIN_DEVICE_GUARD_BYTES = 256 * _MIB
_MAX_DEVICE_GUARD_BYTES = 1024 * _MIB


@dataclass(frozen=True)
class CuPyAllocatorSnapshot:
    """CuPy allocator counters after optional guarded-pool enforcement."""

    raw_free_bytes: int
    raw_total_bytes: int
    pool_used_bytes: int
    pool_total_bytes: int
    pool_free_bytes: int
    pool_limit_bytes: int
    effective_device_limit_bytes: int
    active_headroom_bytes: int
    guaranteed_fresh_allocation_bytes: int
    pool_limit_applied: bool
    pool_trimmed_to_limit: bool
    pool_trimmed_for_fragmentation: bool


def cupy_allocator_memory_info(cupy: Any) -> dict[str, int]:
    """Return raw and CuPy-pool-aware device-memory counters."""
    pool = cupy.get_default_memory_pool()
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    pool_used_bytes = int(pool.used_bytes())
    pool_total_bytes = int(pool.total_bytes())
    pool_cached_bytes = max(0, pool_total_bytes - pool_used_bytes)
    effective_free_bytes = min(int(total_bytes), int(free_bytes) + int(pool_cached_bytes))
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "pool_used_bytes": int(pool_used_bytes),
        "pool_total_bytes": int(pool_total_bytes),
        "pool_cached_bytes": int(pool_cached_bytes),
        "effective_free_bytes": int(effective_free_bytes),
    }


def cupy_pool_limit_bytes(pool: Any) -> int:
    """Return the active CuPy pool limit, or zero when none is configured."""
    get_limit = getattr(pool, "get_limit", None)
    if get_limit is None:
        return 0
    try:
        return int(get_limit())
    except TypeError:
        return int(get_limit(device_id=None))


def set_cupy_pool_limit(pool: Any, *, size: int) -> None:
    """Set a CuPy pool limit across supported CuPy API variants."""
    try:
        pool.set_limit(size=int(size))
    except TypeError:
        pool.set_limit(size=int(size), device_id=None)


def guarded_device_limit_bytes(total_bytes: int) -> int:
    """Return a device-residency ceiling below physical GPU memory.

    The guard prevents Windows/WDDM from making an automatic policy appear to
    fit by spilling CuPy allocations into shared host memory. Dedicated Linux
    devices then follow the same decision boundary instead of failing later.
    """
    total = int(total_bytes)
    device_guard = min(
        _MAX_DEVICE_GUARD_BYTES,
        max(_MIN_DEVICE_GUARD_BYTES, int(float(total) * _DEVICE_GUARD_FRACTION)),
    )
    return max(1, total - int(device_guard))


def guaranteed_fresh_allocation_bytes(
    *, raw_free_bytes: int, pool_total_bytes: int, pool_limit_bytes: int
) -> int:
    """Return bytes available without reusing a cached pool block."""
    return max(
        0,
        min(
            int(raw_free_bytes),
            int(pool_limit_bytes) - int(pool_total_bytes),
        ),
    )


def cupy_allocator_snapshot(
    cupy: Any,
    *,
    apply_pool_limit: bool,
    required_fresh_allocation_bytes: int = 0,
) -> CuPyAllocatorSnapshot:
    """Snapshot allocator state and optionally enforce the guarded pool limit.

    Cached pool blocks are released only when the configured limit is already
    exceeded or when the next known large allocation cannot be served from
    fresh driver/pool headroom. This preserves normal pool reuse while avoiding
    false aggregate-headroom assumptions for one large contiguous allocation.
    """
    pool = cupy.get_default_memory_pool()
    raw_free, raw_total = cupy.cuda.runtime.memGetInfo()
    raw_total = int(raw_total)
    guarded_limit = guarded_device_limit_bytes(raw_total)
    existing_limit = cupy_pool_limit_bytes(pool)
    effective_limit = guarded_limit if existing_limit <= 0 else min(existing_limit, guarded_limit)
    pool_limit_applied = False
    if bool(apply_pool_limit) and (
        existing_limit <= 0 or int(existing_limit) > int(effective_limit)
    ):
        set_cupy_pool_limit(pool, size=int(effective_limit))
        pool_limit_applied = True

    pool_used = int(pool.used_bytes())
    pool_total = int(pool.total_bytes())
    pool_free = max(0, pool_total - pool_used)
    guaranteed_fresh = guaranteed_fresh_allocation_bytes(
        raw_free_bytes=int(raw_free),
        pool_total_bytes=int(pool_total),
        pool_limit_bytes=int(effective_limit),
    )
    trim_to_limit = pool_total > int(effective_limit)
    trim_for_fragmentation = (
        int(required_fresh_allocation_bytes) > int(guaranteed_fresh) and pool_free > 0
    )
    pool_trimmed_to_limit = False
    pool_trimmed_for_fragmentation = False
    if (
        bool(apply_pool_limit)
        and pool_free > 0
        and (bool(trim_to_limit) or bool(trim_for_fragmentation))
    ):
        pool.free_all_blocks()
        pool_trimmed_to_limit = bool(trim_to_limit)
        pool_trimmed_for_fragmentation = bool(trim_for_fragmentation)
        raw_free, raw_total = cupy.cuda.runtime.memGetInfo()
        raw_total = int(raw_total)
        pool_used = int(pool.used_bytes())
        pool_total = int(pool.total_bytes())
        pool_free = max(0, pool_total - pool_used)
        guaranteed_fresh = guaranteed_fresh_allocation_bytes(
            raw_free_bytes=int(raw_free),
            pool_total_bytes=int(pool_total),
            pool_limit_bytes=int(effective_limit),
        )

    active_headroom = max(0, int(effective_limit) - int(pool_used))
    return CuPyAllocatorSnapshot(
        raw_free_bytes=int(raw_free),
        raw_total_bytes=raw_total,
        pool_used_bytes=pool_used,
        pool_total_bytes=pool_total,
        pool_free_bytes=pool_free,
        pool_limit_bytes=int(effective_limit),
        effective_device_limit_bytes=int(effective_limit),
        active_headroom_bytes=int(active_headroom),
        guaranteed_fresh_allocation_bytes=int(guaranteed_fresh),
        pool_limit_applied=pool_limit_applied,
        pool_trimmed_to_limit=pool_trimmed_to_limit,
        pool_trimmed_for_fragmentation=pool_trimmed_for_fragmentation,
    )


__all__ = [
    "CuPyAllocatorSnapshot",
    "cupy_allocator_memory_info",
    "cupy_allocator_snapshot",
    "cupy_pool_limit_bytes",
    "guaranteed_fresh_allocation_bytes",
    "guarded_device_limit_bytes",
    "set_cupy_pool_limit",
]
