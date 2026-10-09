"""Control-flow regressions; device arithmetic is exercised by GPU tests."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pyceles.core.operators.mlfmm_cupy as runtime


def test_fused_selected_receive_does_not_gather_unused_particle_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OriginalSchedule:
        def __getitem__(self, _key: Any) -> Any:
            raise AssertionError("Fused receive must not materialize a particle-index gather")

    schedule = OriginalSchedule()
    rows = np.array([0, 1], dtype=np.int32)
    group: Any = SimpleNamespace(
        leaf_ids=rows,
        occupancy=2,
        particle_indices=schedule,
        pair_deltas=np.zeros((2, 2, 3)),
    )
    tables: Any = SimpleNamespace(nmodes_out=6)
    calls = 0

    def matched(**_kwargs: Any) -> tuple[Any, Any]:
        return rows, rows

    def launch(**kwargs: Any) -> None:
        nonlocal calls
        assert kwargs["particle_indices"] is schedule
        np.testing.assert_array_equal(kwargs["row_indices"], rows)
        calls += 1

    monkeypatch.setattr(runtime, "_matched_sorted_rows", matched)
    monkeypatch.setattr(runtime, "_launch_leaf_otf_receive_fused", launch)
    runtime._receive_selected_leaf_boxes_to_particles(
        np.ones((2, 6, 1), dtype=np.complex128),
        selected_leaf_ids=rows,
        leaf_groups=(group,),
        leaf_apply_mode="on_the_fly",
        leaf_translation_tables=tables,
        leaf_otf_chunk_leaves=None,
        receive_adjoint_cache=None,
        nm=3,
        out=np.zeros((4, 3, 1), dtype=np.complex128),
        cupy=np,
    )
    assert calls == 1


def test_streamed_frontier_releases_projection_before_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous: weakref.ReferenceType[Any] | None = None
    projections = 0
    received = 0

    def ignored(**_kwargs: Any) -> None:
        pass

    def per_box(**_kwargs: Any) -> int:
        return 32

    def project(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal previous, projections
        assert previous is None or previous() is None
        result = np.ones((1, 6, 1), dtype=np.complex128)
        previous = weakref.ref(result)
        projections += 1
        return result

    def receive(incoming: Any, **_kwargs: Any) -> None:
        nonlocal received
        assert previous is not None and previous() is incoming
        received += 1

    monkeypatch.setattr(runtime, "_apply_same_level_far_streamed_chunk_group", ignored)
    monkeypatch.setattr(runtime, "_level_group_bytes_per_box", per_box)
    monkeypatch.setattr(runtime, "_directional_to_box_regular_cupy", project)
    monkeypatch.setattr(runtime, "_receive_selected_leaf_boxes_to_particles", receive)
    level: Any = SimpleNamespace(directional=object())
    runtime._apply_multilevel_frontier_streamed(
        levels=(level,),
        transfer_by_parent={},
        leaf_groups=(),
        leaf_apply_mode="on_the_fly",
        leaf_translation_tables=None,
        receive_adjoint_cache=None,
        leaf_otf_chunk_leaves=None,
        streamed_far_chunk_bytes_budget=1024,
        streamed_far_frontier_bytes_budget=1024,
        frontier_bytes_in_flight=0,
        level_idx=0,
        leaf_level=0,
        frontier_chunks=[
            (np.array([i], dtype=np.int32), np.zeros((1, 2, 1, 1), dtype=np.complex128))
            for i in range(2)
        ],
        box_nm_leaf=6,
        x_states=np.zeros((2, 3, 1), dtype=np.complex128),
        y_out=np.zeros((2, 3, 1), dtype=np.complex128),
        nm=3,
        nrhs=1,
        cupy=np,
        stream_stats=None,
    )
    assert projections == received == 2
    assert previous is not None and previous() is None
