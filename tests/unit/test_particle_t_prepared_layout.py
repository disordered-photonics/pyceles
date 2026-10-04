from __future__ import annotations

import numpy as np
import pytest

from pyceles.core.operators import single_body_cupy
from pyceles.core.operators.groups import DiagonalTGroup
from pyceles.core.operators.single_body import CompositeParticleTOperator


@pytest.mark.parametrize("gpu_class", (False, True))
def test_particle_t_partition_rejects_repeated_ids_within_one_group(gpu_class: bool) -> None:
    with pytest.raises(ValueError, match="unique"):
        if gpu_class:
            gpu_group = single_body_cupy.CuPyDiagonalTGroup(
                particle_indices=np.array([0, 1, 1]),
                T_M=np.ones((1, 2)),
                T_N=np.ones((1, 2)),
                T_diag=np.ones((1, 6)),
            )
            single_body_cupy.CuPyCompositeParticleTOperator(
                lmax=1, n_particles=2, groups=(gpu_group,)
            )
        else:
            cpu_group = DiagonalTGroup(
                particle_indices=np.array([0, 1, 1]),
                T_M=np.ones((1, 2)),
                T_N=np.ones((1, 2)),
                T_diag=np.ones((1, 6)),
            )
            CompositeParticleTOperator(lmax=1, n_particles=2, groups=(cpu_group,))


@pytest.mark.parametrize(
    "group_type", (single_body_cupy.CuPyDiagonalTGroup, single_body_cupy.CuPyDenseTGroup)
)
def test_cupy_t_group_rejects_out_of_bounds_operator_ids(group_type) -> None:
    kwargs = dict(particle_indices=np.arange(3), operator_indices=np.array([0, 2, 0]))
    if group_type is single_body_cupy.CuPyDiagonalTGroup:
        kwargs.update(T_M=np.ones((2, 2)), T_N=np.ones((2, 2)), T_diag=np.ones((2, 6)))
    else:
        kwargs["T_blocks"] = np.ones((2, 6, 6))
    with pytest.raises(IndexError, match="out-of-range"):
        group_type(**kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ("contiguous", "interleaved", "permuted"))
@pytest.mark.parametrize("nrhs", (1, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_cupy_particle_t_layout_is_prepared_once(
    cupy_runtime, monkeypatch, layout: str, nrhs: int, adjoint: bool
) -> None:
    cupy, _ = cupy_runtime
    first, second = {
        "contiguous": ([0, 1], [2, 3]),
        "interleaved": ([0, 2], [1, 3]),
        "permuted": ([2, 0], [3, 1]),
    }[layout]
    rng = np.random.default_rng(20261004)
    diag = rng.standard_normal((2, 6)) + 1j * rng.standard_normal((2, 6))
    blocks = rng.standard_normal((2, 6, 6)) + 1j * rng.standard_normal((2, 6, 6))
    groups = (
        single_body_cupy.CuPyDiagonalTGroup(
            particle_indices=np.array(first),
            T_M=np.ones((2, 2)),
            T_N=np.ones((2, 2)),
            T_diag=diag,
        ),
        single_body_cupy.CuPyDenseTGroup(particle_indices=np.array(second), T_blocks=blocks),
    )
    operator = single_body_cupy.CuPyCompositeParticleTOperator(lmax=1, n_particles=4, groups=groups)
    host = rng.standard_normal((4, 6, nrhs)) + 1j * rng.standard_normal((4, 6, nrhs))
    expected = np.empty_like(host)
    expected[first] = (diag.conj() if adjoint else diag)[:, :, None] * host[first]
    expected[second] = np.einsum(
        "gij,gjr->gir", blocks.conj().swapaxes(-1, -2) if adjoint else blocks, host[second]
    )
    values = cupy.asarray(host.reshape(24) if nrhs == 1 else host.reshape(24, nrhs))

    class NoLayoutScans:
        def __getattr__(self, name):
            if name in {"arange", "array_equal", "diff", "unique"}:
                raise AssertionError(f"Layout discovery {name} repeated during T application")
            return getattr(np, name)

    monkeypatch.setattr(single_body_cupy, "np", NoLayoutScans())
    action = operator.apply_adjoint if adjoint else operator.apply
    for _ in range(2):
        result = action(values)
        np.testing.assert_allclose(
            cupy.asnumpy(result).reshape(host.shape), expected, rtol=1e-13, atol=1e-13
        )
    np.testing.assert_array_equal(cupy.asnumpy(values).reshape(host.shape), host)
