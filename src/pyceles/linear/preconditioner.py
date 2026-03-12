from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from tqdm.auto import tqdm

from pyceles._optional import asnumpy, import_cupy, is_cupy_array
from pyceles.core.indexing import n_modes
from pyceles.core.operators import PreparedOperator
from pyceles.core.translation import translation_block


def _normalize_subdivisions(subdivisions: int | tuple[int, int, int]) -> tuple[int, int, int]:
    """Normalize grid-subdivision input to explicit `(nx, ny, nz)` integers."""
    if isinstance(subdivisions, (int, np.integer)):
        n = int(subdivisions)
        if n < 1:
            raise ValueError(f"`subdivisions` must be >= 1. Got {subdivisions!r}.")
        return (n, n, n)
    if isinstance(subdivisions, (tuple, list)) and len(subdivisions) == 3:
        out = tuple(int(v) for v in subdivisions)
        if any(v < 1 for v in out):
            raise ValueError(f"`subdivisions` entries must be >= 1. Got {subdivisions!r}.")
        return out  # type: ignore[return-value]
    raise ValueError(
        f"`subdivisions` must be an int or a length-3 tuple/list of ints. Got {subdivisions!r}."
    )


def regular_grid_partition(
    positions: np.ndarray,
    *,
    subdivisions: int | tuple[int, int, int] = 2,
    cubic_bbox: bool = True,
) -> list[np.ndarray]:
    """Partition particle centers into regular-grid spatial blocks.

    Parameters
    ----------
    positions:
        Particle-center array with shape `(Ns, 3)`.
    subdivisions:
        Either one integer `n` (interpreted as `(n,n,n)`) or explicit
        `(nx, ny, nz)` counts.
    cubic_bbox:
        If True, use an enclosing cube. If False, use the axis-wise
        bounding box (parallelepiped).
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"`positions` must have shape (Ns,3). Got {p.shape}.")
    if p.shape[0] == 0:
        return []

    nx, ny, nz = _normalize_subdivisions(subdivisions)
    nxyz = np.array([nx, ny, nz], dtype=np.int64)

    p_min = np.min(p, axis=0)
    p_max = np.max(p, axis=0)
    span = p_max - p_min
    if cubic_bbox:
        side = float(np.max(span))
        if side <= 0.0:
            side = 1.0
        center = 0.5 * (p_min + p_max)
        lo = center - 0.5 * side
        span_vec = np.array([side, side, side], dtype=float)
    else:
        lo = p_min
        span_vec = np.where(span > 0.0, span, 1.0)

    t = (p - lo[None, :]) / span_vec[None, :]
    # Clamp to [0,1) so the max-boundary point falls in the last cell.
    t = np.clip(t, 0.0, np.nextafter(1.0, 0.0))
    idx = np.floor(t * nxyz[None, :]).astype(np.int64)
    idx = np.clip(idx, 0, nxyz[None, :] - 1)

    # Deterministic ordering by linearized cell id.
    linear = idx[:, 0] + nx * (idx[:, 1] + ny * idx[:, 2])
    order = np.argsort(linear, kind="stable")
    linear_sorted = linear[order]

    blocks: list[np.ndarray] = []
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and linear_sorted[end] == linear_sorted[start]:
            end += 1
        blocks.append(np.asarray(order[start:end], dtype=np.int64))
        start = end
    return blocks


def _pair_block(
    prepared: PreparedOperator,
    i: int,
    j: int,
    *,
    store_translations: bool,
) -> np.ndarray:
    """Return one pair-translation block `W_ij`, optionally reusing cache."""
    coupling = prepared.coupling
    if not hasattr(coupling, "ab5") or not hasattr(coupling, "radial_lut"):
        raise TypeError(
            "Grid-block preconditioning currently requires a pairwise free-space "
            f"coupling backend exposing `ab5` and `radial_lut`. Got {type(coupling).__name__}."
        )
    cache_enabled = bool(getattr(coupling, "cache_translation_blocks", False))
    cache_dict = getattr(coupling, "_W_cache", None)
    key = (int(i), int(j))
    Wij = cache_dict.get(key) if cache_enabled and isinstance(cache_dict, dict) else None
    if Wij is None:
        rvec = prepared.positions[i] - prepared.positions[j]
        Wij = translation_block(
            prepared.lmax,
            # prepared.k is real by construction in the current homogeneous-medium path.
            prepared.k,
            rvec,
            ab5=np.asarray(coupling.ab5, dtype=prepared.dtype),
            radial_lut=coupling.radial_lut,
        )
        if store_translations and cache_enabled and isinstance(cache_dict, dict):
            cache_dict[key] = Wij
    return np.asarray(Wij, dtype=prepared.dtype)


def _assemble_local_A_block(
    prepared: PreparedOperator,
    particle_ids: np.ndarray,
    *,
    store_translations: bool,
) -> np.ndarray:
    """Assemble dense local block of `A = I - T W` for one particle partition."""
    ids = np.asarray(particle_ids, dtype=np.int64).reshape(-1)
    Nm = n_modes(prepared.lmax)
    nblk = ids.size * Nm
    A = np.eye(nblk, dtype=prepared.dtype)
    for li, gi in enumerate(ids):
        rs = slice(li * Nm, (li + 1) * Nm)
        for lj, gj in enumerate(ids):
            if gi == gj:
                continue
            cs = slice(lj * Nm, (lj + 1) * Nm)
            Wij = _pair_block(prepared, int(gi), int(gj), store_translations=store_translations)
            A[rs, cs] = -np.asarray(
                asnumpy(prepared.apply_particle_block(int(gi), Wij)),
                dtype=prepared.dtype,
            )
    return A


@dataclass
class GridBlockPreconditioner:
    """Block-diagonal preconditioner built from regular-grid particle groups.

    Each spatial block contributes one local dense system that is LU-factorized.
    Applying the preconditioner solves each block independently.

    The important architectural point is that this object no longer assumes
    spherical particles. It works with any prepared operator that can assemble
    local `T_i @ W_ij` block rows through `PreparedOperator.apply_particle_block`.
    That keeps the preconditioner compatible with layered spheres today and with
    future mixed diagonal/axisymmetric/dense particle sets.
    """

    particle_blocks: list[np.ndarray]
    lu_factors: list[tuple[np.ndarray, np.ndarray]]
    n_particles: int
    n_modes: int
    dtype: np.dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply block-diagonal inverse to one or many RHS vectors."""
        import scipy.linalg

        arr = np.asarray(x, dtype=self.dtype)
        if arr.ndim == 1:
            if arr.size != self.n_particles * self.n_modes:
                raise ValueError(
                    f"Input length must be {self.n_particles * self.n_modes}, got {arr.size}."
                )
            arr3 = arr.reshape(self.n_particles, self.n_modes, 1)
            squeeze = True
        elif arr.ndim == 2:
            if arr.shape[0] != self.n_particles * self.n_modes:
                raise ValueError(
                    f"Input first dimension must be {self.n_particles * self.n_modes}, got {arr.shape[0]}."
                )
            arr3 = arr.reshape(self.n_particles, self.n_modes, arr.shape[1])
            squeeze = False
        else:
            raise ValueError(f"Input must be 1D or 2D. Got shape {arr.shape}.")

        out = np.zeros_like(arr3, dtype=self.dtype)
        for ids, (lu, piv) in zip(self.particle_blocks, self.lu_factors):
            rhs_loc = arr3[ids, :, :].reshape(ids.size * self.n_modes, -1)
            sol_loc = scipy.linalg.lu_solve((lu, piv), rhs_loc, check_finite=False)
            out[ids, :, :] = sol_loc.reshape(ids.size, self.n_modes, -1)

        out2 = out.reshape(self.n_particles * self.n_modes, -1)
        return out2[:, 0] if squeeze else out2

    @property
    def n_blocks(self) -> int:
        """Number of populated spatial blocks in the preconditioner."""
        return len(self.particle_blocks)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        """Particle counts per spatial block (for diagnostics/monitoring)."""
        return tuple(int(b.size) for b in self.particle_blocks)


@dataclass
class CuPyGridBlockPreconditioner:
    """GPU-resident grid-block preconditioner with CPU assembly and GPU LU solves.

    The local dense blocks are still assembled through the prepared-operator
    boundary on CPU. That keeps the block-preconditioner logic representation-
    agnostic and avoids duplicating translation/local-operator assembly paths.
    Once assembled, each block is uploaded and LU-factorized on device so CuPy
    GMRES applies the preconditioner without leaving the GPU hot path.
    """

    particle_blocks: list[np.ndarray]
    lu_factors: list[tuple[object, object]]
    n_particles: int
    n_modes: int
    dtype: np.dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        cupy, _ = import_cupy()
        import cupyx.scipy.linalg

        arr = np.asarray(x) if not is_cupy_array(x) else x
        arr_gpu = cupy.asarray(arr, dtype=self.dtype)
        if arr_gpu.ndim == 1:
            if arr_gpu.size != self.n_particles * self.n_modes:
                raise ValueError(
                    f"Input length must be {self.n_particles * self.n_modes}, got {arr_gpu.size}."
                )
            arr3 = arr_gpu.reshape(self.n_particles, self.n_modes, 1)
            squeeze = True
        elif arr_gpu.ndim == 2:
            if arr_gpu.shape[0] != self.n_particles * self.n_modes:
                raise ValueError(
                    f"Input first dimension must be {self.n_particles * self.n_modes}, got {arr_gpu.shape[0]}."
                )
            arr3 = arr_gpu.reshape(self.n_particles, self.n_modes, arr_gpu.shape[1])
            squeeze = False
        else:
            raise ValueError(f"Input must be 1D or 2D. Got shape {arr_gpu.shape}.")

        out = cupy.zeros_like(arr3, dtype=self.dtype)
        for ids, lu_payload in zip(self.particle_blocks, self.lu_factors):
            rhs_loc = arr3[ids, :, :].reshape(ids.size * self.n_modes, -1)
            sol_loc = cupyx.scipy.linalg.lu_solve(lu_payload, rhs_loc)
            out[ids, :, :] = sol_loc.reshape(ids.size, self.n_modes, -1)

        out2 = out.reshape(self.n_particles * self.n_modes, -1)
        out_ret = out2[:, 0] if squeeze else out2
        return out_ret if is_cupy_array(x) else asnumpy(out_ret)

    @property
    def n_blocks(self) -> int:
        return len(self.particle_blocks)

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(int(b.size) for b in self.particle_blocks)


def make_grid_block_preconditioner(
    prepared: PreparedOperator,
    *,
    backend: Literal["numpy", "cupy"] = "numpy",
    subdivisions: int | tuple[int, int, int] = 2,
    cubic_bbox: bool = True,
    max_block_unknowns: int | None = None,
    store_translations: bool = False,
    show_progress: bool = False,
) -> GridBlockPreconditioner | CuPyGridBlockPreconditioner:
    """Build a regular-grid block-diagonal preconditioner.

    The particle cloud is partitioned on a regular 3D grid. For each populated
    block, we assemble the local dense block of `A = I - T W` and LU-factorize it.

    Because local assembly goes through the prepared-operator boundary, this
    routine is representation-agnostic: the same code can precondition diagonal
    spheres, layered spheres, or future mixed clusters as long as the
    particle-local T operator can left-apply each particle-local `T_i`.
    """
    import scipy.linalg

    Ns = int(prepared.positions.shape[0])
    Nm = int(n_modes(prepared.lmax))
    blocks = regular_grid_partition(
        prepared.positions,
        subdivisions=subdivisions,
        cubic_bbox=bool(cubic_bbox),
    )

    if len(blocks) == 0:
        raise ValueError("No populated blocks were generated for the preconditioner.")

    max_u = None if max_block_unknowns is None else int(max_block_unknowns)
    if max_u is not None and max_u < 1:
        raise ValueError(f"`max_block_unknowns` must be >= 1 when set. Got {max_block_unknowns!r}.")

    block_iter: Iterable[np.ndarray] = blocks
    if show_progress:
        block_iter = tqdm(blocks, desc="Build block preconditioner", total=len(blocks))

    lu_factors: list[tuple[np.ndarray, np.ndarray]] = []
    lu_factors_gpu: list[tuple[object, object]] = []
    kept_blocks: list[np.ndarray] = []
    use_cupy = backend == "cupy"
    if use_cupy:
        cupy, _ = import_cupy()
        import cupyx.scipy.linalg
    for ids in block_iter:
        nblk_u = int(ids.size * Nm)
        if max_u is not None and nblk_u > max_u:
            raise ValueError(
                f"Preconditioner block has {nblk_u} unknowns (> max_block_unknowns={max_u}). "
                "Increase subdivisions or raise max_block_unknowns."
            )
        Ablk = _assemble_local_A_block(
            prepared,
            ids,
            store_translations=bool(store_translations),
        )
        if use_cupy:
            Ablk_gpu = cupy.asarray(Ablk, dtype=prepared.dtype)
            lu_factors_gpu.append(cupyx.scipy.linalg.lu_factor(Ablk_gpu))
        else:
            lu, piv = scipy.linalg.lu_factor(Ablk, overwrite_a=False, check_finite=False)
            lu_factors.append((lu, piv))
        kept_blocks.append(np.asarray(ids, dtype=np.int64))

    if use_cupy:
        return CuPyGridBlockPreconditioner(
            particle_blocks=kept_blocks,
            lu_factors=lu_factors_gpu,
            n_particles=Ns,
            n_modes=Nm,
            dtype=np.dtype(prepared.dtype),
        )
    return GridBlockPreconditioner(
        particle_blocks=kept_blocks,
        lu_factors=lu_factors,
        n_particles=Ns,
        n_modes=Nm,
        dtype=np.dtype(prepared.dtype),
    )
