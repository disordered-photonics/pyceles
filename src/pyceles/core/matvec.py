"""Many-sphere matvec assembly (CPU reference, CELES conventions).

We solve the multiple-scattering system in the CELES form:

Let x be the stacked *outgoing/scattered* multipole coefficients for all spheres.
Let W_ij be the CELES translation block mapping sphere j coefficients into the
*incident* coefficients at i.
Let T_i be the diagonal single-sphere T operator (scattered = T * incident).

Self-consistent equation:
  x_i = T_i ( b_i + sum_{j!=i} W_ij x_j )

Linear system:
  (I - T W) x = T b

This file provides a correctness-first O(N^2) implementation:
- apply_W_numpy: y = W x
- apply_A_numpy: y = (I - T W) x
- rhs_Tb_numpy:  r = T b
- assemble_dense_A_numpy: explicit dense A = I - T W (small systems only)

Performance notes
-----------------
Even for the O(N^2) reference, it is crucial to precompute reusable quantities:
- per-sphere T-diagonal entries (`precompute_T_diagonal`)
- angular translation table `ab5`
- radial translation LUT (`RadialLUT`, always used)

Current implementation is exact (no distance binning approximation) and supports:
- matrix-free translation blocks on the fly (CELES-style default)
- optional exact block caching (`cache_translation_blocks=True`) for small systems

Future speedups:
- block-diagonal preconditioner (CELES)
- GPU backend (CuPy) for hot paths

Practical usage
---------------
Use `prepare_matvec(...)` once, then repeatedly call:
- `prepared.apply_A(x)` inside GMRES
- `prepared.rhs_Tb(b)` for the right-hand side

This avoids accidental recomputation of T-diagonal entries, ab5 tables, LUTs,
and pair translation blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .geometry_bounds import conservative_set_diameter
from .indexing import n_modes
from .particles import Particle, Sphere
from .tmatrix import particle_T_diagonal, sphere_T_diagonal
from .translation import RadialLUT, translation_ab5_table, translation_block

Array = np.ndarray


@dataclass(frozen=True)
class Geometry:
    """Minimal geometry container for sphere-center coordinates."""

    positions: Array  # (Ns,3)


@dataclass
class PreparedMatvec:
    """Prepared data for repeated many-sphere matvec calls.

    This keeps expensive geometry/material precomputations out of GMRES inner
    iterations. Users can keep this object and call `apply_A` / `rhs_Tb`.
    """

    lmax: int
    # This solver path assumes a homogeneous non-absorbing host medium.
    # Therefore the medium wavenumber k is real-valued in all translation kernels.
    k: float
    positions: Array
    T_M: Array
    T_N: Array
    T_diag: Array
    ab5: Array
    radial_lut: RadialLUT
    dtype: np.dtype = np.dtype(np.complex128)
    cache_translation_blocks: bool = False
    _W_cache: dict[tuple[int, int], Array] = field(default_factory=dict)

    def apply_W(self, x: Array) -> Array:
        """Apply inter-particle translation operator `W` to coefficient vector."""
        return apply_W_numpy(
            self.lmax,
            self.k,
            self.positions,
            x,
            self.ab5,
            dtype=self.dtype,
            radial_lut=self.radial_lut,
            block_cache=self._W_cache if self.cache_translation_blocks else None,
        )

    def apply_A(self, x: Array) -> Array:
        """Apply full linear operator `A = I - T W` used by iterative solvers."""
        return apply_A_numpy(
            self.lmax,
            self.k,
            self.positions,
            x,
            T_M=self.T_M,
            T_N=self.T_N,
            T_diag=self.T_diag,
            ab5=self.ab5,
            dtype=self.dtype,
            radial_lut=self.radial_lut,
            block_cache=self._W_cache if self.cache_translation_blocks else None,
        )

    def rhs_Tb(self, b: Array) -> Array:
        """Apply right-hand side mapping `b -> T b` in CELES formulation."""
        return rhs_Tb_numpy(
            self.lmax, b, T_M=self.T_M, T_N=self.T_N, T_diag=self.T_diag, dtype=self.dtype
        )

    def populate_translation_cache(self, *, show_progress: bool = False) -> None:
        """Precompute and cache all pair translation blocks W_ij (i!=j).

        Warning: memory scales as O(N^2 * Nm^2), so this is for small/debug
        cases only.
        """

        if not self.cache_translation_blocks:
            return

        Ns = self.positions.shape[0]
        pair_iter = [(i, j) for i in range(Ns) for j in range(Ns) if i != j]
        if show_progress:
            pair_iter = tqdm(pair_iter, desc="Precompute W_ij")

        for i, j in pair_iter:
            key = (i, j)
            if key in self._W_cache:
                continue
            rvec = self.positions[i] - self.positions[j]
            self._W_cache[key] = translation_block(
                self.lmax,
                self.k,
                rvec,
                ab5=self.ab5,
                radial_lut=self.radial_lut,
            )


def _build_T_mode_diagonal(lmax: int, T_M: Array, T_N: Array) -> Array:
    """Expand per-(sphere,l) diagonal entries to per-(sphere,mode) factors."""

    lmax = int(lmax)
    T_M = np.asarray(T_M)
    T_N = np.asarray(T_N)

    if T_M.shape != T_N.shape:
        raise ValueError(
            f"T_M and T_N must have identical shapes. Got {T_M.shape} and {T_N.shape}."
        )

    Ns = T_M.shape[0]
    Nm = n_modes(lmax)
    Nscl = lmax * (lmax + 2)
    out_dtype = np.result_type(T_M.dtype, T_N.dtype, np.complex64)
    T_diag = np.zeros((Ns, Nm), dtype=out_dtype)

    for l in range(1, lmax + 1):
        start = (l - 1) * (l + 1)
        end = start + (2 * l + 1)
        T_diag[:, start:end] = T_M[:, l : l + 1]
        T_diag[:, Nscl + start : Nscl + end] = T_N[:, l : l + 1]

    return T_diag


def _infer_rmax(positions: Array) -> float:
    """Conservative upper bound on center-to-center separation for LUT sizing.

    Uses the axis-aligned bounding-box diagonal. This is O(N) in both time and
    memory and safely upper-bounds the true maximum pair distance.
    """

    return conservative_set_diameter(np.asarray(positions, dtype=float))


def prepare_matvec(
    *,
    lmax: int,
    k: float,
    positions: Array,
    radii: Array,
    n_particle: Array,
    n_medium: complex = 1.0 + 0j,
    radial_lut_dr: float,
    cache_translation_blocks: bool = False,
    operator_dtype: npt.DTypeLike = np.complex128,
) -> PreparedMatvec:
    """Prepare reusable data for many `(I - T W)` applications.

    This always builds a `RadialLUT` from geometry and `radial_lut_dr`.

    Notes
    -----
    The default execution model remains matrix-free: GMRES repeatedly calls
    `PreparedMatvec.apply_A` without assembling a global dense matrix. Optional
    `cache_translation_blocks=True` stores exact pair blocks W_ij to trade RAM
    for speed on systems where memory headroom is available.
    """

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (Ns,3). Got {positions.shape}.")

    op_dtype = np.dtype(operator_dtype)

    # Keep k explicitly real for this unbounded homogeneous-medium path:
    # a complex host index is intentionally rejected at SimulationConfig level.
    k_f = float(k)

    T_M, T_N = precompute_T_diagonal(
        lmax=lmax,
        k=k_f,
        radii=radii,
        n_particle=n_particle,
        n_medium=n_medium,
        dtype=op_dtype,
    )
    T_diag = _build_T_mode_diagonal(lmax, T_M, T_N)

    ab5 = translation_ab5_table(lmax, dtype=op_dtype)

    dr = float(radial_lut_dr)
    if dr <= 0.0:
        raise ValueError(f"radial_lut_dr must be > 0, got {dr}.")
    lut = RadialLUT(lmax=int(lmax), k=k_f, r_max=_infer_rmax(positions), dr=dr, dtype=op_dtype)

    return PreparedMatvec(
        lmax=int(lmax),
        k=k_f,
        positions=positions,
        T_M=T_M,
        T_N=T_N,
        T_diag=T_diag,
        ab5=ab5,
        radial_lut=lut,
        dtype=op_dtype,
        cache_translation_blocks=bool(cache_translation_blocks),
    )


def prepare_matvec_from_particles(
    *,
    lmax: int,
    k: float,
    particles: list[Particle],
    n_medium: complex = 1.0 + 0j,
    radial_lut_dr: float,
    cache_translation_blocks: bool = False,
    operator_dtype: npt.DTypeLike = np.complex128,
) -> PreparedMatvec:
    """Prepare reusable matvec data from an explicit particle list.

    This is the extensible geometry path for mixed particle families that still
    expose diagonal per-(tau,l,m) response entries.
    """
    part = list(particles)
    positions = np.asarray(
        [np.asarray(p.position, dtype=float) for p in part], dtype=float
    ).reshape(-1, 3)
    op_dtype = np.dtype(operator_dtype)
    k_f = float(k)

    T_M, T_N = precompute_T_diagonal_from_particles(
        lmax=int(lmax),
        k=k_f,
        particles=part,
        n_medium=n_medium,
    )
    T_M = np.asarray(T_M, dtype=op_dtype, copy=False)
    T_N = np.asarray(T_N, dtype=op_dtype, copy=False)
    T_diag = _build_T_mode_diagonal(int(lmax), T_M, T_N)
    ab5 = translation_ab5_table(int(lmax), dtype=op_dtype)

    dr = float(radial_lut_dr)
    if dr <= 0.0:
        raise ValueError(f"radial_lut_dr must be > 0, got {dr}.")
    lut = RadialLUT(lmax=int(lmax), k=k_f, r_max=_infer_rmax(positions), dr=dr, dtype=op_dtype)

    return PreparedMatvec(
        lmax=int(lmax),
        k=k_f,
        positions=positions,
        T_M=T_M,
        T_N=T_N,
        T_diag=T_diag,
        ab5=ab5,
        radial_lut=lut,
        dtype=op_dtype,
        cache_translation_blocks=bool(cache_translation_blocks),
    )


def estimate_translation_cache_bytes(
    N: int, lmax: int, *, dtype: npt.DTypeLike = np.complex128
) -> int:
    """Estimate memory of storing all pair blocks W_ij (i!=j).

    This excludes Python-dictionary/object overhead from cache bookkeeping.
    """

    N = int(N)
    lmax = int(lmax)
    if N < 0:
        raise ValueError("N must be non-negative")
    Nm = n_modes(lmax)
    pairs = N * (N - 1)
    block_entries = Nm * Nm
    return pairs * block_entries * np.dtype(dtype).itemsize


def make_prepared_A_and_rhs(
    prepared: PreparedMatvec, b: Array
) -> tuple[Callable[[Array], Array], Array]:
    """Return `(A_mv, rhs)` from a prepared system and incident coefficients."""

    rhs = prepared.rhs_Tb(b)

    def A_mv(x: Array) -> Array:
        """Matrix-free apply of `A = I - T W` using precomputed prepared data."""
        return prepared.apply_A(x)

    return A_mv, rhs


def assemble_dense_A_numpy(
    prepared: PreparedMatvec,
    *,
    show_progress: bool = False,
    use_cache: bool = False,
    store_blocks: bool = False,
) -> Array:
    """Assemble dense A = I - T W from a prepared system.

    This is intended for small systems where a direct dense solve is feasible.
    Assembly is blockwise over sphere pairs:
      A_ii = I
      A_ij = -diag(T_i) @ W_ij, i != j
    """

    Ns = prepared.positions.shape[0]
    Nm = n_modes(prepared.lmax)
    n = Ns * Nm
    A = np.zeros((n, n), dtype=prepared.dtype)
    A[np.arange(n), np.arange(n)] = 1.0 + 0.0j

    pair_iter = ((i, j) for i in range(Ns) for j in range(Ns) if i != j)
    if show_progress:
        pair_iter = tqdm(pair_iter, total=Ns * (Ns - 1), desc="Assemble A (blockwise)")

    cache = prepared._W_cache if use_cache else None
    for i, j in pair_iter:
        key = (i, j)
        Wij = cache.get(key) if cache is not None else None
        if Wij is None:
            rvec = prepared.positions[i] - prepared.positions[j]
            Wij = translation_block(
                prepared.lmax,
                prepared.k,
                rvec,
                ab5=prepared.ab5,
                radial_lut=prepared.radial_lut,
            )
            if store_blocks:
                prepared._W_cache[key] = Wij

        blk = -(prepared.T_diag[i][:, None] * Wij)
        rs = slice(i * Nm, (i + 1) * Nm)
        cs = slice(j * Nm, (j + 1) * Nm)
        A[rs, cs] = blk

    return A


def precompute_T_diagonal(
    *,
    lmax: int,
    k: float,
    radii: Array,
    n_particle: Array,
    n_medium: complex = 1.0 + 0j,
    dtype: npt.DTypeLike = np.complex128,
) -> tuple[Array, Array]:
    """Precompute per-sphere diagonal T entries.

    Returns
    -------
    T_M, T_N:
        Arrays of shape (Ns, lmax+1) where index l=0 is unused.
    """

    out_dtype = np.dtype(dtype)
    lmax = int(lmax)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    n_particle = np.asarray(n_particle, dtype=out_dtype).reshape(-1)
    Ns = radii.size

    T_M = np.zeros((Ns, lmax + 1), dtype=out_dtype)
    T_N = np.zeros((Ns, lmax + 1), dtype=out_dtype)

    # Reuse identical single-sphere evaluations (common in monodisperse clusters).
    # Keep the memo bounded so highly heterogeneous systems do not grow a huge dict.
    MAX_TMEMO_ENTRIES = 100_000
    tmemo: dict[tuple[float, complex, complex, complex, int], tuple[Array, Array]] = {}
    for i in range(Ns):
        ri = float(radii[i])
        npi = complex(n_particle[i])
        nmed = complex(n_medium)
        key = (ri, npi, complex(k), nmed, int(lmax))
        cached = tmemo.get(key)
        if cached is None:
            Td = sphere_T_diagonal(lmax, k, ri, npi, nmed)
            cached = (
                np.asarray(Td[1], dtype=out_dtype).copy(),
                np.asarray(Td[2], dtype=out_dtype).copy(),
            )
            if len(tmemo) < MAX_TMEMO_ENTRIES:
                tmemo[key] = cached
        T_M[i, :] = cached[0]
        T_N[i, :] = cached[1]

    return T_M, T_N


def precompute_T_diagonal_from_particles(
    *,
    lmax: int,
    k: float,
    particles: list[Particle],
    n_medium: complex = 1.0 + 0j,
) -> tuple[Array, Array]:
    """Precompute per-particle diagonal T entries from particle objects.

    This is the extensible counterpart of `precompute_T_diagonal` and is intended
    for future non-spherical particle support.
    """
    lmax = int(lmax)
    Ns = len(particles)
    T_M = np.zeros((Ns, lmax + 1), dtype=np.complex128)
    T_N = np.zeros((Ns, lmax + 1), dtype=np.complex128)

    # Fast reuse for identical spheres.
    sphere_memo: dict[tuple[float, complex, complex, complex, int], tuple[Array, Array]] = {}
    for i, p in enumerate(particles):
        if isinstance(p, Sphere):
            key = (
                float(p.radius),
                complex(p.refractive_index),
                complex(k),
                complex(n_medium),
                int(lmax),
            )
            cached = sphere_memo.get(key)
            if cached is None:
                Td = sphere_T_diagonal(lmax, k, p.radius, p.refractive_index, n_medium)
                cached = (Td[1].copy(), Td[2].copy())
                sphere_memo[key] = cached
            T_M[i, :] = cached[0]
            T_N[i, :] = cached[1]
            continue

        Td = particle_T_diagonal(lmax=lmax, k_medium=k, particle=p, n_medium=n_medium)
        T_M[i, :] = np.asarray(Td[1], dtype=np.complex128)
        T_N[i, :] = np.asarray(Td[2], dtype=np.complex128)

    return T_M, T_N


def apply_W_numpy(
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    ab5: Array,
    *,
    dtype: npt.DTypeLike = np.complex128,
    radial_lut: Optional[RadialLUT],
    block_cache: Optional[dict[tuple[int, int], Array]] = None,
) -> Array:
    """Compute y = W x (NumPy), excluding self-interaction.

    If `radial_lut` is `None`, the translation radial functions are evaluated
    exactly (no LUT approximation).

    This routine is the matrix-free hot path: W_ij blocks are formed per pair
    and applied directly, rather than materializing a global W matrix.
    """

    out_dtype = np.dtype(dtype)
    Ns = positions.shape[0]
    Nm = n_modes(lmax)
    x = np.asarray(x, dtype=out_dtype).reshape(Ns, Nm)
    y = np.zeros_like(x, dtype=out_dtype)

    for i in range(Ns):
        for j in range(Ns):
            if i == j:
                continue
            key = (i, j)
            Wij = block_cache.get(key) if block_cache is not None else None
            if Wij is None:
                rvec = positions[i] - positions[j]
                Wij = translation_block(lmax, k, rvec, ab5=ab5, radial_lut=radial_lut)
                if block_cache is not None:
                    block_cache[key] = Wij
            y[i] += Wij @ x[j]

    return y.reshape(Ns * Nm)


def apply_A_numpy(
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    *,
    T_M: Array,
    T_N: Array,
    T_diag: Optional[Array] = None,
    ab5: Optional[Array] = None,
    dtype: npt.DTypeLike = np.complex128,
    radial_lut: Optional[RadialLUT],
    block_cache: Optional[dict[tuple[int, int], Array]] = None,
) -> Array:
    """Compute y = (I - T W) x with precomputed diagonal T entries.

    If `radial_lut` is `None`, the translation radial functions are evaluated
    exactly (no LUT approximation).
    """

    out_dtype = np.dtype(dtype)
    Ns = positions.shape[0]
    Nm = n_modes(lmax)
    if ab5 is None:
        ab5 = translation_ab5_table(lmax)

    Wx = apply_W_numpy(
        lmax,
        k,
        positions,
        x,
        ab5,
        dtype=out_dtype,
        radial_lut=radial_lut,
        block_cache=block_cache,
    ).reshape(Ns, Nm)

    y = np.asarray(x, dtype=out_dtype).reshape(Ns, Nm).copy()

    if T_diag is None:
        T_diag = _build_T_mode_diagonal(lmax, T_M, T_N)
    else:
        T_diag = np.asarray(T_diag, dtype=out_dtype)

    y -= T_diag * Wx

    return y.reshape(Ns * Nm)


def rhs_Tb_numpy(
    lmax: int,
    b: Array,
    *,
    T_M: Array,
    T_N: Array,
    T_diag: Optional[Array] = None,
    dtype: npt.DTypeLike = np.complex128,
) -> Array:
    """Compute r = T b (NumPy) where b is stacked incident coefficients per sphere."""

    out_dtype = np.dtype(dtype)
    lmax = int(lmax)
    Nm = n_modes(lmax)

    b = np.asarray(b, dtype=out_dtype)
    Ns = b.size // Nm
    b = b.reshape(Ns, Nm)

    if T_diag is None:
        T_diag = _build_T_mode_diagonal(lmax, T_M, T_N)
    else:
        T_diag = np.asarray(T_diag, dtype=out_dtype)

    r = T_diag * b

    return r.reshape(Ns * Nm)
