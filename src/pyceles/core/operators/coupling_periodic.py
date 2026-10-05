"""Periodic coupling-operator descriptors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from pyceles.core.indexing import n_modes
from pyceles.core.periodic import PeriodicSpec
from pyceles.core.periodic.directsum import (
    apply_periodic_direct_sum,
    apply_periodic_direct_sum_adjoint,
)
from pyceles.core.periodic.ewald import (
    EwaldShellWorkspace,
    apply_periodic_ewald_adjoint_sum,
    apply_periodic_ewald_sum,
    ewald_self_correction,
    ewald_structural_sums_2d_batch,
    fill_periodic_ewald_block_cache,
    periodic_ewald_blocks_for_source,
    resolve_ewald_eta,
)
from pyceles.core.periodic.rayleigh import (
    RayleighNearCacheEstimate,
    RayleighPlan,
    apply_rayleigh_far_adjoint_numpy,
    apply_rayleigh_far_numpy,
    near_pair_csr,
    prepare_rayleigh_plan,
    rayleigh_near_cache_estimate,
    valid_structural_indices,
)
from pyceles.core.periodic.scalar import structural_sum_m_normalization
from pyceles.core.periodic.structural import translation_contraction_tensor

from .base import SourceBlockBatch

Array = np.ndarray


@dataclass
class PeriodicCouplingOperator:
    """Bloch-reduced periodic coupling descriptor for rectangular 2D lattices.

    ``dtype`` controls stored coefficients and the public coupling output;
    ``accum_dtype`` controls sensitive Rayleigh scan recurrences. Generic
    Ewald structural sums retain their established complex128 reference
    arithmetic, while the translation contraction and stored blocks use the
    requested operator dtype.
    """

    lmax: int
    k: float
    positions: Array
    ab5: Array
    periodic: PeriodicSpec
    k_parallel: Array
    dtype: np.dtype
    accum_dtype: np.dtype | None = None
    cache_blocks: bool = False
    circumscribing_radii: Array | None = None
    _ewald_block_cache: dict[tuple[int, int], Array] = field(
        default_factory=dict, init=False, repr=False
    )
    _ewald_shell_workspace: EwaldShellWorkspace | None = field(default=None, init=False, repr=False)
    _resolved_ewald_eta: float | None = field(default=None, init=False, repr=False)
    _structural_contraction_tensor: Array | None = field(default=None, init=False, repr=False)
    _rayleigh_plan_cache: RayleighPlan | None = field(default=None, init=False, repr=False)
    _near_indptr: Array | None = field(default=None, init=False, repr=False)
    _near_destinations: Array | None = field(default=None, init=False, repr=False)
    _near_structural_sums: Array | None = field(default=None, init=False, repr=False)
    _near_cache_estimate: RayleighNearCacheEstimate | None = field(
        default=None, init=False, repr=False
    )
    _self_block_cache: Array | None = field(default=None, init=False, repr=False)
    _near_contraction_tensor_cache: Array | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        self.accum_dtype = self.dtype if self.accum_dtype is None else np.dtype(self.accum_dtype)
        if self.accum_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
            raise TypeError(
                "Periodic coupling accumulation requires complex64 or complex128; "
                f"got {self.accum_dtype!r}."
            )
        if self.accum_dtype.itemsize < self.dtype.itemsize:
            raise ValueError(
                "Periodic coupling accumulation dtype cannot be narrower than the "
                f"operator dtype ({self.dtype.name}); got {self.accum_dtype.name}."
            )
        if self.periodic.options.method == "rayleigh" and self.cache_blocks:
            raise ValueError(
                "`cache_translation_blocks=True` is incompatible with periodic "
                "method='rayleigh'; the hybrid operator already caches only its sparse near data."
            )

    @property
    def n_particles(self) -> int:
        return int(np.asarray(self.positions).reshape(-1, 3).shape[0])

    @property
    def n_modes(self) -> int:
        return n_modes(int(self.lmax))

    def _ewald_eta(self) -> float:
        """Return the effective Ewald split for this operator."""
        eta = self.periodic.options.eta
        if eta is not None:
            return float(eta)
        if self._resolved_ewald_eta is None:
            rayleigh_z_cut = (
                self._rayleigh_plan().z_cut if self.periodic.options.method == "rayleigh" else None
            )
            self._resolved_ewald_eta = resolve_ewald_eta(
                periodic=self.periodic,
                k=float(self.k),
                k_parallel=self.k_parallel,
                positions=self.positions,
                lmax=int(self.lmax),
                max_vertical_offset=rayleigh_z_cut,
            )
        return float(self._resolved_ewald_eta)

    def _contraction_tensor(self) -> Array:
        """Return the reusable tensor mapping scalar structural sums to W blocks."""
        tensor = self._structural_contraction_tensor
        if tensor is None:
            tensor = translation_contraction_tensor(
                lmax=int(self.lmax),
                ab5=self.ab5,
                dtype=self.dtype,
            )
            self._structural_contraction_tensor = tensor
        return tensor

    def _workspace(self) -> EwaldShellWorkspace:
        """Return the reusable non-pair Ewald shell workspace for this operator."""
        workspace = self._ewald_shell_workspace
        if workspace is None:
            workspace = EwaldShellWorkspace(
                lattice=self.periodic.lattice,
                k=float(self.k),
                k_parallel=np.asarray(self.k_parallel, dtype=float).reshape(2),
                eta=self._ewald_eta(),
            )
            self._ewald_shell_workspace = workspace
        return workspace

    def _rayleigh_plan(self) -> RayleighPlan:
        plan = self._rayleigh_plan_cache
        if plan is not None:
            return plan
        plan = prepare_rayleigh_plan(
            lmax=int(self.lmax),
            k=float(self.k),
            positions=self.positions,
            circumscribing_radii=self.circumscribing_radii,
            periodic=self.periodic,
            k_parallel=self.k_parallel,
            dtype=self.dtype,
        )
        self._rayleigh_plan_cache = plan
        return plan

    def _near_structure(self) -> tuple[Array, Array]:
        if self._near_indptr is None or self._near_destinations is None:
            indptr, destinations, _sources = near_pair_csr(
                self.positions, self._rayleigh_plan().z_cut
            )
            self._near_indptr = indptr
            self._near_destinations = destinations
            self._near_cache_estimate = rayleigh_near_cache_estimate(
                pair_count=int(destinations.size),
                lmax=int(self.lmax),
                dtype=self.dtype,
            )
        return self._near_indptr, self._near_destinations

    def _near_contraction_tensor(self) -> Array:
        """Return the translation tensor restricted to valid structural channels."""
        tensor = self._near_contraction_tensor_cache
        if tensor is not None:
            return tensor
        degrees, orders = valid_structural_indices(int(self.lmax))
        tensor = self._contraction_tensor()[:, :, degrees, orders]
        self._near_contraction_tensor_cache = tensor
        return tensor

    def _release_rayleigh_preparation_state(self) -> None:
        """Drop Ewald-only preparation state once Rayleigh caches are complete."""
        if self._self_block_cache is None or self._near_structural_sums is None:
            return
        # Materialize the compact contraction view before releasing the full
        # structural tensor that was needed to build the exact self block.
        self._near_contraction_tensor()
        self._ewald_shell_workspace = None
        self._structural_contraction_tensor = None

    def _self_block(self) -> Array:
        """Return the exact periodic self block shared by every particle."""
        block = self._self_block_cache
        if block is not None:
            return block
        order = 2 * int(self.lmax)
        options = self.periodic.options
        origin = np.zeros((3,), dtype=float)
        structural = ewald_structural_sums_2d_batch(
            lmax_struct=int(self.lmax),
            k=float(self.k),
            destinations=origin[None, :],
            source=origin,
            lattice=self.periodic.lattice,
            k_parallel=self.k_parallel,
            eta=self._ewald_eta(),
            real_shells=options.real_shells,
            reciprocal_shells=options.reciprocal_shells,
            shell_tolerance=float(options.shell_tolerance),
            max_shells=int(options.max_shells),
            dtype=np.complex128,
            workspace=self._workspace(),
        )[0]
        structural[0, order] += structural_sum_m_normalization(0) * ewald_self_correction(
            float(self.k), self._ewald_eta()
        )
        block = np.asarray(
            np.einsum(
                "pm,ijpm->ij",
                structural.astype(self.dtype, copy=False),
                self._contraction_tensor(),
                optimize=True,
            ),
            dtype=self.dtype,
        )
        self._self_block_cache = block
        return block

    def _populate_near_structural_sums(self, *, show_progress: bool = False) -> None:
        if self._near_structural_sums is not None:
            return
        indptr, destinations = self._near_structure()
        degrees, orders = valid_structural_indices(int(self.lmax))
        sums = np.empty((destinations.size, degrees.size), dtype=self.dtype)
        options = self.periodic.options
        pos = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        sources: Iterable[int] = range(self.n_particles)
        if show_progress:
            estimate = self._near_cache_estimate
            if estimate is None:
                raise RuntimeError("Periodic near-cache estimation failed.")
            size_gib = float(estimate.structural_bytes) / float(1024**3)
            sources = tqdm(
                sources,
                total=self.n_particles,
                desc=(
                    f"Build periodic near Ewald cache ({estimate.dtype.name}, {size_gib:.2f} GiB)"
                ),
            )
        for source in sources:
            start = int(indptr[source])
            stop = int(indptr[source + 1])
            if start == stop:
                continue
            dest = destinations[start:stop]
            local = ewald_structural_sums_2d_batch(
                lmax_struct=int(self.lmax),
                k=float(self.k),
                destinations=pos[dest],
                source=pos[source],
                lattice=self.periodic.lattice,
                k_parallel=self.k_parallel,
                eta=self._ewald_eta(),
                real_shells=options.real_shells,
                reciprocal_shells=options.reciprocal_shells,
                shell_tolerance=float(options.shell_tolerance),
                max_shells=int(options.max_shells),
                dtype=np.complex128,
                workspace=self._workspace(),
            )
            sums[start:stop] = local[:, degrees, orders].astype(self.dtype, copy=False)
        self._near_structural_sums = sums

    def _reshape_rayleigh_input(self, x: Array) -> tuple[Array, bool]:
        arr_raw = np.asarray(x, dtype=self.dtype)
        expected = self.n_particles * self.n_modes
        if arr_raw.ndim == 1:
            if arr_raw.size != expected:
                raise ValueError(f"Input length must match n_particles * n_modes ({expected}).")
            return arr_raw.reshape(self.n_particles, self.n_modes, 1), True
        if arr_raw.ndim == 2:
            if arr_raw.shape[0] != expected:
                raise ValueError(
                    f"Input first dimension must match n_particles * n_modes ({expected})."
                )
            return arr_raw.reshape(self.n_particles, self.n_modes, arr_raw.shape[1]), False
        raise ValueError(f"Input must be 1D or 2D. Got shape {arr_raw.shape}.")

    def _flatten_rayleigh_output(self, values: Array, *, squeezed: bool) -> Array:
        flat = np.asarray(values, dtype=self.dtype).reshape(
            self.n_particles * self.n_modes, int(values.shape[2])
        )
        return flat[:, 0] if squeezed else flat

    def _apply_rayleigh_far_reshaped(self, arr: Array) -> Array:
        return apply_rayleigh_far_numpy(
            self._rayleigh_plan(), arr, accumulation_dtype=self.accum_dtype
        )

    def _apply_rayleigh_near_reshaped(self, arr: Array, *, target: Array | None = None) -> Array:
        self_contribution = np.einsum("ij,ajr->air", self._self_block(), arr, optimize=True)
        y = self_contribution if target is None else target
        if target is not None:
            y += self_contribution
        self._populate_near_structural_sums(show_progress=False)
        self._release_rayleigh_preparation_state()
        indptr, destinations = self._near_structure()
        sums = self._near_structural_sums
        if sums is None:
            raise RuntimeError("Periodic near Ewald cache population failed.")
        tensor = self._near_contraction_tensor()
        for source in range(self.n_particles):
            start = int(indptr[source])
            stop = int(indptr[source + 1])
            if start == stop:
                continue
            y[destinations[start:stop]] += np.einsum(
                "av,ijv,jr->air", sums[start:stop], tensor, arr[source], optimize=True
            )
        return np.asarray(y, dtype=self.dtype)

    def _apply_rayleigh_far_adjoint_reshaped(self, arr: Array) -> Array:
        return apply_rayleigh_far_adjoint_numpy(
            self._rayleigh_plan(), arr, accumulation_dtype=self.accum_dtype
        )

    def _apply_rayleigh_near_adjoint_reshaped(
        self, arr: Array, *, target: Array | None = None
    ) -> Array:
        # Apply H actions as conj(M.T @ conj(x)) so the adjoint path can reuse
        # the exact forward self block, structural sums, and contraction tensor.
        arr_conj = np.conjugate(arr)
        self_contribution = np.einsum("ji,ajr->air", self._self_block(), arr_conj, optimize=True)
        np.conjugate(self_contribution, out=self_contribution)
        y = self_contribution if target is None else target
        if target is not None:
            y += self_contribution
        self._populate_near_structural_sums(show_progress=False)
        self._release_rayleigh_preparation_state()
        indptr, destinations = self._near_structure()
        sums = self._near_structural_sums
        if sums is None:
            raise RuntimeError("Periodic near Ewald cache population failed.")
        tensor = self._near_contraction_tensor()
        for source in range(self.n_particles):
            start = int(indptr[source])
            stop = int(indptr[source + 1])
            if start == stop:
                continue
            contribution = np.einsum(
                "av,ijv,air->jr",
                sums[start:stop],
                tensor,
                arr_conj[destinations[start:stop]],
                optimize=True,
            )
            np.conjugate(contribution, out=contribution)
            y[source] += contribution
        return np.asarray(y, dtype=self.dtype)

    def _apply_rayleigh_near(self, x: Array) -> Array:
        arr, squeezed = self._reshape_rayleigh_input(x)
        return self._flatten_rayleigh_output(
            self._apply_rayleigh_near_reshaped(arr), squeezed=squeezed
        )

    def _apply_rayleigh_far(self, x: Array) -> Array:
        arr, squeezed = self._reshape_rayleigh_input(x)
        return self._flatten_rayleigh_output(
            self._apply_rayleigh_far_reshaped(arr), squeezed=squeezed
        )

    def _apply_rayleigh(self, x: Array) -> Array:
        arr, squeezed = self._reshape_rayleigh_input(x)
        y = self._apply_rayleigh_far_reshaped(arr)
        self._apply_rayleigh_near_reshaped(arr, target=y)
        return self._flatten_rayleigh_output(y, squeezed=squeezed)

    def _apply_rayleigh_adjoint(self, x: Array) -> Array:
        arr, squeezed = self._reshape_rayleigh_input(x)
        y = self._apply_rayleigh_far_adjoint_reshaped(arr)
        self._apply_rayleigh_near_adjoint_reshaped(arr, target=y)
        return self._flatten_rayleigh_output(y, squeezed=squeezed)

    def apply_adjoint(self, x: Array) -> Array:
        method = self.periodic.options.method
        if method == "rayleigh":
            return self._apply_rayleigh_adjoint(x)
        if method == "directsum":
            return apply_periodic_direct_sum_adjoint(
                lmax=int(self.lmax),
                k=float(self.k),
                positions=self.positions,
                x=x,
                lattice=self.periodic.lattice,
                k_parallel=self.k_parallel,
                window=int(self.periodic.options.directsum_window),
                ab5=self.ab5,
                dtype=self.dtype,
            )
        options = self.periodic.options
        return apply_periodic_ewald_adjoint_sum(
            lmax=int(self.lmax),
            k=float(self.k),
            positions=self.positions,
            x=x,
            lattice=self.periodic.lattice,
            k_parallel=self.k_parallel,
            eta=self._ewald_eta(),
            real_shells=options.real_shells,
            reciprocal_shells=options.reciprocal_shells,
            ab5=self.ab5,
            shell_tolerance=float(options.shell_tolerance),
            max_shells=int(options.max_shells),
            dtype=self.dtype,
            block_cache=self._ewald_block_cache if self.cache_blocks else None,
            workspace=self._workspace(),
            contraction_tensor=self._contraction_tensor(),
        )

    def apply(self, x: Array) -> Array:
        """Apply the configured periodic coupling model to stacked coefficients."""
        method = self.periodic.options.method
        if method == "directsum":
            return apply_periodic_direct_sum(
                lmax=int(self.lmax),
                k=float(self.k),
                positions=self.positions,
                x=x,
                lattice=self.periodic.lattice,
                k_parallel=self.k_parallel,
                window=int(self.periodic.options.directsum_window),
                ab5=self.ab5,
                dtype=self.dtype,
            )
        if method == "rayleigh":
            return self._apply_rayleigh(x)
        options = self.periodic.options
        workspace = self._workspace()
        return apply_periodic_ewald_sum(
            lmax=int(self.lmax),
            k=float(self.k),
            positions=self.positions,
            x=x,
            lattice=self.periodic.lattice,
            k_parallel=self.k_parallel,
            eta=self._ewald_eta(),
            real_shells=options.real_shells,
            reciprocal_shells=options.reciprocal_shells,
            ab5=self.ab5,
            shell_tolerance=float(options.shell_tolerance),
            max_shells=int(options.max_shells),
            dtype=self.dtype,
            block_cache=self._ewald_block_cache if self.cache_blocks else None,
            workspace=workspace,
            contraction_tensor=self._contraction_tensor(),
        )

    def populate(self, *, show_progress: bool = False) -> None:
        """Eagerly populate reusable periodic coupling data."""
        if self.periodic.options.method == "rayleigh":
            plan = self._rayleigh_plan()
            if show_progress:
                tqdm.write(
                    "[Rayleigh] plan "
                    f"reciprocal_orders={plan.n_modes_reciprocal} "
                    f"z_cut={plan.z_cut:.6g}"
                )
            self._self_block()
            self._populate_near_structural_sums(show_progress=show_progress)
            self._release_rayleigh_preparation_state()
            return
        if self.periodic.options.method != "ewald" or not self.cache_blocks:
            return
        pos = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        ns = pos.shape[0]
        options = self.periodic.options
        workspace = self._workspace()
        sources: Iterable[int] = range(ns)
        if show_progress:
            sources = tqdm(sources, total=ns, desc="Populate periodic W cache")
        fill_periodic_ewald_block_cache(
            cache=self._ewald_block_cache,
            lmax=int(self.lmax),
            k=float(self.k),
            positions=pos,
            lattice=self.periodic.lattice,
            k_parallel=self.k_parallel,
            eta=self._ewald_eta(),
            real_shells=options.real_shells,
            reciprocal_shells=options.reciprocal_shells,
            ab5=self.ab5,
            dtype=self.dtype,
            shell_tolerance=float(options.shell_tolerance),
            max_shells=int(options.max_shells),
            source_indices=sources,
            workspace=workspace,
            contraction_tensor=self._contraction_tensor(),
        )

    def iter_source_block_batches(
        self, *, show_progress: bool = False, for_dense_assembly: bool = False
    ) -> Iterable[SourceBlockBatch]:
        """Yield exact Ewald source-major blocks for dense assembly."""
        pos = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        options = self.periodic.options
        workspace = self._workspace()
        sources: Iterable[int] = range(pos.shape[0])
        if show_progress:
            sources = tqdm(sources, total=pos.shape[0], desc="Build periodic source blocks")
        for source_index in sources:
            source = int(source_index)
            if self.cache_blocks:
                fill_periodic_ewald_block_cache(
                    cache=self._ewald_block_cache,
                    lmax=int(self.lmax),
                    k=float(self.k),
                    positions=pos,
                    lattice=self.periodic.lattice,
                    k_parallel=self.k_parallel,
                    eta=self._ewald_eta(),
                    real_shells=options.real_shells,
                    reciprocal_shells=options.reciprocal_shells,
                    ab5=self.ab5,
                    dtype=self.dtype,
                    shell_tolerance=float(options.shell_tolerance),
                    max_shells=int(options.max_shells),
                    source_indices=(source,),
                    workspace=workspace,
                    contraction_tensor=self._contraction_tensor(),
                )
                blocks = np.stack(
                    [
                        self._ewald_block_cache[(destination, source)]
                        for destination in range(pos.shape[0])
                    ],
                    axis=0,
                )
            else:
                blocks = periodic_ewald_blocks_for_source(
                    source_index=source,
                    lmax=int(self.lmax),
                    k=float(self.k),
                    positions=pos,
                    lattice=self.periodic.lattice,
                    k_parallel=self.k_parallel,
                    eta=self._ewald_eta(),
                    real_shells=options.real_shells,
                    reciprocal_shells=options.reciprocal_shells,
                    ab5=self.ab5,
                    dtype=self.dtype,
                    shell_tolerance=float(options.shell_tolerance),
                    max_shells=int(options.max_shells),
                    workspace=workspace,
                    contraction_tensor=self._contraction_tensor(),
                )
            yield SourceBlockBatch(
                source_indices=(source,),
                blocks=np.asarray(blocks, dtype=self.dtype)[None, ...],
            )
            del blocks

    def supports_source_block_dense_assembly(self) -> bool:
        """Return whether exact Ewald source blocks match the configured apply."""
        return self.periodic.options.method == "ewald"


__all__ = ["PeriodicCouplingOperator"]
