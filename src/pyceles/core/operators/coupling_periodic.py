"""Periodic coupling-operator descriptors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from pyceles.core.periodic import PeriodicSpec
from pyceles.core.periodic.directsum import apply_periodic_direct_sum
from pyceles.core.periodic.ewald import (
    EwaldShellWorkspace,
    apply_periodic_ewald_sum,
    fill_periodic_ewald_block_cache,
    resolve_ewald_eta,
)
from pyceles.core.periodic.structural import translation_contraction_tensor

Array = np.ndarray


@dataclass
class PeriodicCouplingOperator:
    """Bloch-reduced periodic coupling descriptor for rectangular 2D lattices."""

    lmax: int
    k: float
    positions: Array
    ab5: Array
    periodic: PeriodicSpec
    k_parallel: Array
    dtype: np.dtype
    cache_blocks: bool = False
    _ewald_block_cache: dict[tuple[int, int], Array] = field(default_factory=dict)
    _ewald_shell_workspace: EwaldShellWorkspace | None = field(default=None, init=False, repr=False)
    _resolved_ewald_eta: float | None = field(default=None, init=False, repr=False)
    _structural_contraction_tensor: Array | None = field(default=None, init=False, repr=False)

    def _ewald_eta(self) -> float:
        """Return the effective Ewald split for this operator."""
        eta = self.periodic.options.eta
        if eta is not None:
            return float(eta)
        if self._resolved_ewald_eta is None:
            self._resolved_ewald_eta = resolve_ewald_eta(
                periodic=self.periodic,
                k=float(self.k),
                k_parallel=self.k_parallel,
                positions=self.positions,
                lmax=int(self.lmax),
            )
        return float(self._resolved_ewald_eta)

    def _contraction_tensor(self) -> Array:
        """Return the reusable tensor mapping scalar structural sums to W blocks."""
        tensor = self._structural_contraction_tensor
        if tensor is None:
            tensor = translation_contraction_tensor(
                lmax=int(self.lmax),
                ab5=self.ab5,
                dtype=np.result_type(self.ab5.dtype, self.dtype, np.complex64),
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

    def apply(self, x: Array) -> Array:
        """Apply the configured periodic coupling model to stacked coefficients."""
        if self.periodic.options.method == "directsum":
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
        """Eagerly populate reusable Ewald coupling blocks."""
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


__all__ = ["PeriodicCouplingOperator"]
