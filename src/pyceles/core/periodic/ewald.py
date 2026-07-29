"""Ewald structural constants for rectangular two-dimensional lattices."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pyceles.core.indexing import n_modes
from pyceles.core.lattice import RectangularLattice2D
from pyceles.core.spherical import (
    legendre_normalized_trigon,
    legendre_normalized_trigon_scalar,
)

from .scalar import (
    chebyshev_shell_indices,
    factorial_int,
    real_integral_sequence,
    reciprocal_gamma_with_zero_mask,
    same_plane_z_tolerance,
    structural_sum_m_normalization,
)
from .shells import (
    PeriodicEwaldConvergenceError,
    accumulate_lattice_shell_series,
    make_lattice_shell_control,
)
from .special import (
    shifted_delta_sequence,
    shifted_delta_sequence_batched,
    upper_gamma_sequence,
    upper_incomplete_gamma_int_or_halfint,
)
from .structural import (
    apply_structural_sums_to_vector,
    block_from_structural_sums,
    blocks_from_structural_sums,
)

Array = np.ndarray


@dataclass(frozen=True)
class _ReciprocalShellData:
    kgt: Array
    rho: Array
    phi: Array
    gamma: Array
    xarg: Array
    rayleigh_zero: Array


@dataclass(frozen=True)
class _RealShellData:
    shifts: Array
    shifts_xy: Array
    phase_xy: Array
    radii_xy: Array
    phi_xy: Array


@dataclass
class EwaldShellWorkspace:
    """Cache reusable shell geometry for periodic Ewald structural sums."""

    lattice: RectangularLattice2D
    k: float
    k_parallel: Array
    eta: float
    _reciprocal_shell_cache: dict[int, _ReciprocalShellData] = field(default_factory=dict)
    _real_shell_cache: dict[int, _RealShellData] = field(default_factory=dict)
    _upper_gamma_cache: dict[int, tuple[int, Array]] = field(default_factory=dict)
    _propagating_min_shell_cache: dict[float, int] = field(default_factory=dict)

    def reciprocal_shell(self, shell: int) -> _ReciprocalShellData:
        cached = self._reciprocal_shell_cache.get(int(shell))
        if cached is not None:
            return cached
        reciprocal = np.asarray(
            [
                p * self.lattice.b1 + q * self.lattice.b2
                for p, q in chebyshev_shell_indices(int(shell))
            ],
            dtype=float,
        )
        kgt = np.asarray(self.k_parallel, dtype=float).reshape(2)[None, :] + reciprocal
        rho = np.linalg.norm(kgt, axis=1)
        phi = np.arctan2(kgt[:, 1], kgt[:, 0])
        gamma, rayleigh_zero = reciprocal_gamma_with_zero_mask(float(self.k), rho)
        xarg = -(gamma * gamma) / (4.0 * float(self.eta) * float(self.eta))
        data = _ReciprocalShellData(
            kgt=np.asarray(kgt, dtype=float),
            rho=np.asarray(rho, dtype=float),
            phi=np.asarray(phi, dtype=float),
            gamma=np.asarray(gamma, dtype=np.complex128),
            xarg=np.asarray(xarg, dtype=np.complex128),
            rayleigh_zero=np.asarray(rayleigh_zero, dtype=bool),
        )
        self._reciprocal_shell_cache[int(shell)] = data
        return data

    def real_shell(self, shell: int) -> _RealShellData:
        cached = self._real_shell_cache.get(int(shell))
        if cached is not None:
            return cached
        shifts = np.asarray(
            [
                p * self.lattice.a1 + q * self.lattice.a2
                for p, q in chebyshev_shell_indices(int(shell))
            ],
            dtype=float,
        )
        shifts_xy = np.asarray(shifts[:, :2], dtype=float)
        phase_xy = np.exp(1j * (shifts_xy @ np.asarray(self.k_parallel, dtype=float).reshape(2)))
        radii_xy = np.linalg.norm(shifts_xy, axis=1)
        phi_xy = np.arctan2(shifts_xy[:, 1], shifts_xy[:, 0])
        data = _RealShellData(
            shifts=np.asarray(shifts, dtype=float),
            shifts_xy=np.asarray(shifts_xy, dtype=float),
            phase_xy=np.asarray(phase_xy, dtype=np.complex128),
            radii_xy=np.asarray(radii_xy, dtype=float),
            phi_xy=np.asarray(phi_xy, dtype=float),
        )
        self._real_shell_cache[int(shell)] = data
        return data

    def upper_gamma(self, shell: int, max_index: int) -> Array:
        shell_i = int(shell)
        max_i = int(max_index)
        cached = self._upper_gamma_cache.get(shell_i)
        if cached is not None and cached[0] >= max_i:
            return cached[1][:, : max_i + 1]
        data = self.reciprocal_shell(shell_i)
        out = np.asarray(upper_gamma_sequence(max_i, data.xarg), dtype=np.complex128)
        self._upper_gamma_cache[shell_i] = (max_i, out)
        return out

    def minimum_reciprocal_shell_for_propagating_orders(self, *, rayleigh_margin: float) -> int:
        margin = float(rayleigh_margin)
        cached = self._propagating_min_shell_cache.get(margin)
        if cached is not None:
            return int(cached)
        k_lim = float(self.k) * (1.0 + margin)
        kp = np.asarray(self.k_parallel, dtype=float).reshape(2)
        b1 = np.asarray(self.lattice.b1, dtype=float).reshape(2)
        b2 = np.asarray(self.lattice.b2, dtype=float).reshape(2)
        min_b = min(float(np.linalg.norm(b1)), float(np.linalg.norm(b2)))
        if min_b <= 0.0:
            raise ValueError("Reciprocal basis vectors must be nonzero.")
        g_bound = float(np.linalg.norm(kp)) + k_lim
        pq_max = int(np.ceil(g_bound / min_b)) + 1
        min_shell = 0
        for p in range(-pq_max, pq_max + 1):
            for q in range(-pq_max, pq_max + 1):
                g = float(p) * b1 + float(q) * b2
                if float(np.linalg.norm(kp + g)) <= (k_lim + 1.0e-15):
                    min_shell = max(min_shell, abs(int(p)), abs(int(q)))
        self._propagating_min_shell_cache[margin] = int(min_shell)
        return int(min_shell)


def _ensure_workspace(
    *,
    workspace: EwaldShellWorkspace | None,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
) -> EwaldShellWorkspace:
    if workspace is None:
        return EwaldShellWorkspace(
            lattice=lattice,
            k=float(k),
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            eta=float(eta),
        )
    return workspace


def default_ewald_eta(lattice: RectangularLattice2D) -> float:
    """Return the canonical dimensional Ewald split for a 2D unit cell.

    The value ``sqrt(pi / area)`` balances direct- and reciprocal-lattice
    shell scales for a rectangular 2D lattice.
    """
    return float(np.sqrt(np.pi / lattice.area))


_ETA_PROBE_GROWTH = math.sqrt(2.0)
_ETA_PROBE_MAX_K_FACTOR = 0.35
_ETA_PROBE_STABILITY_RTOL = 1.0e-3


@dataclass(frozen=True)
class EwaldShellCounts:
    """Resolved fixed shell counts for non-adaptive periodic evaluators.

    The NumPy reference path may still use adaptive shell accumulation when the
    user leaves ``real_shells``/``reciprocal_shells`` unset.  Accelerated paths
    such as the CuPy in-slab near-field evaluator need fixed shell ranges to
    avoid host/device synchronization after every shell.  This payload records
    the one-shot fixed counts chosen from the same periodic options.
    """

    real_shells: int
    reciprocal_shells: int
    selected_by: str


def _candidate_ewald_etas(
    *, canonical_eta: float, k: float, max_steps: int = 16
) -> tuple[float, ...]:
    """Return the intentionally bounded eta ladder used by the automatic selector.

    The selector is a stability preflight, not an optimizer.  Keep the ladder
    finite and k-scaled so that ``eta=None`` remains a cheap default.
    """
    canonical = float(canonical_eta)
    if not np.isfinite(canonical) or canonical <= 0.0:
        raise ValueError(f"`canonical_eta` must be finite and positive. Got {canonical_eta!r}.")
    eta_ceiling = max(canonical, _ETA_PROBE_MAX_K_FACTOR * abs(float(k)))
    candidates = [canonical]
    eta = canonical
    for _ in range(max(1, int(max_steps))):
        eta_next = eta * _ETA_PROBE_GROWTH
        if eta_next >= eta_ceiling:
            if eta_ceiling > candidates[-1] * (1.0 + 1.0e-12):
                candidates.append(eta_ceiling)
            break
        candidates.append(float(eta_next))
        eta = float(eta_next)
    return tuple(candidates)


def _nearest_periodic_delta_xy(delta_xy: Array, lattice: RectangularLattice2D) -> Array:
    out = np.asarray(delta_xy, dtype=float).reshape(2).copy()
    ax = float(np.linalg.norm(lattice.a1))
    ay = float(np.linalg.norm(lattice.a2))
    if ax > 0.0:
        out[0] -= ax * np.round(out[0] / ax)
    if ay > 0.0:
        out[1] -= ay * np.round(out[1] / ay)
    return out


def _append_eta_probe_offset(offsets: list[Array], offset: Array, *, atol: float = 1.0e-9) -> None:
    arr = np.asarray(offset, dtype=float).reshape(3)
    if float(np.linalg.norm(arr)) <= 0.0:
        return
    if not any(np.allclose(arr, old, rtol=0.0, atol=float(atol)) for old in offsets):
        offsets.append(arr)


def _representative_ewald_eta_offsets(
    *,
    positions: Array,
    lattice: RectangularLattice2D,
    k: float,
    max_offsets: int = 6,
) -> Array:
    """Return a compact, cheap set of offsets that stress eta stability.

    This production preflight deliberately avoids all-pairs searches and KD-tree
    dependencies. It combines z-extreme packing probes with two near-self
    synthetic offsets; this is enough to catch the large-area small-eta failures
    seen so far without turning eta selection into an optimization pass.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    ax = float(np.linalg.norm(lattice.a1))
    ay = float(np.linalg.norm(lattice.a2))
    cell = max(ax, ay, 1.0)
    offsets: list[Array] = []
    ns = int(pos.shape[0])
    if ns >= 2:
        order = np.argsort(pos[:, 2])
        z_sorted = pos[order, 2]
        dz = np.diff(z_sorted)
        nonzero = np.flatnonzero(dz > max(1.0e-9, 1.0e-12 * cell))
        if nonzero.size:
            iz = int(nonzero[np.argmin(dz[nonzero])])
            raw = pos[order[iz + 1]] - pos[order[iz]]
            dxy = _nearest_periodic_delta_xy(raw[:2], lattice)
            _append_eta_probe_offset(offsets, np.array([dxy[0], dxy[1], raw[2]], dtype=float))

        z_low = int(np.argmin(pos[:, 2]))
        z_high = int(np.argmax(pos[:, 2]))
        if z_low != z_high:
            raw = pos[z_high] - pos[z_low]
            dxy = _nearest_periodic_delta_xy(raw[:2], lattice)
            _append_eta_probe_offset(offsets, np.array([dxy[0], dxy[1], raw[2]], dtype=float))

    # Single-particle and near-coincident periodic configurations do not expose
    # these offsets through actual pairs, but they are exactly where a too-small
    # eta can make same-plane and small-shift terms numerically fragile.
    wavelength = 2.0 * math.pi / max(float(abs(k)), 1.0e-300)
    near = max(0.02 * wavelength, 1.0e-6 * cell)
    _append_eta_probe_offset(offsets, np.array([near, 0.0, 0.0], dtype=float))
    _append_eta_probe_offset(offsets, np.array([0.0, 0.0, near], dtype=float))
    return np.asarray(offsets[: int(max_offsets)], dtype=float).reshape(-1, 3)


def _eta_probe_vector(
    *,
    eta: float,
    offsets: Array,
    lmax_struct: int,
    k: float,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    real_shells: int,
    reciprocal_shells: int,
    shell_tolerance: float,
    max_shells: int,
) -> Array | None:
    workspace = EwaldShellWorkspace(
        lattice=lattice,
        k=float(k),
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        eta=float(eta),
    )
    with np.errstate(all="ignore"):
        try:
            sums = ewald_structural_sums_2d_batch(
                lmax_struct=int(lmax_struct),
                k=float(k),
                destinations=np.asarray(offsets, dtype=float).reshape(-1, 3),
                source=np.zeros(3, dtype=float),
                lattice=lattice,
                k_parallel=k_parallel,
                eta=float(eta),
                real_shells=int(real_shells),
                reciprocal_shells=int(reciprocal_shells),
                shell_tolerance=float(shell_tolerance),
                max_shells=int(max_shells),
                dtype=np.complex128,
                workspace=workspace,
            )
        except (ArithmeticError, ValueError, PeriodicEwaldConvergenceError):
            return None
    n_offsets = int(np.asarray(offsets).reshape(-1, 3).shape[0])
    flat = np.asarray(sums, dtype=np.complex128).reshape(n_offsets, -1)
    if not np.all(np.isfinite(flat)):
        return None
    return flat


def _eta_probe_relative_difference(a: Array, b: Array) -> float:
    lhs = np.asarray(a, dtype=np.complex128).reshape(a.shape[0], -1)
    rhs = np.asarray(b, dtype=np.complex128).reshape(b.shape[0], -1)
    diff = np.linalg.norm(lhs - rhs, axis=1)
    lhs_norm = np.linalg.norm(lhs, axis=1)
    rhs_norm = np.linalg.norm(rhs, axis=1)
    scale = np.maximum.reduce((lhs_norm, rhs_norm, np.ones_like(diff)))
    return float(np.max(diff / scale))


def select_ewald_eta(
    *,
    lattice: RectangularLattice2D,
    k: float,
    k_parallel: Array,
    positions: Array,
    lmax: int,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    real_shells: int | None = None,
    reciprocal_shells: int | None = None,
    stability_rtol: float = _ETA_PROBE_STABILITY_RTOL,
) -> float:
    """Choose an automatic Ewald split for this lattice and particle packing.

    The selector starts from the canonical ``sqrt(pi / area)`` value and raises
    eta only when a cheap structural-sum preflight shows that the current value
    is unstable against the next geometric candidate.
    """
    canonical = default_ewald_eta(lattice)
    candidates = _candidate_ewald_etas(canonical_eta=canonical, k=float(k))
    if len(candidates) == 1:
        return float(canonical)
    offsets = _representative_ewald_eta_offsets(
        positions=positions,
        lattice=lattice,
        k=float(k),
    )
    if offsets.size == 0:
        return float(canonical)
    # Keep the preflight intentionally cheap: low-order structural sums are
    # sufficient to catch the catastrophic eta regimes seen in large cells.
    lmax_probe = max(1, min(int(lmax), 2))
    probe_real = int(real_shells) if real_shells is not None else min(12, max(1, int(max_shells)))
    probe_recip = (
        int(reciprocal_shells)
        if reciprocal_shells is not None
        else min(
            12,
            max(1, int(max_shells)),
        )
    )
    previous_eta: float | None = None
    previous_vec: Array | None = None
    best_eta: float | None = None
    best_rel = math.inf
    for eta in candidates:
        vec = _eta_probe_vector(
            eta=float(eta),
            offsets=offsets,
            lmax_struct=lmax_probe,
            k=float(k),
            lattice=lattice,
            k_parallel=k_parallel,
            real_shells=probe_real,
            reciprocal_shells=probe_recip,
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
        )
        if vec is None:
            continue
        if previous_vec is not None and previous_eta is not None:
            rel = _eta_probe_relative_difference(previous_vec, vec)
            if rel < best_rel:
                best_rel = rel
                best_eta = float(previous_eta)
            if rel <= float(stability_rtol):
                return float(previous_eta)
        previous_eta = float(eta)
        previous_vec = vec
    if best_eta is not None:
        return float(best_eta)
    if previous_eta is not None:
        return float(previous_eta)
    return float(canonical)


def resolve_ewald_eta(
    *,
    periodic,
    k: float,
    k_parallel: Array,
    positions: Array,
    lmax: int,
) -> float:
    """Return the effective Ewald split for one periodic configuration.

    This is the single internal policy helper used by the periodic operator and
    by postprocessing evaluators.  Explicit ``periodic.options.eta`` values are
    honored exactly; ``None`` delegates to :func:`select_ewald_eta` so every
    periodic code path uses the same automatic preflight.
    """
    eta = periodic.options.eta
    if eta is not None:
        return float(eta)
    return float(
        select_ewald_eta(
            lattice=periodic.lattice,
            k=float(k),
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            positions=np.asarray(positions, dtype=float).reshape(-1, 3),
            lmax=int(lmax),
            shell_tolerance=float(periodic.options.shell_tolerance),
            max_shells=int(periodic.options.max_shells),
            real_shells=periodic.options.real_shells,
            reciprocal_shells=periodic.options.reciprocal_shells,
        )
    )


def _reference_shell_probe_vector(
    *,
    eta: float,
    offsets: Array,
    lmax_struct: int,
    k: float,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    shell_tolerance: float,
    max_shells: int,
) -> Array | None:
    """Return a fixed max-shell reference vector for shell-count selection."""
    return _eta_probe_vector(
        eta=float(eta),
        offsets=np.asarray(offsets, dtype=float).reshape(-1, 3),
        lmax_struct=int(lmax_struct),
        k=float(k),
        lattice=lattice,
        k_parallel=k_parallel,
        real_shells=int(max_shells),
        reciprocal_shells=int(max_shells),
        shell_tolerance=float(shell_tolerance),
        max_shells=int(max_shells),
    )


def _select_one_shell_count(
    *,
    eta: float,
    offsets: Array,
    lmax_struct: int,
    k: float,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    shell_tolerance: float,
    max_shells: int,
    reference: Array,
    which: str,
    fixed_other: int,
) -> int:
    """Choose one fixed shell count by comparison to a max-shell probe."""
    which_s = str(which)
    if which_s not in {"real", "reciprocal"}:
        raise ValueError(f"unknown shell selector {which!r}")
    tol = float(shell_tolerance)
    for count in range(0, int(max_shells) + 1):
        real_count = count if which_s == "real" else int(fixed_other)
        recip_count = count if which_s == "reciprocal" else int(fixed_other)
        vec = _eta_probe_vector(
            eta=float(eta),
            offsets=np.asarray(offsets, dtype=float).reshape(-1, 3),
            lmax_struct=int(lmax_struct),
            k=float(k),
            lattice=lattice,
            k_parallel=k_parallel,
            real_shells=int(real_count),
            reciprocal_shells=int(recip_count),
            shell_tolerance=tol,
            max_shells=int(max_shells),
        )
        if vec is None:
            continue
        if _eta_probe_relative_difference(vec, reference) <= tol:
            return int(count)
    return int(max_shells)


def resolve_ewald_shell_counts(
    *,
    periodic,
    k: float,
    k_parallel: Array,
    positions: Array,
    lmax: int,
    eta: float | None = None,
) -> EwaldShellCounts:
    """Resolve fixed shell counts for accelerated non-adaptive evaluators.

    Explicit user counts are honored exactly.  If either count is unset, this
    helper performs a cheap one-shot probe on representative offsets and compares
    fixed shell truncations to the ``max_shells`` reference using
    ``periodic.options.shell_tolerance``.  This centralizes the policy needed by
    CuPy paths without changing the NumPy reference path, which can still use
    adaptive accumulation directly.
    """
    options = periodic.options
    max_count = int(options.max_shells)
    explicit_real = options.real_shells is not None
    explicit_recip = options.reciprocal_shells is not None
    if explicit_real and explicit_recip:
        return EwaldShellCounts(
            real_shells=max(0, int(options.real_shells)),
            reciprocal_shells=max(0, int(options.reciprocal_shells)),
            selected_by="explicit",
        )

    eta_f = float(
        eta
        if eta is not None
        else resolve_ewald_eta(
            periodic=periodic,
            k=float(k),
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            positions=np.asarray(positions, dtype=float).reshape(-1, 3),
            lmax=int(lmax),
        )
    )
    offsets = _representative_ewald_eta_offsets(
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        lattice=periodic.lattice,
        k=float(k),
    )
    if offsets.size == 0:
        fallback = max(1, min(12, max_count))
        return EwaldShellCounts(
            real_shells=max(0, int(options.real_shells)) if explicit_real else fallback,
            reciprocal_shells=max(0, int(options.reciprocal_shells))
            if explicit_recip
            else fallback,
            selected_by="fallback",
        )

    lmax_probe = max(1, min(int(lmax), 2))
    reference = _reference_shell_probe_vector(
        eta=eta_f,
        offsets=offsets,
        lmax_struct=lmax_probe,
        k=float(k),
        lattice=periodic.lattice,
        k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
        shell_tolerance=float(options.shell_tolerance),
        max_shells=max_count,
    )
    if reference is None:
        fallback = max_count
        return EwaldShellCounts(
            real_shells=max(0, int(options.real_shells)) if explicit_real else fallback,
            reciprocal_shells=max(0, int(options.reciprocal_shells))
            if explicit_recip
            else fallback,
            selected_by="max_shells_reference_failed",
        )

    real_count = (
        max(0, int(options.real_shells))
        if explicit_real
        else _select_one_shell_count(
            eta=eta_f,
            offsets=offsets,
            lmax_struct=lmax_probe,
            k=float(k),
            lattice=periodic.lattice,
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            shell_tolerance=float(options.shell_tolerance),
            max_shells=max_count,
            reference=reference,
            which="real",
            fixed_other=max_count if not explicit_recip else max(0, int(options.reciprocal_shells)),
        )
    )
    reciprocal_count = (
        max(0, int(options.reciprocal_shells))
        if explicit_recip
        else _select_one_shell_count(
            eta=eta_f,
            offsets=offsets,
            lmax_struct=lmax_probe,
            k=float(k),
            lattice=periodic.lattice,
            k_parallel=np.asarray(k_parallel, dtype=float).reshape(2),
            shell_tolerance=float(options.shell_tolerance),
            max_shells=max_count,
            reference=reference,
            which="reciprocal",
            fixed_other=max_count if not explicit_real else max(0, int(options.real_shells)),
        )
    )
    return EwaldShellCounts(
        real_shells=int(real_count),
        reciprocal_shells=int(reciprocal_count),
        selected_by="probe",
    )


def _same_plane_reciprocal_sum(
    degree: int,
    order: int,
    *,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int | None,
    c_xy: Array,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    workspace: EwaldShellWorkspace | None = None,
) -> complex:
    l = int(degree)
    m = int(order)
    if (l - abs(m)) % 2:
        return 0.0 + 0.0j
    root = math.sqrt(2 * l + 1) * math.sqrt(factorial_int(l - m)) * math.sqrt(factorial_int(l + m))
    prefactor = (1j) ** m * root / (lattice.area * float(k) * (2.0 * float(k)) ** l)

    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )
    cxy = np.asarray(c_xy, dtype=float).reshape(2)
    min_recip_shell = ws.minimum_reciprocal_shell_for_propagating_orders(rayleigh_margin=1.0e-12)
    control = make_lattice_shell_control(
        shells=shells,
        max_shells=int(max_shells),
        shell_tolerance=float(shell_tolerance),
        name="reciprocal_shells",
        min_converged_shell=min_recip_shell,
    )

    def shell_increment(shell: int) -> complex:
        shell_data = ws.reciprocal_shell(shell)
        kgt = shell_data.kgt
        rho = shell_data.rho
        phi = shell_data.phi
        gamma = shell_data.gamma
        n_values = np.arange((l - abs(m)) // 2 + 1, dtype=np.int64)
        gamma_fun = ws.upper_gamma(shell, int(n_values[-1]))
        inner = np.zeros_like(gamma, dtype=np.complex128)
        for n in n_values:
            denom = (
                factorial_int(n) * factorial_int((l + m) // 2 - n) * factorial_int((l - m) // 2 - n)
            )
            inner += (
                gamma_fun[:, int(n)] * gamma ** (2 * int(n) - 1) * rho ** (l - 2 * int(n)) / denom
            )
        return complex(np.sum(np.exp(-1j * (kgt @ cxy)) * np.exp(1j * m * phi) * inner))

    acc = accumulate_lattice_shell_series(
        control=control,
        name="reciprocal_shells",
        zero=0.0 + 0.0j,
        evaluate_shell=shell_increment,
    )
    return complex(prefactor * acc)


def _same_plane_real_sum(
    degree: int,
    order: int,
    *,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int | None,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    workspace: EwaldShellWorkspace | None = None,
) -> complex:
    l = int(degree)
    m = int(order)
    if (l - abs(m)) % 2:
        return 0.0 + 0.0j
    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )

    frac = (
        -1j
        * ((-1.0) ** ((l + m) // 2))
        / (2.0 ** (l + 1) * math.pi * factorial_int((l - m) // 2) * factorial_int((l + m) // 2))
    )
    root = math.sqrt(2 * l + 1) * math.sqrt(factorial_int(l - m)) * math.sqrt(factorial_int(l + m))
    prefactor = frac * root

    control = make_lattice_shell_control(
        shells=shells,
        max_shells=int(max_shells),
        shell_tolerance=float(shell_tolerance),
        name="real_shells",
        start_shell=1,
    )

    def shell_increment(shell: int) -> complex:
        shell_data = ws.real_shell(shell)
        radii = shell_data.radii_xy
        if radii.size == 0:
            return 0.0 + 0.0j
        phi = shell_data.phi_xy
        integral = (float(k) * float(k) / 4.0) ** (l + 0.5) * real_integral_sequence(
            l,
            float(eta),
            float(k),
            radii,
        )
        return complex(
            np.sum(
                shell_data.phase_xy
                * np.exp(1j * m * (phi + math.pi))
                / float(k)
                * (2.0 * radii / float(k)) ** l
                * integral
            )
        )

    acc = accumulate_lattice_shell_series(
        control=control,
        name="real_shells",
        zero=0.0 + 0.0j,
        evaluate_shell=shell_increment,
    )
    return complex(prefactor * acc)


def _self_correction(k: float, eta: float) -> complex:
    """Return the same-particle Ewald central-point correction."""
    eta_f = float(eta)
    x = -(float(k) * float(k)) / (4.0 * eta_f * eta_f)
    return complex(upper_incomplete_gamma_int_or_halfint(-0.5, x) / (4.0 * math.pi))


def ewald_self_correction(k: float, eta: float) -> complex:
    """Return the public-internal same-particle Ewald central-point correction."""
    return _self_correction(float(k), float(eta))


def _shifted_reciprocal_sum(
    degree: int,
    order: int,
    *,
    rvec: Array,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int | None,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    workspace: EwaldShellWorkspace | None = None,
) -> complex:
    l = int(degree)
    m = int(order)
    c = -np.asarray(rvec, dtype=float).reshape(3)
    if c[2] == 0.0:
        return _same_plane_reciprocal_sum(
            l,
            m,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=float(eta),
            shells=shells,
            c_xy=c[:2],
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
            workspace=workspace,
        )

    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )
    root = math.sqrt(2 * l + 1) * math.sqrt(factorial_int(l - m)) * math.sqrt(factorial_int(l + m))
    prefactor = (-1j) ** m * root / (((-2.0) ** l) * lattice.area * float(k) * float(k))
    min_recip_shell = ws.minimum_reciprocal_shell_for_propagating_orders(rayleigh_margin=1.0e-12)
    control = make_lattice_shell_control(
        shells=shells,
        max_shells=int(max_shells),
        shell_tolerance=float(shell_tolerance),
        name="reciprocal_shells",
        min_converged_shell=min_recip_shell,
    )

    def shell_increment(shell: int) -> complex:
        shell_data = ws.reciprocal_shell(shell)
        kgt = shell_data.kgt
        rho = shell_data.rho
        phi = shell_data.phi
        gamma = shell_data.gamma
        n_values = np.arange(0, l - abs(m) + 1, dtype=np.int64)
        inner = np.zeros((rho.size, n_values.size), dtype=np.complex128)
        for n in n_values:
            s_values = np.arange(int(n), min(l - abs(m), 2 * int(n)) + 1, dtype=np.int64)
            if (l - abs(m)) % 2:
                s_values = s_values[s_values % 2 == 1]
            else:
                s_values = s_values[s_values % 2 == 0]
            if s_values.size == 0:
                continue
            terms = np.zeros_like(rho, dtype=np.complex128)
            for s in s_values:
                denom = (
                    factorial_int(2 * int(n) - int(s))
                    * factorial_int(int(s) - int(n))
                    * factorial_int((l + abs(m) - int(s)) // 2)
                    * factorial_int((l - abs(m) - int(s)) // 2)
                )
                terms += (
                    (-float(k) * c[2]) ** (2 * int(n) - int(s))
                    * (rho / float(k)) ** (l - int(s))
                    / denom
                )
            inner[:, int(n)] = terms
        delta_order = int(n_values[-1])
        delta = shifted_delta_sequence(
            delta_order,
            gamma,
            float(c[2]),
            float(eta),
            upper_gamma_provider=lambda max_index: ws.upper_gamma(shell, max_index),
            series_exclusion=shell_data.rayleigh_zero,
        )
        return complex(
            np.sum(
                np.exp(-1j * (kgt @ c[:2]))
                * np.exp(1j * m * phi)
                * np.sum((gamma / float(k))[:, None] ** (2 * n_values - 1) * delta * inner, axis=1)
            )
        )

    acc = accumulate_lattice_shell_series(
        control=control,
        name="reciprocal_shells",
        zero=0.0 + 0.0j,
        evaluate_shell=shell_increment,
    )
    return complex(prefactor * acc)


def _shifted_real_sum(
    degree: int,
    order: int,
    *,
    rvec: Array,
    k: float,
    k_parallel: Array,
    lattice: RectangularLattice2D,
    eta: float,
    shells: int | None,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    workspace: EwaldShellWorkspace | None = None,
) -> complex:
    l = int(degree)
    m = int(order)
    c = -np.asarray(rvec, dtype=float).reshape(3)
    if c[2] == 0.0 and (l - abs(m)) % 2:
        return 0.0 + 0.0j

    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )
    kp = np.asarray(ws.k_parallel, dtype=float).reshape(2)
    control = make_lattice_shell_control(
        shells=shells,
        max_shells=int(max_shells),
        shell_tolerance=float(shell_tolerance),
        name="real_shells",
    )

    def shell_increment(shell: int) -> complex:
        shell_data = ws.real_shell(shell)
        shifted = -(shell_data.shifts + c)
        radii = np.linalg.norm(shifted, axis=1)
        mask = radii > 0.0
        if not np.any(mask):
            return 0.0 + 0.0j
        shifts_xy = shell_data.shifts_xy[mask]
        shifted = shifted[mask]
        radii = radii[mask]
        ct = shifted[:, 2] / radii
        st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
        phi = np.arctan2(shifted[:, 1], shifted[:, 0])

        angular = np.zeros_like(radii, dtype=np.complex128)
        for idx, (ct_i, st_i, phi_i) in enumerate(zip(ct, st, phi, strict=True)):
            plm = legendre_normalized_trigon_scalar(float(ct_i), float(st_i), max(1, l))
            angular[idx] = (
                plm[l, abs(m)] * np.exp(1j * m * float(phi_i)) / structural_sum_m_normalization(m)
            )

        integral = (0.5) ** (l + 1.5) * real_integral_sequence(
            l,
            float(eta),
            float(k),
            radii,
        )
        return complex(
            np.sum(np.exp(1j * (shifts_xy @ kp)) * (float(k) * radii) ** l * angular * integral)
        )

    acc = accumulate_lattice_shell_series(
        control=control,
        name="real_shells",
        zero=0.0 + 0.0j,
        evaluate_shell=shell_increment,
    )
    return complex(-1j * math.sqrt(2.0 / math.pi) * acc)


def ewald_structural_constant_2d(
    degree: int,
    order: int,
    *,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    exclude_zero_shift: bool = False,
    workspace: EwaldShellWorkspace | None = None,
) -> complex:
    """Evaluate one pyceles-normalized scalar Ewald structural constant.

    The constant matches the scalar table consumed by
    `block_from_structural_sums`: degree `L`, order `M` stores the lattice sum
    of `h_L^(1)(k r) P_L^|M|(cos theta) exp(i M phi)` with the Bloch phase
    convention used by the periodic direct-sum oracle.
    """
    l = int(degree)
    m = int(order)
    if l < 0:
        raise ValueError(f"`degree` must be >= 0. Got {degree!r}.")
    if abs(m) > l:
        raise ValueError(f"`order` must satisfy |order| <= degree. Got {(degree, order)!r}.")
    eta_f = float(eta)
    if not np.isfinite(eta_f) or eta_f <= 0.0:
        raise ValueError(f"`eta` must be finite and positive. Got {eta!r}.")
    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=eta_f,
    )
    rvec = np.asarray(destination, dtype=float).reshape(3) - np.asarray(
        source, dtype=float
    ).reshape(3)
    coordinate_scale = float(
        max(
            np.max(np.abs(np.asarray(destination, dtype=float).reshape(3))),
            np.max(np.abs(np.asarray(source, dtype=float).reshape(3))),
        )
    )
    if abs(float(rvec[2])) <= same_plane_z_tolerance(float(k), coordinate_scale=coordinate_scale):
        rvec = np.asarray(rvec, dtype=float).copy()
        rvec[2] = 0.0
    is_self = bool(exclude_zero_shift) and float(np.linalg.norm(rvec)) == 0.0
    if is_self:
        value = _same_plane_reciprocal_sum(
            l,
            m,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=reciprocal_shells,
            c_xy=np.zeros(2, dtype=float),
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
            workspace=ws,
        ) + _same_plane_real_sum(
            l,
            m,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=real_shells,
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
            workspace=ws,
        )
        if l == 0:
            value += _self_correction(float(k), eta_f)
    else:
        value = _shifted_reciprocal_sum(
            l,
            m,
            rvec=rvec,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=reciprocal_shells,
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
            workspace=ws,
        ) + _shifted_real_sum(
            l,
            m,
            rvec=rvec,
            k=float(k),
            k_parallel=k_parallel,
            lattice=lattice,
            eta=eta_f,
            shells=real_shells,
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
            workspace=ws,
        )
    return complex(structural_sum_m_normalization(m) * value)


def ewald_structural_sums_2d(
    *,
    lmax: int,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    exclude_zero_shift: bool = False,
    dtype: npt.DTypeLike = np.complex128,
    workspace: EwaldShellWorkspace | None = None,
) -> Array:
    """Evaluate a scalar structural table using rectangular-lattice Ewald sums."""
    out_dtype = np.dtype(dtype)
    order = 2 * int(lmax)
    if order < 0:
        raise ValueError(f"`lmax` must be >= 0. Got {lmax!r}.")
    sums = np.zeros((order + 1, 2 * order + 1), dtype=np.complex128)
    offset = order
    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )
    for degree in range(order + 1):
        for m in range(-degree, degree + 1):
            sums[degree, m + offset] = ewald_structural_constant_2d(
                degree,
                m,
                k=float(k),
                destination=destination,
                source=source,
                lattice=lattice,
                k_parallel=k_parallel,
                eta=float(eta),
                real_shells=real_shells,
                reciprocal_shells=reciprocal_shells,
                shell_tolerance=float(shell_tolerance),
                max_shells=int(max_shells),
                exclude_zero_shift=bool(exclude_zero_shift),
                workspace=ws,
            )
    return np.asarray(sums, dtype=out_dtype)


def ewald_structural_sums_2d_batch(
    *,
    lmax_struct: int,
    k: float,
    destinations: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    dtype: npt.DTypeLike = np.complex128,
    workspace: EwaldShellWorkspace | None = None,
) -> Array:
    """Evaluate batched pyceles-normalized scalar structural sums.

    This helper owns the vectorized CPU reference implementation used by
    periodic in-slab near-field evaluation. Keeping it in ``core.periodic``
    makes the same structural-sum contract available to future accelerated
    backends without tying it to one postprocessing module.
    """
    out_dtype = np.dtype(dtype)
    dest = np.asarray(destinations, dtype=float).reshape(-1, 3)
    source_arr = np.asarray(source, dtype=float).reshape(3)
    n_points = int(dest.shape[0])
    order = 2 * int(lmax_struct)
    if order < 0:
        raise ValueError(f"`lmax_struct` must be >= 0. Got {lmax_struct!r}.")
    offset = order
    sums = np.zeros((n_points, order + 1, 2 * order + 1), dtype=np.complex128)
    if n_points == 0:
        return np.asarray(sums, dtype=out_dtype)
    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )
    c = source_arr[None, :] - dest
    cxy = np.asarray(c[:, :2], dtype=float)
    cz = np.asarray(c[:, 2], dtype=float)
    coordinate_scale = float(max(np.max(np.abs(source_arr)), np.max(np.abs(dest))))
    same_plane_atol = same_plane_z_tolerance(float(k), coordinate_scale=coordinate_scale)
    same_plane_points = np.abs(cz) <= same_plane_atol
    if np.any(same_plane_points):
        c = np.asarray(c, dtype=float).copy()
        c[same_plane_points, 2] = 0.0
        cz = np.asarray(c[:, 2], dtype=float)

    reciprocal_sums = np.zeros_like(sums)
    max_same_n = max(0, order // 2)
    min_recip_shell = ws.minimum_reciprocal_shell_for_propagating_orders(rayleigh_margin=1.0e-12)
    reciprocal_control = make_lattice_shell_control(
        shells=reciprocal_shells,
        max_shells=int(max_shells),
        shell_tolerance=float(shell_tolerance),
        name="reciprocal_shells",
        min_converged_shell=min_recip_shell,
    )

    def reciprocal_increment(shell: int) -> Array:
        shell_data = ws.reciprocal_shell(shell)
        kgt = shell_data.kgt
        rho = shell_data.rho
        phi = shell_data.phi
        gamma = shell_data.gamma
        phase_all = np.exp(-1j * (cxy @ kgt.T))
        exp_m_phi = {m: np.exp(1j * m * phi) for m in range(-order, order + 1)}
        inc = np.zeros_like(sums)

        same_mask = same_plane_points
        if np.any(same_mask):
            phase = phase_all[same_mask]
            gamma_fun = ws.upper_gamma(shell, max_same_n)
            for degree in range(order + 1):
                for m in range(-degree, degree + 1):
                    if (degree - abs(m)) % 2:
                        continue
                    root = (
                        np.sqrt(2 * degree + 1.0)
                        * math.sqrt(factorial_int(degree - m))
                        * math.sqrt(factorial_int(degree + m))
                    )
                    prefactor = (
                        (1j) ** m * root / (lattice.area * float(k) * (2.0 * float(k)) ** degree)
                    )
                    n_vals = np.arange((degree - abs(m)) // 2 + 1, dtype=np.int64)
                    inner = np.zeros_like(gamma, dtype=np.complex128)
                    for n in n_vals:
                        denom = (
                            factorial_int(n)
                            * factorial_int((degree + m) // 2 - n)
                            * factorial_int((degree - m) // 2 - n)
                        )
                        inner += (
                            gamma_fun[:, int(n)]
                            * gamma ** (2 * int(n) - 1)
                            * rho ** (degree - 2 * int(n))
                            / denom
                        )
                    vec = exp_m_phi[m] * inner
                    inc[same_mask, degree, m + offset] += (
                        structural_sum_m_normalization(m) * prefactor * (phase @ vec)
                    )

        shifted_mask = ~same_mask
        if np.any(shifted_mask):
            phase = phase_all[shifted_mask]
            cz_shifted = np.asarray(cz[shifted_mask], dtype=float)
            delta_full = shifted_delta_sequence_batched(
                order,
                gamma,
                cz_shifted,
                float(eta),
                upper_gamma_provider=lambda max_index: ws.upper_gamma(
                    shell,
                    max(max_same_n, max_index),
                ),
                series_exclusion=shell_data.rayleigh_zero,
            )
            gamma_over_k = gamma / float(k)
            for degree in range(order + 1):
                for m in range(-degree, degree + 1):
                    root = (
                        np.sqrt(2 * degree + 1.0)
                        * math.sqrt(factorial_int(degree - m))
                        * math.sqrt(factorial_int(degree + m))
                    )
                    prefactor = (
                        (-1j) ** m
                        * root
                        / (((-2.0) ** degree) * lattice.area * float(k) * float(k))
                    )
                    n_vals = np.arange(0, degree - abs(m) + 1, dtype=np.int64)
                    if n_vals.size == 0:
                        continue
                    acc = np.zeros((cz_shifted.size, rho.size), dtype=np.complex128)
                    for n in n_vals:
                        s_vals = np.arange(
                            int(n), min(degree - abs(m), 2 * int(n)) + 1, dtype=np.int64
                        )
                        s_vals = (
                            s_vals[s_vals % 2 == 1]
                            if (degree - abs(m)) % 2
                            else s_vals[s_vals % 2 == 0]
                        )
                        if s_vals.size == 0:
                            continue
                        terms = np.zeros((cz_shifted.size, rho.size), dtype=np.complex128)
                        for s in s_vals:
                            denom = (
                                factorial_int(2 * int(n) - int(s))
                                * factorial_int(int(s) - int(n))
                                * factorial_int((degree + abs(m) - int(s)) // 2)
                                * factorial_int((degree - abs(m) - int(s)) // 2)
                            )
                            terms += (
                                (-float(k) * cz_shifted[:, None]) ** (2 * int(n) - int(s))
                                * (rho[None, :] / float(k)) ** (degree - int(s))
                                / denom
                            )
                        acc += (
                            gamma_over_k[None, :] ** (2 * int(n) - 1)
                            * delta_full[:, :, int(n)]
                            * terms
                        )
                    vec = exp_m_phi[m][None, :] * acc
                    vals = (
                        structural_sum_m_normalization(m)
                        * prefactor
                        * np.sum(
                            phase * vec,
                            axis=1,
                        )
                    )
                    inc[shifted_mask, degree, m + offset] += vals
        return inc

    reciprocal_sums = np.asarray(
        accumulate_lattice_shell_series(
            control=reciprocal_control,
            name="reciprocal_shells",
            zero=np.zeros_like(sums),
            evaluate_shell=reciprocal_increment,
        ),
        dtype=np.complex128,
    )

    real_sums = np.zeros_like(sums)
    real_control = make_lattice_shell_control(
        shells=real_shells,
        max_shells=int(max_shells),
        shell_tolerance=float(shell_tolerance),
        name="real_shells",
    )

    def real_increment(shell: int) -> Array:
        shell_data = ws.real_shell(shell)
        shifted = -(shell_data.shifts[None, :, :] + c[:, None, :])
        radii = np.linalg.norm(shifted, axis=2)
        inc = np.zeros_like(sums)
        mask = radii > 0.0
        if np.any(mask):
            point_idx, shell_idx = np.nonzero(mask)
            shifted_valid = shifted[point_idx, shell_idx, :]
            radii_valid = radii[point_idx, shell_idx]
            ct = shifted_valid[:, 2] / radii_valid
            st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
            phi = np.arctan2(shifted_valid[:, 1], shifted_valid[:, 0])
            plm = np.asarray(legendre_normalized_trigon(ct, st, max(1, order)), dtype=np.float64)
            phase_shell = np.asarray(shell_data.phase_xy[shell_idx], dtype=np.complex128)
            exp_m_phi = {m: np.exp(1j * m * phi) for m in range(-order, order + 1)}
            kz_r = float(k) * radii_valid
            for degree in range(order + 1):
                integral = (0.5) ** (degree + 1.5) * real_integral_sequence(
                    degree, float(eta), float(k), radii_valid
                )
                radial = phase_shell * kz_r**degree * integral
                for m in range(-degree, degree + 1):
                    angular = (
                        plm[degree, abs(m), :] * exp_m_phi[m] / structural_sum_m_normalization(m)
                    )
                    contrib = -1j * np.sqrt(2.0 / np.pi) * radial * angular
                    if (degree - abs(m)) % 2:
                        # Exact same-plane real-space terms vanish for odd
                        # degree-|m|.  Apply this per point, not per batch:
                        # regular xz/yz diagnostic slices often contain a
                        # same-z row embedded in otherwise shifted rows.
                        contrib = np.where(same_plane_points[point_idx], 0.0 + 0.0j, contrib)
                    np.add.at(
                        inc[:, degree, m + offset],
                        point_idx,
                        structural_sum_m_normalization(m) * contrib,
                    )
        return inc

    real_sums = np.asarray(
        accumulate_lattice_shell_series(
            control=real_control,
            name="real_shells",
            zero=np.zeros_like(sums),
            evaluate_shell=real_increment,
        ),
        dtype=np.complex128,
    )

    sums = reciprocal_sums + real_sums
    return np.asarray(sums, dtype=out_dtype)


def periodic_ewald_block(
    *,
    lmax: int,
    k: float,
    destination: Array,
    source: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    exclude_zero_shift: bool = False,
    workspace: EwaldShellWorkspace | None = None,
) -> Array:
    """Assemble one periodic SVWF coupling block from Ewald structural constants."""
    sums = ewald_structural_sums_2d(
        lmax=int(lmax),
        k=float(k),
        destination=destination,
        source=source,
        lattice=lattice,
        k_parallel=k_parallel,
        eta=float(eta),
        real_shells=real_shells,
        reciprocal_shells=reciprocal_shells,
        shell_tolerance=float(shell_tolerance),
        max_shells=int(max_shells),
        exclude_zero_shift=bool(exclude_zero_shift),
        dtype=np.complex128,
        workspace=workspace,
    )
    return block_from_structural_sums(
        lmax=int(lmax),
        structural_sums=sums,
        ab5=ab5,
        dtype=dtype,
    )


def _add_self_correction_to_batched_sums(
    sums: Array,
    *,
    local_destination_index: int,
    k: float,
    eta: float,
) -> None:
    """Patch the same-particle central-point correction into one batched table."""
    idx = int(local_destination_index)
    if idx < 0 or idx >= sums.shape[0]:
        return
    order = int(sums.shape[1] - 1)
    sums[idx, 0, order] += structural_sum_m_normalization(0) * _self_correction(
        float(k), float(eta)
    )


def periodic_ewald_blocks_for_source(
    *,
    source_index: int,
    lmax: int,
    k: float,
    positions: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    ab5: Array,
    dtype: npt.DTypeLike,
    shell_tolerance: float,
    max_shells: int,
    workspace: EwaldShellWorkspace,
    contraction_tensor: Array | None = None,
    destination_indices: Sequence[int] | None = None,
) -> Array:
    """Return selected destination blocks for one source in one structural batch."""
    src_idx = int(source_index)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    destinations = (
        np.arange(pos.shape[0], dtype=np.int64)
        if destination_indices is None
        else np.asarray(destination_indices, dtype=np.int64).reshape(-1)
    )
    sums = ewald_structural_sums_2d_batch(
        lmax_struct=int(lmax),
        k=float(k),
        destinations=pos[destinations],
        source=pos[src_idx],
        lattice=lattice,
        k_parallel=k_parallel,
        eta=float(eta),
        real_shells=real_shells,
        reciprocal_shells=reciprocal_shells,
        shell_tolerance=float(shell_tolerance),
        max_shells=int(max_shells),
        dtype=np.complex128,
        workspace=workspace,
    )
    self_rows = np.flatnonzero(destinations == src_idx)
    if self_rows.size:
        _add_self_correction_to_batched_sums(
            sums,
            local_destination_index=int(self_rows[0]),
            k=float(k),
            eta=float(eta),
        )
    return blocks_from_structural_sums(
        lmax=int(lmax),
        structural_sums=sums,
        ab5=ab5,
        dtype=dtype,
        contraction_tensor=contraction_tensor,
    )


def _fill_block_cache_for_source(
    *,
    source_index: int,
    cache: dict[tuple[int, int], Array],
    lmax: int,
    k: float,
    positions: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    ab5: Array,
    dtype: npt.DTypeLike,
    shell_tolerance: float,
    max_shells: int,
    workspace: EwaldShellWorkspace,
    contraction_tensor: Array | None = None,
) -> None:
    """Populate missing dense blocks for one source using one structural batch."""
    src_idx = int(source_index)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    missing = [i for i in range(pos.shape[0]) if (i, src_idx) not in cache]
    if not missing:
        return
    blocks = periodic_ewald_blocks_for_source(
        source_index=src_idx,
        lmax=int(lmax),
        k=float(k),
        positions=pos,
        lattice=lattice,
        k_parallel=k_parallel,
        eta=float(eta),
        real_shells=real_shells,
        reciprocal_shells=reciprocal_shells,
        ab5=ab5,
        dtype=dtype,
        shell_tolerance=float(shell_tolerance),
        max_shells=int(max_shells),
        workspace=workspace,
        contraction_tensor=contraction_tensor,
        destination_indices=missing,
    )
    for local_idx, dst_idx in enumerate(missing):
        cache[(int(dst_idx), src_idx)] = blocks[int(local_idx)]


def fill_periodic_ewald_block_cache(
    *,
    cache: dict[tuple[int, int], Array],
    lmax: int,
    k: float,
    positions: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    source_indices: Iterable[int] | None = None,
    workspace: EwaldShellWorkspace | None = None,
    contraction_tensor: Array | None = None,
) -> None:
    """Populate a dense periodic block cache source-by-source."""
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )
    sources = range(pos.shape[0]) if source_indices is None else source_indices
    for j in sources:
        _fill_block_cache_for_source(
            source_index=int(j),
            cache=cache,
            lmax=int(lmax),
            k=float(k),
            positions=pos,
            lattice=lattice,
            k_parallel=k_parallel,
            eta=float(eta),
            real_shells=real_shells,
            reciprocal_shells=reciprocal_shells,
            ab5=ab5,
            dtype=dtype,
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
            workspace=ws,
            contraction_tensor=contraction_tensor,
        )


def apply_periodic_ewald_sum(
    *,
    lmax: int,
    k: float,
    positions: Array,
    x: Array,
    lattice: RectangularLattice2D,
    k_parallel: Array,
    eta: float,
    real_shells: int | None,
    reciprocal_shells: int | None,
    ab5: Array,
    dtype: npt.DTypeLike = np.complex128,
    shell_tolerance: float = 1.0e-10,
    max_shells: int = 32,
    block_cache: dict[tuple[int, int], Array] | None = None,
    workspace: EwaldShellWorkspace | None = None,
    contraction_tensor: Array | None = None,
) -> Array:
    """Apply the Ewald Bloch image sum to stacked SVWF coefficients.

    The no-cache path batches all destinations for one source particle at a
    time. This keeps iterative CPU runs matrix-free without recomputing one
    complete scalar Ewald table independently for every pair block.
    """
    out_dtype = np.dtype(dtype)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    ns = pos.shape[0]
    nm = n_modes(int(lmax))
    arr = np.asarray(x, dtype=out_dtype).reshape(ns, nm)
    y = np.zeros_like(arr, dtype=out_dtype)
    ws = _ensure_workspace(
        workspace=workspace,
        k=float(k),
        k_parallel=k_parallel,
        lattice=lattice,
        eta=float(eta),
    )

    for j in range(ns):
        if block_cache is None:
            sums = ewald_structural_sums_2d_batch(
                lmax_struct=int(lmax),
                k=float(k),
                destinations=pos,
                source=pos[j],
                lattice=lattice,
                k_parallel=k_parallel,
                eta=float(eta),
                real_shells=real_shells,
                reciprocal_shells=reciprocal_shells,
                shell_tolerance=float(shell_tolerance),
                max_shells=int(max_shells),
                dtype=np.complex128,
                workspace=ws,
            )
            _add_self_correction_to_batched_sums(
                sums,
                local_destination_index=j,
                k=float(k),
                eta=float(eta),
            )
            y += apply_structural_sums_to_vector(
                lmax=int(lmax),
                structural_sums=sums,
                ab5=ab5,
                vector=arr[j],
                dtype=out_dtype,
                contraction_tensor=contraction_tensor,
            )
            continue

        _fill_block_cache_for_source(
            source_index=j,
            cache=block_cache,
            lmax=int(lmax),
            k=float(k),
            positions=pos,
            lattice=lattice,
            k_parallel=k_parallel,
            eta=float(eta),
            real_shells=real_shells,
            reciprocal_shells=reciprocal_shells,
            ab5=ab5,
            dtype=out_dtype,
            shell_tolerance=float(shell_tolerance),
            max_shells=int(max_shells),
            workspace=ws,
            contraction_tensor=contraction_tensor,
        )
        for i in range(ns):
            y[i] += block_cache[(i, j)] @ arr[j]
    return y.reshape(ns * nm)


__all__ = [
    "EwaldShellCounts",
    "EwaldShellWorkspace",
    "PeriodicEwaldConvergenceError",
    "apply_periodic_ewald_sum",
    "default_ewald_eta",
    "ewald_self_correction",
    "ewald_structural_constant_2d",
    "ewald_structural_sums_2d",
    "ewald_structural_sums_2d_batch",
    "fill_periodic_ewald_block_cache",
    "periodic_ewald_block",
    "resolve_ewald_eta",
    "resolve_ewald_shell_counts",
    "select_ewald_eta",
]
