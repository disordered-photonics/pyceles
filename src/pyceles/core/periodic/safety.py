"""Rayleigh/Wood-anomaly diagnostics for rectangular periodic cells.

The helpers in this module are deliberately geometry-only. They locate and rank
clearance from Rayleigh/Wood thresholds, where a reciprocal diffraction order is
grazing. They do not evaluate the full periodic coupling operator and therefore
should not be interpreted as a coupling-norm optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import numpy.typing as npt

from pyceles.core.lattice import RectangularLattice2D


@dataclass(frozen=True)
class PeriodicRayleighOrder:
    """One diffraction order in a Rayleigh/Wood safety report.

    The clearance is ``|kz| / k``. It vanishes when the order is exactly
    grazing, where reciprocal-space lattice sums contain singular factors.
    """

    m: int
    n: int
    k_parallel: np.ndarray
    kz: complex
    clearance: float
    propagating: bool


@dataclass(frozen=True)
class PeriodicRayleighReport:
    """Geometry-only warning report for artificial periodicity hazards."""

    period_over_host_wavelength_x: float
    period_over_host_wavelength_y: float
    min_clearance: float
    nearest_orders: tuple[PeriodicRayleighOrder, ...]
    warning_level: Literal["ok", "near", "at"]
    message: str


@dataclass(frozen=True)
class PeriodicRayleighThreshold:
    """One grouped Rayleigh threshold in scaled rectangular-lattice units."""

    scale: float
    orders: tuple[tuple[int, int], ...]
    multiplicity: int


@dataclass(frozen=True)
class SafePeriodScaleCandidate:
    """One Rayleigh-clear period-scale candidate between thresholds.

    The score favors wide threshold gaps and large grazing-order clearance. It is
    a cheap preflight ranking, not a guarantee that the full periodic coupling
    norm is minimal for a particular multipole truncation or particle system.
    """

    scale: float
    left_threshold: float
    right_threshold: float
    gap_width: float
    clearance: float
    nearest_orders: tuple[tuple[int, int], ...]
    score: float


def _validate_k_and_k_parallel(k: float, k_parallel: npt.ArrayLike) -> tuple[float, np.ndarray]:
    k_f = float(k)
    if not np.isfinite(k_f) or k_f <= 0.0:
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    kp = np.asarray(k_parallel, dtype=float).reshape(2)
    if not np.all(np.isfinite(kp)):
        raise ValueError("`k_parallel` must contain finite values.")
    if float(np.linalg.norm(kp)) > k_f * (1.0 + 1.0e-12):
        raise ValueError("`k_parallel` must satisfy |k_parallel| <= k for propagating incidence.")
    return k_f, kp


def _kz_from_order(
    k: float, k_parallel_order: np.ndarray, *, threshold: float
) -> tuple[complex, bool]:
    kz2 = float(k) * float(k) - float(np.dot(k_parallel_order, k_parallel_order))
    tol = float(threshold)
    if kz2 >= -tol:
        return complex(math.sqrt(max(kz2, 0.0)), 0.0), True
    return complex(0.0, math.sqrt(-kz2)), False


def _automatic_order_bounds(
    lattice: RectangularLattice2D,
    *,
    k: float,
    k_parallel: np.ndarray,
    margin: float,
) -> tuple[int, int]:
    bx = abs(float(lattice.b1[0]))
    by = abs(float(lattice.b2[1]))
    if bx <= 0.0 or by <= 0.0:
        raise ValueError("Reciprocal basis vectors must be nonzero.")
    k_lim = float(k) * (1.0 + float(margin))
    m_bound = math.ceil((abs(float(k_parallel[0])) + k_lim) / bx) + 1
    n_bound = math.ceil((abs(float(k_parallel[1])) + k_lim) / by) + 1
    return max(0, m_bound), max(0, n_bound)


def rayleigh_report(
    lattice: RectangularLattice2D,
    *,
    k: float,
    k_parallel: npt.ArrayLike,
    order_bound: int | None = None,
    include_specular: bool = False,
    near_threshold: float = 0.08,
    at_threshold: float = 1.0e-8,
    rayleigh_margin: float = 1.0e-12,
    max_nearest_orders: int = 12,
) -> PeriodicRayleighReport:
    """Return the nearest Rayleigh/Wood anomaly for a periodic cell.

    This is a cheap geometry-only preflight: it enumerates reciprocal-lattice
    orders near the host-medium light circle and reports the minimum
    ``|kz| / k`` clearance. It intentionally does not inspect particle
    material, T-matrices, or the Ewald splitting parameter, because those
    influence the response strength but not the singular locations.
    """

    if not isinstance(lattice, RectangularLattice2D):
        raise TypeError("`lattice` must be a RectangularLattice2D instance.")
    k_f, kp0 = _validate_k_and_k_parallel(k, k_parallel)
    if int(max_nearest_orders) < 1:
        raise ValueError("`max_nearest_orders` must be positive.")
    near = float(near_threshold)
    at = float(at_threshold)
    if not np.isfinite(near) or near <= 0.0:
        raise ValueError("`near_threshold` must be finite and positive.")
    if not np.isfinite(at) or at < 0.0 or at >= near:
        raise ValueError("`at_threshold` must be finite and satisfy 0 <= at < near.")
    if order_bound is None:
        m_bound, n_bound = _automatic_order_bounds(
            lattice,
            k=k_f,
            k_parallel=kp0,
            margin=float(rayleigh_margin),
        )
    else:
        value = int(order_bound)
        if value < 0:
            raise ValueError("`order_bound` must be nonnegative when set.")
        m_bound = n_bound = value

    orders: list[PeriodicRayleighOrder] = []
    threshold = (float(at_threshold) * k_f) ** 2
    for m in range(-m_bound, m_bound + 1):
        for n in range(-n_bound, n_bound + 1):
            if not include_specular and m == 0 and n == 0:
                continue
            kp_order = kp0 + lattice.reciprocal_vector(m, n)
            kz, propagating = _kz_from_order(k_f, kp_order, threshold=threshold)
            clearance = abs(kz) / k_f
            orders.append(
                PeriodicRayleighOrder(
                    m=int(m),
                    n=int(n),
                    k_parallel=np.asarray(kp_order, dtype=float),
                    kz=complex(kz),
                    clearance=float(clearance),
                    propagating=bool(propagating),
                )
            )
    if not orders:
        raise RuntimeError("No diffraction orders were available for Rayleigh safety analysis.")
    orders.sort(key=lambda item: (item.clearance, abs(item.m) + abs(item.n), item.m, item.n))
    min_clearance = float(orders[0].clearance)
    nearest = tuple(
        item
        for item in orders
        if item.clearance <= min_clearance + max(1.0e-12, 1.0e-10 * max(1.0, min_clearance))
    )[: int(max_nearest_orders)]

    sx = float(lattice.ax) * k_f / (2.0 * math.pi)
    sy = float(lattice.ay) * k_f / (2.0 * math.pi)
    if min_clearance <= at:
        level: Literal["ok", "near", "at"] = "at"
        prefix = "Periodic cell is at a Rayleigh/Wood anomaly"
    elif min_clearance < near:
        level = "near"
        prefix = "Periodic cell is near a Rayleigh/Wood anomaly"
    else:
        level = "ok"
        prefix = "Periodic Rayleigh/Wood clearance is acceptable"
    order_text = ", ".join(f"({item.m},{item.n})" for item in nearest[:4])
    message = (
        f"{prefix}: min |kz|/k = {min_clearance:.4g}; "
        f"nearest order(s): {order_text}; "
        f"ax/lambda_host = {sx:.6g}, ay/lambda_host = {sy:.6g}."
    )
    return PeriodicRayleighReport(
        period_over_host_wavelength_x=sx,
        period_over_host_wavelength_y=sy,
        min_clearance=min_clearance,
        nearest_orders=nearest,
        warning_level=level,
        message=message,
    )


def _positive_scaled_threshold(
    *,
    m: int,
    n: int,
    aspect_y_over_x: float,
    u_parallel: tuple[float, float],
) -> float | None:
    aspect = float(aspect_y_over_x)
    if not np.isfinite(aspect) or aspect <= 0.0:
        raise ValueError("`aspect_y_over_x` must be finite and positive.")
    ux, uy = (float(u_parallel[0]), float(u_parallel[1]))
    u2 = ux * ux + uy * uy
    if u2 >= 1.0:
        raise ValueError("`u_parallel` must satisfy |u_parallel| < 1.")
    hx = float(m)
    hy = float(n) / aspect
    h2 = hx * hx + hy * hy
    if h2 == 0.0:
        return None
    udoth = ux * hx + uy * hy
    c = 1.0 - u2
    root = (udoth + math.sqrt(udoth * udoth + c * h2)) / c
    if root <= 0.0 or not math.isfinite(root):
        return None
    return float(root)


def rayleigh_threshold_scales(
    *,
    scale_min: float,
    scale_max: float,
    aspect_y_over_x: float = 1.0,
    u_parallel: tuple[float, float] = (0.0, 0.0),
    grouping_rtol: float = 1.0e-10,
) -> tuple[PeriodicRayleighThreshold, ...]:
    """Enumerate Rayleigh thresholds for scaled rectangular lattices.

    ``scale`` is ``ax / lambda_host``. The y period is tied to x by
    ``ay = aspect_y_over_x * ax``. For square normal incidence, the thresholds
    reduce to ``sqrt(m**2 + n**2)``.
    """

    s0 = float(scale_min)
    s1 = float(scale_max)
    if not (np.isfinite(s0) and np.isfinite(s1) and 0.0 < s0 < s1):
        raise ValueError("Require finite scales with 0 < scale_min < scale_max.")
    aspect = float(aspect_y_over_x)
    ux, uy = (float(u_parallel[0]), float(u_parallel[1]))
    u_norm = math.hypot(ux, uy)
    grouping = float(grouping_rtol)
    if not np.isfinite(grouping) or grouping <= 0.0:
        raise ValueError("`grouping_rtol` must be finite and positive.")
    if u_norm >= 1.0:
        raise ValueError("`u_parallel` must satisfy |u_parallel| < 1.")
    mmax = math.ceil((1.0 + u_norm) * s1) + 2
    nmax = math.ceil((1.0 + u_norm) * aspect * s1) + 2
    raw: list[tuple[float, tuple[int, int]]] = []
    for m in range(-mmax, mmax + 1):
        for n in range(-nmax, nmax + 1):
            if m == 0 and n == 0:
                continue
            scale = _positive_scaled_threshold(
                m=m,
                n=n,
                aspect_y_over_x=aspect,
                u_parallel=(ux, uy),
            )
            if scale is not None and s0 <= scale <= s1:
                raw.append((float(scale), (int(m), int(n))))
    raw.sort(key=lambda item: item[0])

    grouped: list[list[tuple[float, tuple[int, int]]]] = []
    for item in raw:
        if not grouped:
            grouped.append([item])
            continue
        scale = item[0]
        reference = grouped[-1][0][0]
        if abs(scale - reference) <= grouping * max(1.0, abs(reference)):
            grouped[-1].append(item)
        else:
            grouped.append([item])
    out: list[PeriodicRayleighThreshold] = []
    for group in grouped:
        scale = float(sum(item[0] for item in group) / len(group))
        orders = tuple(sorted({item[1] for item in group}))
        out.append(
            PeriodicRayleighThreshold(
                scale=scale,
                orders=orders,
                multiplicity=len(orders),
            )
        )
    return tuple(out)


def _scaled_clearance(
    scale: float,
    *,
    aspect_y_over_x: float,
    u_parallel: tuple[float, float],
) -> tuple[float, tuple[tuple[int, int], ...]]:
    lattice = RectangularLattice2D(
        2.0 * math.pi * float(scale), 2.0 * math.pi * float(scale) * aspect_y_over_x
    )
    report = rayleigh_report(
        lattice,
        k=1.0,
        k_parallel=np.asarray(u_parallel, dtype=float),
        include_specular=False,
        near_threshold=1.0,
    )
    return report.min_clearance, tuple((item.m, item.n) for item in report.nearest_orders)


def suggest_safe_period_scales(
    *,
    scale_min: float,
    scale_max: float,
    aspect_y_over_x: float = 1.0,
    u_parallel: tuple[float, float] = (0.0, 0.0),
    min_gap_width: float = 0.05,
    prefer_smaller_period: float = 0.15,
    max_results: int = 12,
) -> tuple[SafePeriodScaleCandidate, ...]:
    """Rank Rayleigh-clear period ratios between neighboring thresholds.

    The returned scales are good candidates for artificial-period preflights
    because they sit away from grazing diffraction orders. For fine selection
    inside a gap, a coupling-norm scan can still move the optimum away from this
    geometric mid-gap value.
    """

    gap_min = float(min_gap_width)
    small_period_bias = float(prefer_smaller_period)
    n_results = int(max_results)
    if not np.isfinite(gap_min) or gap_min < 0.0:
        raise ValueError("`min_gap_width` must be finite and nonnegative.")
    if not np.isfinite(small_period_bias):
        raise ValueError("`prefer_smaller_period` must be finite.")
    if n_results < 1:
        raise ValueError("`max_results` must be positive.")

    thresholds = rayleigh_threshold_scales(
        scale_min=scale_min,
        scale_max=scale_max,
        aspect_y_over_x=aspect_y_over_x,
        u_parallel=u_parallel,
    )
    edges = (float(scale_min), *(item.scale for item in thresholds), float(scale_max))
    candidates: list[SafePeriodScaleCandidate] = []
    for left, right in pairwise(edges):
        width = float(right - left)
        if width < gap_min:
            continue
        scale = 0.5 * (float(left) + float(right))
        clearance, orders = _scaled_clearance(
            scale,
            aspect_y_over_x=float(aspect_y_over_x),
            u_parallel=u_parallel,
        )
        score = (width * clearance) / (scale**small_period_bias)
        candidates.append(
            SafePeriodScaleCandidate(
                scale=scale,
                left_threshold=float(left),
                right_threshold=float(right),
                gap_width=width,
                clearance=clearance,
                nearest_orders=orders,
                score=float(score),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return tuple(candidates[:n_results])


__all__ = [
    "PeriodicRayleighOrder",
    "PeriodicRayleighReport",
    "PeriodicRayleighThreshold",
    "SafePeriodScaleCandidate",
    "rayleigh_report",
    "rayleigh_threshold_scales",
    "suggest_safe_period_scales",
]
