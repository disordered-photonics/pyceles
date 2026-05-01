"""Diffraction-order enumeration helpers for periodic far-field workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyceles.core.lattice import RectangularLattice2D


@dataclass(frozen=True)
class DiffractionOrders:
    """Rectangular-lattice diffraction-order table."""

    m: np.ndarray
    n: np.ndarray
    k_parallel: np.ndarray
    kz: np.ndarray
    propagating: np.ndarray


def enumerate_diffraction_orders_rectangular(
    *,
    lattice: RectangularLattice2D,
    k_parallel_incident: np.ndarray,
    k: float,
    output_bmax: float | None = None,
    threshold: float = 1e-12,
) -> DiffractionOrders:
    """Enumerate diffraction orders in an explicit reciprocal-radius domain.

    When `output_bmax` is set, all orders satisfying
    `|k_parallel_incident + m*b1 + n*b2| <= output_bmax` are included.

    When `output_bmax` is `None`, the returned table is propagating-only
    (`|k_parallel_mn| <= k`), which is sufficient for `R/T/A`.
    """
    k_f = float(k)
    if not np.isfinite(k_f) or k_f <= 0.0:
        raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
    kp0 = np.asarray(k_parallel_incident, dtype=float).reshape(2)
    tol = float(threshold)
    bmax = float(k_f if output_bmax is None else output_bmax)
    if not np.isfinite(bmax) or bmax <= 0.0:
        raise ValueError(
            f"`output_bmax` must be finite and positive when set. Got {output_bmax!r}."
        )

    b1 = np.asarray(lattice.b1, dtype=float).reshape(2)
    b2 = np.asarray(lattice.b2, dtype=float).reshape(2)
    b1_norm = float(np.linalg.norm(b1))
    b2_norm = float(np.linalg.norm(b2))
    if b1_norm <= 0.0 or b2_norm <= 0.0:
        raise ValueError("Reciprocal lattice vectors must be non-zero.")

    # Rectangular lattice: b1 is x-aligned and b2 is y-aligned.
    m_min = int(np.floor((-bmax - float(kp0[0])) / b1_norm))
    m_max = int(np.ceil((bmax - float(kp0[0])) / b1_norm))
    n_min = int(np.floor((-bmax - float(kp0[1])) / b2_norm))
    n_max = int(np.ceil((bmax - float(kp0[1])) / b2_norm))

    radius_sq_max = float((bmax + tol) * (bmax + tol))
    prop_sq_max = float((k_f + tol) * (k_f + tol))
    rows: list[tuple[int, int, np.ndarray, complex, bool]] = []
    for m in range(m_min, m_max + 1):
        for n in range(n_min, n_max + 1):
            kp = kp0 + lattice.reciprocal_vector(m, n)
            kp_sq = float(np.dot(kp, kp))
            if kp_sq > radius_sq_max:
                continue
            if output_bmax is None and kp_sq > prop_sq_max:
                continue
            kz2 = k_f * k_f - kp_sq
            if kz2 >= -tol:
                kz = complex(float(np.sqrt(max(kz2, 0.0))), 0.0)
                propagating = True
            else:
                kz = complex(0.0, float(np.sqrt(-kz2)))
                propagating = False
            rows.append((int(m), int(n), kp, kz, bool(propagating)))

    if not rows:
        return DiffractionOrders(
            m=np.zeros((0,), dtype=np.int32),
            n=np.zeros((0,), dtype=np.int32),
            k_parallel=np.zeros((0, 2), dtype=float),
            kz=np.zeros((0,), dtype=np.complex128),
            propagating=np.zeros((0,), dtype=bool),
        )

    m_arr = np.asarray([row[0] for row in rows], dtype=np.int32)
    n_arr = np.asarray([row[1] for row in rows], dtype=np.int32)
    kp_arr = np.asarray([row[2] for row in rows], dtype=float).reshape(-1, 2)
    kz_arr = np.asarray([row[3] for row in rows], dtype=np.complex128).reshape(-1)
    prop_arr = np.asarray([row[4] for row in rows], dtype=bool).reshape(-1)
    return DiffractionOrders(
        m=m_arr,
        n=n_arr,
        k_parallel=kp_arr,
        kz=kz_arr,
        propagating=prop_arr,
    )


__all__ = ["DiffractionOrders", "enumerate_diffraction_orders_rectangular"]
