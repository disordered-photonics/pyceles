"""Lattice geometry helpers for periodic workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def real_transverse_wavevector(value: npt.ArrayLike) -> np.ndarray:
    """Validate a real, finite Bloch vector and return owned xy components."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw) and np.any(raw.imag != 0):
        raise ValueError("k_parallel must be real")
    q = np.asarray(raw.real, dtype=float).reshape(-1)
    if q.size != 2 or not np.all(np.isfinite(q)):
        raise ValueError("k_parallel must contain two finite real components")
    return q.copy()


def canonical_cell_points(
    points: np.ndarray, *, ax: float, ay: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce xy representatives to [0, ax) x [0, ay), returning their shifts."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    shifts = np.zeros_like(points)
    periods = np.asarray([ax, ay], dtype=float)
    shifts[:, :2] = np.floor(points[:, :2] / periods) * periods
    return points - shifts, shifts


def nearest_image_points(
    points: np.ndarray, center: np.ndarray, *, ax: float, ay: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return points beside one rectangular-lattice centre and their cell shifts."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    shifts = np.zeros_like(points)
    periods = np.asarray([ax, ay], dtype=float)
    shifts[:, :2] = np.rint((points[:, :2] - np.asarray(center)[:2]) / periods) * periods
    return points - shifts, shifts


def periodic_image_mask(
    points: np.ndarray, center: np.ndarray, *, ax: float, ay: float
) -> np.ndarray:
    """Identify exact image coordinates without a displacement tolerance.

    Subtracting a large cell shift need not recover a fractional base coordinate
    exactly. Compare with the constructed image instead of that subtraction.
    """
    _, shifts = nearest_image_points(points, center, ax=ax, ay=ay)
    return np.all(np.asarray(points).reshape(-1, 3) == np.asarray(center) + shifts, axis=1)


@dataclass(frozen=True)
class DiffractionOrder2D:
    """One rectangular-lattice diffraction order in a homogeneous medium."""

    m: int
    n: int
    k_parallel: np.ndarray
    kz: complex
    propagating: bool


@dataclass(frozen=True)
class RectangularLattice2D:
    """Rectangular 2D lattice embedded in the `xy` plane."""

    ax: float
    ay: float

    def __post_init__(self) -> None:
        ax = float(self.ax)
        ay = float(self.ay)
        if not np.isfinite(ax) or ax <= 0.0:
            raise ValueError(f"`ax` must be finite and positive. Got {self.ax!r}.")
        if not np.isfinite(ay) or ay <= 0.0:
            raise ValueError(f"`ay` must be finite and positive. Got {self.ay!r}.")

    @property
    def a1(self) -> np.ndarray:
        """First direct-lattice vector."""
        return np.array([float(self.ax), 0.0, 0.0], dtype=float)

    @property
    def a2(self) -> np.ndarray:
        """Second direct-lattice vector."""
        return np.array([0.0, float(self.ay), 0.0], dtype=float)

    @property
    def b1(self) -> np.ndarray:
        """First reciprocal-lattice vector."""
        return np.array([2.0 * np.pi / float(self.ax), 0.0], dtype=float)

    @property
    def b2(self) -> np.ndarray:
        """Second reciprocal-lattice vector."""
        return np.array([0.0, 2.0 * np.pi / float(self.ay)], dtype=float)

    @property
    def area(self) -> float:
        """Unit-cell area."""
        return float(self.ax) * float(self.ay)

    def lattice_vector(self, p: int, q: int) -> np.ndarray:
        """Return direct-lattice shift `p*a1 + q*a2` as a 3D vector."""
        return np.asarray(int(p) * self.a1 + int(q) * self.a2, dtype=float)

    def reciprocal_vector(self, m: int, n: int) -> np.ndarray:
        """Return reciprocal-lattice shift `m*b1 + n*b2` as a 2D vector."""
        return np.asarray(int(m) * self.b1 + int(n) * self.b2, dtype=float)

    def diffraction_orders(
        self,
        *,
        k_parallel: np.ndarray,
        k: float,
        max_order: int,
    ) -> tuple[DiffractionOrder2D, ...]:
        """Enumerate rectangular-lattice diffraction orders.

        Propagating orders use the non-negative real `kz` branch. Evanescent
        orders use the positive-imaginary branch.
        """
        kp0 = np.asarray(k_parallel, dtype=float).reshape(2)
        k_f = float(k)
        if not np.isfinite(k_f) or k_f <= 0.0:
            raise ValueError(f"`k` must be finite and positive. Got {k!r}.")
        order = int(max_order)
        if order < 0:
            raise ValueError(f"`max_order` must be >= 0. Got {max_order!r}.")
        out: list[DiffractionOrder2D] = []
        for m in range(-order, order + 1):
            for n in range(-order, order + 1):
                kp = kp0 + self.reciprocal_vector(m, n)
                kz2 = k_f * k_f - float(np.dot(kp, kp))
                kz = complex(np.sqrt(complex(kz2)))
                propagating = kz.imag == 0.0
                out.append(
                    DiffractionOrder2D(
                        m=int(m),
                        n=int(n),
                        k_parallel=kp,
                        kz=kz,
                        propagating=propagating,
                    )
                )
        return tuple(out)


__all__ = ["DiffractionOrder2D", "RectangularLattice2D"]
