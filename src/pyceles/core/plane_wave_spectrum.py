"""Typed TE/TM plane-wave spectra on one shared angular grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(eq=False, frozen=True, slots=True, repr=False)
class PlaneWaveSpectrum:
    """TE/TM plane-wave coefficients on one common wavevector grid.

    Plane-wave polarization components are one physical spectrum, not two
    independently gridded payloads. Keeping them together makes their shared
    angular and wavevector coordinates an explicit invariant.

    Arrays remain mutable by design, matching other numerical result payloads
    in pyceles. Constructing a spectrum normalizes shapes but does not copy
    already compatible arrays.
    """

    alpha: np.ndarray
    beta: np.ndarray
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    coeff_te: np.ndarray
    coeff_tm: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shared ``(n_alpha, n_beta)`` grid shape."""
        return int(self.coeff_te.shape[0]), int(self.coeff_te.shape[1])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape}, coeff_dtype={self.coeff_te.dtype!r})"

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=float).reshape(-1)
        beta = np.asarray(self.beta, dtype=float).reshape(-1)
        shape = (alpha.size, beta.size)
        grids = {
            "kx": np.asarray(self.kx, dtype=float),
            "ky": np.asarray(self.ky, dtype=float),
            "kz": np.asarray(self.kz, dtype=float),
        }
        coefficients = {
            "coeff_te": np.asarray(self.coeff_te),
            "coeff_tm": np.asarray(self.coeff_tm),
        }
        for name, value in (*grids.items(), *coefficients.items()):
            if value.shape != shape:
                raise ValueError(f"`{name}` must have shape {shape}. Got {value.shape}.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        for name, value in (*grids.items(), *coefficients.items()):
            object.__setattr__(self, name, value)

    @classmethod
    def empty(cls, dtype: npt.DTypeLike = np.complex128) -> PlaneWaveSpectrum:
        """Return an empty spectrum for disabled finite far-field output."""
        return cls(
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0, 0), dtype=float),
            np.zeros((0, 0), dtype=float),
            np.zeros((0, 0), dtype=float),
            np.zeros((0, 0), dtype=np.dtype(dtype)),
            np.zeros((0, 0), dtype=np.dtype(dtype)),
        )

    def with_coefficients(
        self,
        coeff_te: np.ndarray,
        coeff_tm: np.ndarray,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> PlaneWaveSpectrum:
        """Return this grid with replacement TE/TM coefficients."""
        if dtype is not None:
            coeff_te = np.asarray(coeff_te, dtype=np.dtype(dtype))
            coeff_tm = np.asarray(coeff_tm, dtype=np.dtype(dtype))
        return type(self)(
            self.alpha,
            self.beta,
            self.kx,
            self.ky,
            self.kz,
            coeff_te,
            coeff_tm,
        )

    def astype(self, dtype: npt.DTypeLike) -> PlaneWaveSpectrum:
        """Return this spectrum with both coefficient arrays cast to ``dtype``."""
        ctype = np.dtype(dtype)
        return self.with_coefficients(
            self.coeff_te.astype(ctype, copy=False),
            self.coeff_tm.astype(ctype, copy=False),
        )

    def linear_combination(
        self,
        other: PlaneWaveSpectrum,
        *,
        weight_self: complex,
        weight_other: complex,
        dtype: npt.DTypeLike,
    ) -> PlaneWaveSpectrum:
        """Coherently combine two spectra defined on the same grid."""
        self.require_same_grid(other)
        ctype = np.dtype(dtype)
        return self.with_coefficients(
            np.asarray(
                weight_self * self.coeff_te + weight_other * other.coeff_te,
                dtype=ctype,
            ),
            np.asarray(
                weight_self * self.coeff_tm + weight_other * other.coeff_tm,
                dtype=ctype,
            ),
        )

    def require_same_grid(self, other: PlaneWaveSpectrum) -> None:
        """Raise when ``other`` does not use exactly the same coordinates."""
        for name in ("alpha", "beta", "kx", "ky", "kz"):
            left = getattr(self, name)
            right = getattr(other, name)
            if left is not right and not np.array_equal(left, right):
                raise ValueError(
                    f"Cannot combine plane-wave spectra with different `{name}` grids."
                )


__all__ = ["PlaneWaveSpectrum"]
