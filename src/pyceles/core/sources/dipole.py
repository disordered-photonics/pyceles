from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from ..indexing import n_modes
from .common import (
    _as_complex_triplet,
    _as_float_triplet,
    _dipole_outgoing_coeff_vector,
    _incident_coeffs_from_outgoing_expansion,
    _normalize_dipole_collection_inputs,
)


@dataclass(frozen=True)
class DipoleSource:
    """Single electric point dipole in a homogeneous medium."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    dipole_moment: tuple[complex, complex, complex] = (1.0 + 0j, 0.0 + 0j, 0.0 + 0j)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = 1.0
    radial_lut_dr: float = 0.0

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if not (n.real > 0):
            raise ValueError(f"medium_n must have positive real part. Got {n!r}")
        if abs(n.imag) > 0:
            raise ValueError(
                "DipoleSource currently requires real `medium_n` in this homogeneous solver path "
                "(legacy policy from the original beam-only workflow, not a dipole-physics limit)."
            )
        _as_complex_triplet("dipole_moment", self.dipole_moment)
        _as_float_triplet("position", self.position)
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if float(self.radial_lut_dr) < 0.0:
            raise ValueError(f"`radial_lut_dr` must be >= 0. Got {self.radial_lut_dr!r}.")

    def has_finite_incident_power(self) -> bool:
        """Beam-power diagnostics do not apply to local dipole emitters."""
        return False

    def dipole_positions(self) -> np.ndarray:
        return _as_float_triplet("position", self.position).reshape(1, 3)

    def dipole_moments(self) -> np.ndarray:
        mu = _as_complex_triplet("dipole_moment", self.dipole_moment)
        return (complex(self.amplitude) * mu).reshape(1, 3)

    def angular_frequency(self) -> float:
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background(self) -> float:
        k0 = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * k0
        mu = self.dipole_moments().reshape(3)
        mu2 = float(np.sum(np.abs(mu) ** 2))
        return float(mu2 * k * (k0**3) / (12.0 * np.pi))

    def cartesian_basis_sources(
        self,
        *,
        moment_magnitude: complex = 1.0 + 0j,
    ) -> dict[str, "DipoleSource"]:
        m = complex(moment_magnitude)
        if not np.isfinite(m.real) or not np.isfinite(m.imag):
            raise ValueError(f"`moment_magnitude` must be finite. Got {moment_magnitude!r}.")
        return {
            "px": replace(self, dipole_moment=(m, 0.0 + 0j, 0.0 + 0j)),
            "py": replace(self, dipole_moment=(0.0 + 0j, m, 0.0 + 0j)),
            "pz": replace(self, dipole_moment=(0.0 + 0j, 0.0 + 0j, m)),
        }

    def outgoing_coeffs(
        self,
        lmax: int,
        *,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        ctype = np.dtype(dtype)
        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        coeff = _dipole_outgoing_coeff_vector(
            lmax=int(lmax),
            k0=float(k0),
            k_medium=float(k),
            dipole_moment=self.dipole_moments().reshape(3),
            dtype=ctype,
        )
        return coeff.reshape(1, -1)

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        del polar_angles, azimuthal_angles
        ctype = np.dtype(dtype)
        pos_rcv = np.asarray(positions, dtype=float).reshape(-1, 3)
        nm = n_modes(int(lmax))
        if pos_rcv.shape[0] == 0:
            return np.zeros((0, nm), dtype=ctype)
        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        return _incident_coeffs_from_outgoing_expansion(
            receiver_positions=pos_rcv,
            source_positions=self.dipole_positions(),
            outgoing_coeffs=self.outgoing_coeffs(int(lmax), dtype=ctype),
            lmax=int(lmax),
            k_medium=float(k),
            radial_lut_dr=float(self.radial_lut_dr),
            dtype=ctype,
        )


@dataclass(frozen=True)
class DipoleCollection:
    """Collection of electric point dipoles in a homogeneous medium."""

    wavelength: float
    medium_n: complex = 1.0 + 0j
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    dipole_moments: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.complex128)
    )
    amplitude: float = 1.0
    radial_lut_dr: float = 0.0

    def __post_init__(self) -> None:
        n = complex(self.medium_n)
        if not (n.real > 0):
            raise ValueError(f"medium_n must have positive real part. Got {n!r}")
        if abs(n.imag) > 0:
            raise ValueError(
                "DipoleCollection currently requires real `medium_n` in this homogeneous solver path "
                "(legacy policy from the original beam-only workflow, not a dipole-physics limit)."
            )
        _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        if not np.isfinite(float(self.amplitude)):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if float(self.radial_lut_dr) < 0.0:
            raise ValueError(f"`radial_lut_dr` must be >= 0. Got {self.radial_lut_dr!r}.")

    def has_finite_incident_power(self) -> bool:
        """Beam-power diagnostics do not apply to local dipole emitters."""
        return False

    def dipole_positions(self) -> np.ndarray:
        pos, _ = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return pos

    def dipole_moments_array(self) -> np.ndarray:
        _, mom = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return complex(self.amplitude) * mom

    def angular_frequency(self) -> float:
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background_per_dipole(self) -> np.ndarray:
        k0 = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * k0
        mu = self.dipole_moments_array()
        mu2 = np.sum(np.abs(mu) ** 2, axis=1)
        return np.asarray(mu2 * k * (k0**3) / (12.0 * np.pi), dtype=float)

    def dissipated_power_homogeneous_background(self) -> float:
        return float(np.sum(self.dissipated_power_homogeneous_background_per_dipole()))

    def outgoing_coeffs(
        self,
        lmax: int,
        *,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        ctype = np.dtype(dtype)
        pos = self.dipole_positions()
        mom = self.dipole_moments_array()
        nm = n_modes(int(lmax))
        out = np.zeros((pos.shape[0], nm), dtype=ctype)
        if pos.shape[0] == 0:
            return out
        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        for j in range(pos.shape[0]):
            out[j] = _dipole_outgoing_coeff_vector(
                lmax=int(lmax),
                k0=float(k0),
                k_medium=float(k),
                dipole_moment=mom[j],
                dtype=ctype,
            )
        return out

    def incident_coeffs(
        self,
        positions: np.ndarray,
        lmax: int,
        *,
        polar_angles: np.ndarray | None = None,
        azimuthal_angles: np.ndarray | None = None,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        del polar_angles, azimuthal_angles
        ctype = np.dtype(dtype)
        pos_rcv = np.asarray(positions, dtype=float).reshape(-1, 3)
        nm = n_modes(int(lmax))
        if pos_rcv.shape[0] == 0:
            return np.zeros((0, nm), dtype=ctype)
        k0 = 2.0 * np.pi / float(self.wavelength)
        k = k0 * float(np.real(complex(self.medium_n)))
        return _incident_coeffs_from_outgoing_expansion(
            receiver_positions=pos_rcv,
            source_positions=self.dipole_positions(),
            outgoing_coeffs=self.outgoing_coeffs(int(lmax), dtype=ctype),
            lmax=int(lmax),
            k_medium=float(k),
            radial_lut_dr=float(self.radial_lut_dr),
            dtype=ctype,
        )
