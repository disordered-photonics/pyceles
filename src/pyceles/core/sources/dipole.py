from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from pyceles._arrays import owned_read_only_view

from ..radial import passive_wavenumber
from .common import (
    _as_complex_triplet,
    _as_float_triplet,
    _complex_triplet_tuple,
    _dipole_outgoing_coeffs,
    _float_triplet_tuple,
    _incident_coeffs_from_outgoing_expansion,
    _normalize_dipole_collection_inputs,
)


@dataclass(frozen=True)
class DipoleSource:
    """Electric point dipole with a specified moment at a finite position.

    In a passive host, E = k0**2 * (I + grad grad/k**2) * exp(i*k*r)/(4*pi*r) * p,
    with k = k0*medium_n and r measured from the emitter. Finite exterior E and
    Z0*H are defined away from the emitter; self-power in an absorbing host
    requires a microscopic emitter model.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    dipole_moment: tuple[complex, complex, complex] = (1.0 + 0j, 0.0 + 0j, 0.0 + 0j)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0
    radial_lut_dr: float = 0.0

    def __post_init__(self) -> None:
        passive_wavenumber(self.medium_n)
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive")
        object.__setattr__(
            self, "dipole_moment", _complex_triplet_tuple("dipole_moment", self.dipole_moment)
        )
        object.__setattr__(self, "position", _float_triplet_tuple("position", self.position))
        if not np.isfinite(self.amplitude):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if not np.isfinite(self.radial_lut_dr) or self.radial_lut_dr < 0.0:
            raise ValueError(f"`radial_lut_dr` must be >= 0. Got {self.radial_lut_dr!r}.")

    def has_finite_incident_power(self) -> bool:
        """Beam-power diagnostics do not apply to local dipole emitters."""
        return False

    @property
    def outgoing_lmax(self) -> int:
        """An electric point dipole emits only degree-one SVWFs."""
        return 1

    def source_positions(self) -> np.ndarray:
        """Return emitter coordinates for local outgoing-expansion workflows."""
        return self.dipole_positions()

    def dipole_positions(self) -> np.ndarray:
        return _as_float_triplet("position", self.position).reshape(1, 3)

    def dipole_moments(self) -> np.ndarray:
        mu = _as_complex_triplet("dipole_moment", self.dipole_moment)
        return (complex(self.amplitude) * mu).reshape(1, 3)

    def angular_frequency(self) -> float:
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background(self) -> float:
        if complex(self.medium_n).imag != 0:
            raise ValueError("Point-dipole self-power requires a nonabsorbing host.")
        k0 = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * k0
        mu = self.dipole_moments().reshape(3)
        mu2 = float(np.sum(np.abs(mu) ** 2))
        return float(mu2 * k * (k0**3) / (12.0 * np.pi))

    def cartesian_basis_sources(
        self,
        *,
        moment_magnitude: complex = 1.0 + 0j,
    ) -> dict[str, DipoleSource]:
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
        *,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        k0 = self.angular_frequency()
        return _dipole_outgoing_coeffs(
            k0=k0,
            k_medium=k0 * self.medium_n,
            dipole_moments=self.dipole_moments(),
            dtype=np.dtype(dtype),
        )

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
        return _incident_coeffs_from_outgoing_expansion(
            receiver_positions=pos_rcv,
            source_positions=self.source_positions(),
            outgoing_coeffs=self.outgoing_coeffs(dtype=ctype),
            outgoing_lmax=self.outgoing_lmax,
            lmax=int(lmax),
            k_medium=self.angular_frequency() * self.medium_n,
            radial_lut_dr=float(self.radial_lut_dr),
            dtype=ctype,
        )


@dataclass(frozen=True)
class DipoleCollection:
    """Coherent electric point dipoles with finite exterior fields in a passive host.

    Moments use the normalization and self-power policy of DipoleSource.
    """

    wavelength: float
    medium_n: complex = 1.0 + 0j
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    dipole_moments: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.complex128)
    )
    amplitude: complex = 1.0
    radial_lut_dr: float = 0.0

    def __post_init__(self) -> None:
        passive_wavenumber(self.medium_n)
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive")
        positions, moments = _normalize_dipole_collection_inputs(
            self.positions, self.dipole_moments
        )
        positions = owned_read_only_view(positions, dtype=np.dtype(float))
        moments = owned_read_only_view(moments, dtype=np.dtype(np.complex128))
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "dipole_moments", moments)
        if not np.isfinite(self.amplitude):
            raise ValueError(f"`amplitude` must be finite. Got {self.amplitude!r}.")
        if not np.isfinite(self.radial_lut_dr) or self.radial_lut_dr < 0.0:
            raise ValueError(f"`radial_lut_dr` must be >= 0. Got {self.radial_lut_dr!r}.")

    def has_finite_incident_power(self) -> bool:
        """Beam-power diagnostics do not apply to local dipole emitters."""
        return False

    @property
    def outgoing_lmax(self) -> int:
        """An electric point dipole emits only degree-one SVWFs."""
        return 1

    def source_positions(self) -> np.ndarray:
        """Return emitter coordinates for local outgoing-expansion workflows."""
        return self.dipole_positions()

    def dipole_positions(self) -> np.ndarray:
        pos, _ = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return pos

    def dipole_moments_array(self) -> np.ndarray:
        _, mom = _normalize_dipole_collection_inputs(self.positions, self.dipole_moments)
        return complex(self.amplitude) * mom

    def angular_frequency(self) -> float:
        return float(2.0 * np.pi / float(self.wavelength))

    def dissipated_power_homogeneous_background_per_dipole(self) -> np.ndarray:
        if complex(self.medium_n).imag != 0:
            raise ValueError("Point-dipole self-power requires a nonabsorbing host.")
        k0 = self.angular_frequency()
        k = float(np.real(complex(self.medium_n))) * k0
        mu = self.dipole_moments_array()
        mu2 = np.sum(np.abs(mu) ** 2, axis=1)
        return np.asarray(mu2 * k * (k0**3) / (12.0 * np.pi), dtype=float)

    def dissipated_power_homogeneous_background(self) -> float:
        return float(np.sum(self.dissipated_power_homogeneous_background_per_dipole()))

    def outgoing_coeffs(
        self,
        *,
        dtype: npt.DTypeLike = np.complex128,
    ) -> np.ndarray:
        k0 = self.angular_frequency()
        return _dipole_outgoing_coeffs(
            k0=k0,
            k_medium=k0 * self.medium_n,
            dipole_moments=self.dipole_moments_array(),
            dtype=np.dtype(dtype),
        )

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
        return _incident_coeffs_from_outgoing_expansion(
            receiver_positions=pos_rcv,
            source_positions=self.source_positions(),
            outgoing_coeffs=self.outgoing_coeffs(dtype=ctype),
            outgoing_lmax=self.outgoing_lmax,
            lmax=int(lmax),
            k_medium=self.angular_frequency() * self.medium_n,
            radial_lut_dr=float(self.radial_lut_dr),
            dtype=ctype,
        )
