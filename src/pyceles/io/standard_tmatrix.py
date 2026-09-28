"""Import the published ``.tmat.h5`` spherical T-matrix format.

The common format stores one or more matrices with interleaved ``(l, m,
polarization)`` modes.  Its documented polarization choices are electric /
magnetic parity and positive / negative helicity.  pyceles uses CELES ordering
(M modes first, then N modes), so loading performs the ordering, normalization,
and, when needed, helicity-to-parity conversion at this boundary.  A selected
wavelength is required for spectral files; this module never chooses a
nearest-neighbour matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import h5py
import numpy as np

from pyceles._arrays import owned_read_only_view
from pyceles.core.indexing import n_modes
from pyceles.core.particles import TMatrixParticle

_SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
_SI_PREFIXES = {
    "y": 1e-24,
    "z": 1e-21,
    "a": 1e-18,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "\u00b5": 1e-6,
    "\u03bc": 1e-6,
    "\u00c2\u00b5": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "": 1.0,
    "da": 1e1,
    "h": 1e2,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "Z": 1e21,
    "Y": 1e24,
}


def _text(value: object) -> str:
    """Decode scalar HDF5 strings without imposing a particular string dtype."""

    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _metadata_value(value: object) -> object:
    """Copy an HDF5 scalar/array into ordinary Python-owned metadata."""

    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        return _text(item) if isinstance(item, (bytes, np.bytes_)) else item
    if array.dtype.kind in {"S", "O", "U"}:
        return tuple(_text(item) for item in array.reshape(-1).tolist())
    return owned_read_only_view(array)


def _owned_metadata_value(value: object) -> object:
    """Detach user-supplied metadata from mutable containers."""
    if isinstance(value, np.ndarray):
        return owned_read_only_view(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _owned_metadata_value(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_owned_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_owned_metadata_value(item) for item in value)
    return value


def _owned_metadata_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _owned_metadata_value(value) for key, value in values.items()}
    )


def _scalar_complex_metadata(value: object | None) -> complex | None:
    """Return one finite complex metadata value, or ``None`` otherwise."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        return None
    try:
        scalar = complex(array.reshape(()).item())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        return None
    return scalar


_SPECTRAL_EMBEDDING_SCALARS = frozenset(
    {"relative_permittivity", "relative_permeability", "chirality", "chirality_parameter"}
)


def _embedding_metadata(
    group: h5py.Group, *, count: int, selected_index: int
) -> Mapping[str, object]:
    """Copy embedding metadata, selecting scalar spectral material arrays."""
    values: dict[str, object] = {
        str(key): _metadata_value(value) for key, value in group.attrs.items()
    }
    for key, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        raw = np.asarray(dataset[...])
        if str(key) in _SPECTRAL_EMBEDDING_SCALARS and raw.ndim == 1 and raw.shape[0] == int(count):
            raw = np.asarray(raw[int(selected_index)])
        values[str(key)] = _metadata_value(raw)
    return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class TMatrixData:
    """One wavelength-selected T matrix in pyceles CELES mode ordering."""

    t_matrix: np.ndarray
    lmax: int
    wavelength: float | None = None
    wavelength_unit: str | None = None
    source_axis: str | None = None
    source_basis: str = "parity"
    embedding: Mapping[str, object] = MappingProxyType({})
    attributes: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        lmax = int(self.lmax)
        if lmax < 1:
            raise ValueError("lmax must be >= 1.")
        matrix = np.asarray(self.t_matrix)
        expected = n_modes(lmax)
        if matrix.shape != (expected, expected):
            raise ValueError(
                f"t_matrix must have shape ({expected}, {expected}) for lmax={self.lmax}."
            )
        if not np.issubdtype(matrix.dtype, np.complexfloating):
            matrix = np.asarray(matrix, dtype=np.complex128)
        if not np.all(np.isfinite(matrix.real)) or not np.all(np.isfinite(matrix.imag)):
            raise ValueError("t_matrix must contain only finite values.")
        owned = owned_read_only_view(matrix)
        object.__setattr__(self, "t_matrix", owned)
        object.__setattr__(self, "lmax", lmax)
        basis = str(self.source_basis).strip().lower()
        if basis not in {"parity", "helicity"}:
            raise ValueError(f"source_basis must be 'parity' or 'helicity'. Got {basis!r}.")
        object.__setattr__(self, "source_basis", basis)
        object.__setattr__(self, "embedding", _owned_metadata_mapping(self.embedding))
        object.__setattr__(self, "attributes", _owned_metadata_mapping(self.attributes))

    def as_particle(
        self,
        *,
        position: Sequence[float],
        radius: float,
        wavelength: float,
        n_medium: complex,
    ) -> TMatrixParticle:
        """Wrap this matrix after checking its simulation context."""
        self.validate_context(
            wavelength=float(wavelength),
            n_medium=complex(n_medium),
        )
        values_tuple = tuple(float(value) for value in position)
        if len(values_tuple) != 3 or not np.all(np.isfinite(values_tuple)):
            raise ValueError("position must contain exactly three finite coordinates.")
        values = (values_tuple[0], values_tuple[1], values_tuple[2])
        return TMatrixParticle(
            position=values,
            radius=float(radius),
            lmax=int(self.lmax),
            t_matrix=self.t_matrix,
        )

    @property
    def embedding_refractive_index(self) -> complex | None:
        """Return the scalar host index when the file provides one.

        The standard stores relative permittivity and permeability rather than
        a named refractive index.  A scalar, finite product can be converted
        to ``sqrt(epsilon_r * mu_r)``; tensor-valued or incomplete metadata is
        left as ``None`` so callers do not accidentally infer a host medium.
        """

        epsilon = _scalar_complex_metadata(self.embedding.get("relative_permittivity"))
        permeability = _scalar_complex_metadata(self.embedding.get("relative_permeability"))
        if epsilon is None or permeability is None:
            return None
        value = epsilon * permeability
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            return None
        return complex(np.sqrt(value))

    def validate_context(
        self,
        *,
        wavelength: float | None = None,
        n_medium: complex | None = None,
        rtol: float = 1.0e-10,
        atol: float = 1.0e-12,
    ) -> None:
        """Check available file metadata against a simulation context.

        ``SimulationConfig`` is intentionally unit-agnostic, so callers must
        express ``wavelength`` in the same numerical length convention as the
        geometry and the selected matrix. The file's wavelength-unit metadata
        is informational only: pyceles does not attach a unit to simulation
        coordinates or convert simulation inputs. Missing metadata is not
        treated as a match.
        """

        mismatches: list[str] = []
        if wavelength is not None:
            target = float(wavelength)
            if self.wavelength is None:
                mismatches.append("the file has no wavelength metadata")
            else:
                if not np.isclose(self.wavelength, target, rtol=float(rtol), atol=float(atol)):
                    mismatches.append(
                        f"wavelength {self.wavelength!r} ({self.wavelength_unit or 'unspecified'}) "
                        f"!= requested {target!r}"
                    )
        if n_medium is not None:
            epsilon_r = _scalar_complex_metadata(self.embedding.get("relative_permittivity"))
            mu_r = _scalar_complex_metadata(self.embedding.get("relative_permeability"))
            host_is_supported = True
            if epsilon_r is None:
                mismatches.append("the file has no finite scalar embedding permittivity")
                host_is_supported = False
            if mu_r is None:
                mismatches.append(
                    "the file has no finite scalar embedding permeability; pyceles requires "
                    "an explicitly nonmagnetic host"
                )
                host_is_supported = False
            elif not np.isclose(mu_r, 1.0 + 0.0j, rtol=float(rtol), atol=float(atol)):
                mismatches.append(
                    f"embedding relative permeability {mu_r!r} is unsupported; "
                    "pyceles assumes mu_r=1"
                )
                host_is_supported = False
            for key in ("chirality", "chirality_parameter"):
                if key not in self.embedding:
                    continue
                chirality = _scalar_complex_metadata(self.embedding[key])
                if chirality is None or not np.isclose(
                    chirality, 0.0 + 0.0j, rtol=float(rtol), atol=float(atol)
                ):
                    mismatches.append(
                        f"embedding {key} is nonzero or non-scalar; pyceles assumes "
                        "a nonchiral host"
                    )
                    host_is_supported = False
                    break
            if host_is_supported and epsilon_r is not None and mu_r is not None:
                embedded = complex(np.sqrt(epsilon_r * mu_r))
                if not np.isclose(
                    embedded,
                    complex(n_medium),
                    rtol=float(rtol),
                    atol=float(atol),
                ):
                    mismatches.append(
                        f"embedding index {embedded!r} != requested {complex(n_medium)!r}"
                    )
        if mismatches:
            raise ValueError(
                "T-matrix metadata is incompatible with the requested simulation context: "
                + "; ".join(mismatches)
            )


def _axis_and_values(
    root: h5py.File, count: int
) -> tuple[str | None, np.ndarray | None, str | None]:
    names = (
        "vacuum_wavelength",
        "vacuum_wavenumber",
        "angular_vacuum_wavenumber",
        "frequency",
        "angular_frequency",
    )
    present = [name for name in names if name in root]
    if not present:
        return None, None, None
    if len(present) > 1:
        raise ValueError(
            "A .tmat.h5 file must provide one spectral axis for a T-matrix import. "
            f"Found {present!r}."
        )
    name = present[0]
    raw_values = np.asarray(root[name][...])
    if raw_values.ndim != 1:
        raise ValueError(f"Spectral axis {name!r} must be one-dimensional.")
    values = np.asarray(raw_values, dtype=float)
    if values.size != count:
        raise ValueError(
            f"Spectral axis {name!r} has {values.size} entries but tmatrix has {count}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Spectral axis {name!r} must contain finite values.")
    if name in {
        "vacuum_wavelength",
        "vacuum_wavenumber",
        "angular_vacuum_wavenumber",
    } and np.any(values <= 0.0):
        raise ValueError(f"Spectral axis {name!r} must contain strictly positive values.")
    unit = root[name].attrs.get("unit")
    return name, values, None if unit is None else _text(unit)


def _wavelength_axis(
    axis_name: str | None,
    values: np.ndarray | None,
    unit: str | None,
) -> tuple[np.ndarray | None, str | None]:
    if axis_name is None or values is None:
        return None, None
    if axis_name == "vacuum_wavelength":
        return values, unit
    if axis_name == "vacuum_wavenumber":
        return 1.0 / values, _inverse_length_unit(unit)
    if axis_name == "angular_vacuum_wavenumber":
        return 2.0 * np.pi / values, _inverse_length_unit(unit)
    if axis_name in {"frequency", "angular_frequency"}:
        if unit is None or not str(unit).strip():
            raise ValueError(
                f"Spectral axis {axis_name!r} requires an explicit frequency unit; "
                "pyceles does not assume Hz."
            )
        unit_text = (
            str(unit)
            .replace("\u00c2\u00b5", "u")
            .replace("\u03bc", "u")
            .replace("\u00b5", "u")
            .strip()
        )
        if unit_text.endswith("Hz"):
            prefix = unit_text[:-2]
            inverse_time = False
        elif unit_text.endswith("s^{-1}"):
            prefix = unit_text[: -len("s^{-1}")]
            inverse_time = True
        elif unit_text.endswith("s^-1"):
            prefix = unit_text[: -len("s^-1")]
            inverse_time = True
        else:
            raise ValueError(f"Unsupported frequency unit {unit!r}; use an SI Hz or s^-1 unit.")
        try:
            prefix_scale = _SI_PREFIXES[prefix]
        except KeyError as exc:
            raise ValueError(f"Unsupported frequency unit {unit!r}.") from exc
        scale = 1.0 / prefix_scale if inverse_time else prefix_scale
        frequency_hz = values * scale
        if np.any(frequency_hz <= 0.0):
            raise ValueError("Frequency values must be strictly positive.")
        factor = 2.0 * np.pi if axis_name == "angular_frequency" else 1.0
        return factor * _SPEED_OF_LIGHT_M_PER_S / frequency_hz, "m"
    return None, None


def _inverse_length_unit(unit: str | None) -> str | None:
    """Return the length unit represented by a reciprocal-length label."""

    text = "" if unit is None else str(unit).strip().replace(" ", "")
    if text.startswith("1/"):
        text = text[2:]
    text = text.replace("^{-1}", "").replace("^-1", "")
    return text or None


def _select_index(
    *,
    count: int,
    wavelengths: np.ndarray | None,
    wavelength_index: int | None,
    wavelength: float | None,
) -> int:
    if wavelength_index is not None and wavelength is not None:
        raise ValueError("Pass only one of wavelength_index or wavelength.")
    if wavelength_index is not None:
        index = int(wavelength_index)
        if index != wavelength_index or index < 0 or index >= count:
            raise IndexError(f"wavelength_index must be in [0, {count}). Got {wavelength_index!r}.")
        return index
    if wavelength is not None:
        if wavelengths is None:
            raise ValueError(
                "This file has no wavelength-resolvable spectral axis; use wavelength_index."
            )
        target = float(wavelength)
        if not np.isfinite(target):
            raise ValueError("wavelength must be finite.")
        scale = max(1.0, abs(target))
        matches = np.flatnonzero(np.isclose(wavelengths, target, rtol=1e-10, atol=1e-12 * scale))
        if matches.size == 0:
            raise ValueError(
                f"No exact wavelength match for {target!r}; available range is "
                f"[{float(np.min(wavelengths))!r}, {float(np.max(wavelengths))!r}]. "
                "The importer never selects a nearest neighbour."
            )
        if matches.size != 1:
            raise ValueError(
                f"Wavelength {target!r} matches {matches.size} spectral entries within "
                "the import tolerance; select the intended matrix with wavelength_index."
            )
        return int(matches[0])
    if count != 1:
        raise ValueError(
            f"The file contains {count} wavelengths; choose wavelength_index or wavelength."
        )
    return 0


def _normalise_polarization(value: object) -> str:
    text = _text(value).strip().lower()
    aliases = {
        "electric": "electric",
        "n": "electric",
        "tm": "electric",
        "magnetic": "magnetic",
        "m": "magnetic",
        "te": "magnetic",
    }
    if text in {"positive", "plus", "+", "helicity+", "helicity_positive"}:
        return "positive"
    if text in {"negative", "minus", "-", "helicity-", "helicity_negative"}:
        return "negative"
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(
            "modes/polarization must use electric/magnetic parity or positive/negative "
            f"helicity labels; got {value!r}."
        ) from exc


def _polarization_basis(labels: Sequence[str]) -> str:
    kinds = {str(label) for label in labels}
    parity = {"electric", "magnetic"}
    helicity = {"positive", "negative"}
    if kinds == parity:
        return "parity"
    if kinds == helicity:
        return "helicity"
    raise ValueError(
        "modes/polarization must contain exactly electric/magnetic parity or "
        f"positive/negative helicity labels; got {sorted(kinds)!r}."
    )


def _validate_single_origin_modes(modes: h5py.Group) -> None:
    """Reject bases with displaced or multiple spherical-wave centers."""
    if "positions" in modes:
        positions = np.asarray(modes["positions"][...], dtype=float)
        if positions.shape != (1, 3) or not np.all(np.isfinite(positions)):
            raise NotImplementedError(
                "TMatrixParticle supports exactly one finite spherical-wave expansion center."
            )
        if np.any(positions != 0.0):
            raise NotImplementedError(
                "Displaced modes/positions are not supported: imported modes must be centered at the origin."
            )
    for name in (
        "index",
        "position_index",
        "positions_index",
        "position_index_incident",
        "position_index_scattered",
        "positions_index_incident",
        "positions_index_scattered",
    ):
        if name not in modes:
            continue
        indices = _integer_labels(np.asarray(modes[name][...]), name)
        if np.any(indices != 0):
            raise NotImplementedError(
                "Multi-center spherical-wave mode indices are not supported by TMatrixParticle."
            )


def _integer_labels(values: np.ndarray, name: str) -> np.ndarray:
    """Return integer mode labels, accepting legacy integer-valued floats."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"modes/{name} must be one-dimensional.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"modes/{name} must contain integer values.")
    if np.issubdtype(array.dtype, np.integer):
        return np.asarray(array, dtype=np.int64)
    if not np.issubdtype(array.dtype, np.floating) or not np.all(np.isfinite(array)):
        raise TypeError(f"modes/{name} must contain integer values.")
    rounded = np.rint(array)
    if not np.all(array == rounded):
        raise TypeError(f"modes/{name} must contain integer values.")
    return np.asarray(rounded, dtype=np.int64)


def _convert_storage_matrix(
    stored: np.ndarray,
    *,
    lmax: int,
    mode_labels: Sequence[tuple[int, int, str]],
) -> tuple[np.ndarray, str]:
    labels = tuple(mode_labels)
    lookup: dict[tuple[int, int, str], int] = {}
    for index, label in enumerate(labels):
        if label in lookup:
            raise ValueError(f"Duplicate mode label {label!r}.")
        lookup[label] = index
    basis = _polarization_basis([label[2] for label in labels])
    polarizations = ("electric", "magnetic") if basis == "parity" else ("positive", "negative")
    expected = tuple(
        (l, m, pol)
        for l in range(1, int(lmax) + 1)
        for m in range(-l, l + 1)
        for pol in polarizations
    )
    if set(labels) != set(expected) or len(labels) != len(expected):
        raise ValueError(
            "modes/l, modes/m, and modes/polarization must describe every "
            f"(l,m,{basis} polarization) mode exactly once."
        )

    # The published storage normalization differs from the CELES spherical
    # harmonics by c(m) = i/sqrt(pi) for m<=0 and its signed partner for m>0.
    # Undo that diagonal similarity in the source basis, then convert helicity
    # to parity if required.  The latter is the real orthogonal block transform
    # positive=(electric+magnetic)/sqrt(2), negative=(electric-magnetic)/sqrt(2).
    native_labels = tuple(
        (l, m, pol)
        for pol in (("magnetic", "electric") if basis == "parity" else polarizations)
        for l in range(1, int(lmax) + 1)
        for m in range(-l, l + 1)
    )
    stored_indices = np.asarray([lookup[label] for label in native_labels], dtype=np.int64)
    stored_array = np.asarray(stored)
    out_dtype = (
        stored_array.dtype
        if stored_array.dtype in {np.dtype(np.complex64), np.dtype(np.complex128)}
        else np.dtype(np.complex128)
    )
    native_m = np.asarray([label[1] for label in native_labels], dtype=float)
    signs = np.where(native_m > 0.0, (-1.0) ** native_m, 1.0).astype(out_dtype)
    phase = np.asarray(1j / np.sqrt(np.pi), dtype=out_dtype) * signs
    selected = np.asarray(stored_array, dtype=out_dtype)[np.ix_(stored_indices, stored_indices)]
    native_source = phase[:, None] * selected / phase[None, :]
    if basis == "parity":
        return np.asarray(native_source, dtype=out_dtype), basis

    # Rows/columns of native_source are ordered (+, -) by (l, m) blocks.  The
    # CELES parity order is all magnetic modes followed by all electric modes.
    source_lookup = {label: index for index, label in enumerate(native_labels)}
    parity_labels = tuple(
        (l, m, pol)
        for pol in ("magnetic", "electric")
        for l in range(1, int(lmax) + 1)
        for m in range(-l, l + 1)
    )
    change = np.zeros((len(parity_labels), len(native_labels)), dtype=out_dtype)
    inv_sqrt2 = np.asarray(1.0 / np.sqrt(2.0), dtype=out_dtype).item()
    for row, (l, m, pol) in enumerate(parity_labels):
        positive = source_lookup[(l, m, "positive")]
        negative = source_lookup[(l, m, "negative")]
        change[row, positive] = inv_sqrt2
        change[row, negative] = inv_sqrt2 if pol == "electric" else -inv_sqrt2
    return np.asarray(change @ native_source @ change.T, dtype=out_dtype), basis


def load_tmatrix_h5(
    path: str | Path,
    *,
    wavelength_index: int | None = None,
    wavelength: float | None = None,
) -> TMatrixData:
    """Load one electric/magnetic T matrix from a standard ``.tmat.h5`` file.

    Parameters
    ----------
    path:
        HDF5 file following the published T-matrix data format.
    wavelength_index, wavelength:
        Select one matrix from a spectral file.  ``wavelength`` is expressed
        in the file's length unit.  Selection is exact within floating-point
        storage tolerance; no nearest-neighbour interpolation is performed.

    Notes
    -----
    Both documented parity and helicity mode labels are accepted.  Helicity
    blocks are converted to the CELES electric/magnetic parity basis during
    import.  Files with different incident/scattered mode sets are rejected
    because ``TMatrixParticle`` currently represents a square particle-local
    operator.
    """

    with h5py.File(str(Path(path)), "r") as root:
        if "tmatrix" not in root:
            raise ValueError("Missing required root dataset 'tmatrix'.")
        raw = root["tmatrix"]
        if raw.ndim == 2:
            count = 1
            matrix_shape = tuple(int(value) for value in raw.shape)
        elif raw.ndim == 3:
            count = int(raw.shape[0])
            matrix_shape = tuple(int(value) for value in raw.shape[1:])
        else:
            raise ValueError(f"tmatrix must be 2D or 3D. Got shape {raw.shape}.")
        if matrix_shape[0] != matrix_shape[1]:
            raise ValueError(f"tmatrix must be square per wavelength. Got {matrix_shape}.")

        modes = root.get("modes")
        if modes is None or any(name not in modes for name in ("l", "m", "polarization")):
            raise ValueError("Missing required modes/l, modes/m, or modes/polarization dataset.")
        if any(
            name in modes
            for name in (
                "l_incident",
                "m_incident",
                "polarization_incident",
                "l_scattered",
                "m_scattered",
                "polarization_scattered",
            )
        ):
            raise NotImplementedError(
                "Separate incident/scattered mode sets are not supported by TMatrixParticle yet."
            )
        _validate_single_origin_modes(modes)
        l_values = _integer_labels(np.asarray(modes["l"][...]), "l")
        m_values = _integer_labels(np.asarray(modes["m"][...]), "m")
        pol_values = np.asarray(modes["polarization"][...])
        if pol_values.ndim != 1:
            raise ValueError("modes/polarization must be a one-dimensional array.")
        if l_values.ndim != 1 or m_values.ndim != 1 or l_values.size != m_values.size:
            raise ValueError("modes/l and modes/m must be aligned one-dimensional arrays.")
        if l_values.size != pol_values.size:
            raise ValueError("modes/l, modes/m, and modes/polarization must have equal length.")
        labels = tuple(
            (int(l), int(m), _normalise_polarization(pol))
            for l, m, pol in zip(l_values, m_values, pol_values, strict=True)
        )
        lmax = max((label[0] for label in labels), default=0)
        if lmax < 1 or matrix_shape != (n_modes(lmax), n_modes(lmax)):
            raise ValueError(
                "tmatrix dimensions do not match the declared mode labels: "
                f"shape={matrix_shape}, lmax={lmax}, expected={n_modes(lmax)}."
            )

        axis_name, axis_values, axis_unit = _axis_and_values(root, count)
        wavelength_values, wavelength_unit = _wavelength_axis(axis_name, axis_values, axis_unit)
        selected_index = _select_index(
            count=count,
            wavelengths=wavelength_values,
            wavelength_index=wavelength_index,
            wavelength=wavelength,
        )
        stored = np.asarray(raw[...] if raw.ndim == 2 else raw[selected_index, ...])
        matrix, source_basis = _convert_storage_matrix(stored, lmax=lmax, mode_labels=labels)
        embedding = (
            _embedding_metadata(root["embedding"], count=count, selected_index=selected_index)
            if "embedding" in root
            else MappingProxyType({})
        )
        attrs = MappingProxyType(
            {str(key): _metadata_value(value) for key, value in root.attrs.items()}
        )
        selected_wavelength = (
            None if wavelength_values is None else float(wavelength_values[selected_index])
        )
        return TMatrixData(
            t_matrix=matrix,
            lmax=lmax,
            wavelength=selected_wavelength,
            wavelength_unit=wavelength_unit,
            source_axis=axis_name,
            source_basis=source_basis,
            embedding=embedding,
            attributes=attrs,
        )


__all__ = ["TMatrixData", "load_tmatrix_h5"]
