from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


def _pathlike(path: str | Path) -> str:
    """Normalize filesystem path inputs for HDF5 open calls."""
    return str(Path(path))


def _reset_group(handle: h5py.File, group: str) -> h5py.Group:
    """Create a fresh group, replacing any stale payload from previous runs."""
    if group in handle:
        del handle[group]
    return handle.create_group(group)


def _write_attrs(group: h5py.Group, attrs: Mapping[str, Any] | None) -> None:
    """Write metadata attributes with scalar-first HDF5-friendly coercion."""
    if attrs is None:
        return
    for key, value in attrs.items():
        if isinstance(value, (str, bytes, int, float, np.integer, np.floating, np.bool_)):
            group.attrs[key] = value
        else:
            group.attrs[key] = np.asarray(value)


def _write_dataset(group: h5py.Group, name: str, value: Any, *, compression: str | None) -> None:
    """Write one dataset while preserving scalar-vs-array semantics."""
    arr = np.asarray(value)
    if arr.shape == ():
        group.create_dataset(name, data=arr)
    else:
        kwargs = {"compression": compression} if compression is not None else {}
        group.create_dataset(name, data=arr, **kwargs)


def save_geometry_h5(
    path: str | Path,
    *,
    positions: np.ndarray,
    radii: np.ndarray,
    n_particle: np.ndarray | complex,
    n_medium: complex,
    wavelength: float,
    lmax: int,
    group: str = "geometry",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Persist particle geometry and optical constants for reproducible reruns."""
    with h5py.File(_pathlike(path), mode) as h5:
        g = _reset_group(h5, group)
        _write_dataset(g, "positions", positions, compression=compression)
        _write_dataset(g, "radii", radii, compression=compression)
        _write_dataset(g, "n_particle", n_particle, compression=compression)
        g.attrs["n_medium"] = complex(n_medium)
        g.attrs["wavelength"] = float(wavelength)
        g.attrs["lmax"] = int(lmax)
        _write_attrs(g, attrs)


def load_geometry_h5(path: str | Path, *, group: str = "geometry") -> dict[str, Any]:
    """Load geometry payload (positions/radii/indexes) plus stored metadata."""
    out: dict[str, Any] = {}
    with h5py.File(_pathlike(path), "r") as h5:
        g = h5[group]
        out["positions"] = g["positions"][...]
        out["radii"] = g["radii"][...]
        out["n_particle"] = g["n_particle"][...]
        out["attrs"] = dict(g.attrs.items())
    return out


def save_solution_h5(
    path: str | Path,
    *,
    coeffs: np.ndarray,
    rhs: np.ndarray | None = None,
    initial_coeffs: np.ndarray | None = None,
    residual_history: np.ndarray | None = None,
    info: int | np.ndarray | None = None,
    group: str = "solution",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Persist solved multipole coefficients and optional solver diagnostics."""
    with h5py.File(_pathlike(path), mode) as h5:
        g = _reset_group(h5, group)
        _write_dataset(g, "coeffs", coeffs, compression=compression)
        if rhs is not None:
            _write_dataset(g, "rhs", rhs, compression=compression)
        if initial_coeffs is not None:
            _write_dataset(g, "initial_coeffs", initial_coeffs, compression=compression)
        if residual_history is not None:
            _write_dataset(g, "residual_history", residual_history, compression=compression)
        if info is not None:
            info_arr = np.asarray(info)
            if info_arr.shape == ():
                g.attrs["gmres_info"] = int(info_arr)
            else:
                _write_dataset(g, "gmres_info", info_arr, compression=compression)
        _write_attrs(g, attrs)


def load_solution_h5(path: str | Path, *, group: str = "solution") -> dict[str, Any]:
    """Load solved coefficients and optional residual/history arrays."""
    out: dict[str, Any] = {}
    with h5py.File(_pathlike(path), "r") as h5:
        g = h5[group]
        out["coeffs"] = g["coeffs"][...]
        if "rhs" in g:
            out["rhs"] = g["rhs"][...]
        if "initial_coeffs" in g:
            out["initial_coeffs"] = g["initial_coeffs"][...]
        if "residual_history" in g:
            out["residual_history"] = g["residual_history"][...]
        out["attrs"] = dict(g.attrs.items())
    return out


def _read_pwp(group: h5py.Group) -> dict[str, Any]:
    """Read one plane-wave pattern group (`alpha`,`beta`,`coeff`)."""
    out: dict[str, Any] = {}
    for key in ("alpha", "beta", "coeff"):
        if key in group:
            out[key] = group[key][...]
    out["attrs"] = dict(group.attrs.items())
    return out


def load_far_field_h5(path: str | Path, *, group: str = "far_field") -> dict[str, Any]:
    """Load far-field patterns saved via `save_far_field_h5`.

    Saved PWPs include ``alpha``, ``beta``, and ``coeff``. Cartesian wavevector
    grids (``kx``, ``ky``, ``kz``) are intentionally not persisted; reconstruct
    them from ``alpha``, ``beta``, and stored ``k_medium`` metadata when needed.
    """
    out: dict[str, Any] = {"patterns": {}, "attrs": {}}
    with h5py.File(_pathlike(path), "r") as h5:
        if group not in h5:
            return out
        root = h5[group]
        out["attrs"] = dict(root.attrs.items())
        for family_name, fam in root.items():
            if not isinstance(fam, h5py.Group):
                continue
            out["patterns"][family_name] = {}
            for pol_name, polg in fam.items():
                if not isinstance(polg, h5py.Group):
                    continue
                out["patterns"][family_name][pol_name] = _read_pwp(polg)
    return out


def save_near_field_h5(
    path: str | Path,
    *,
    X: np.ndarray,
    Z: np.ndarray,
    E: np.ndarray,
    H: np.ndarray,
    inside: np.ndarray | None = None,
    S: np.ndarray | None = None,
    group: str = "near_field",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Save one near-field slice payload (`E`,`H`) and optional masks/flux."""
    with h5py.File(_pathlike(path), mode) as h5:
        g = _reset_group(h5, group)
        _write_dataset(g, "X", X, compression=compression)
        _write_dataset(g, "Z", Z, compression=compression)
        _write_dataset(g, "E", E, compression=compression)
        _write_dataset(g, "H", H, compression=compression)
        if inside is not None:
            _write_dataset(g, "inside", inside, compression=compression)
        if S is not None:
            _write_dataset(g, "S", S, compression=compression)
        _write_attrs(g, attrs)


def load_near_field_components_h5(
    path: str | Path, *, group: str = "near_field_components"
) -> dict[str, Any]:
    """Load near-field component families saved via `save_near_field_components_h5`."""
    out: dict[str, Any] = {"fields": {}, "attrs": {}}
    with h5py.File(_pathlike(path), "r") as h5:
        root = h5[group]
        out["X"] = root["X"][...]
        out["Z"] = root["Z"][...]
        if "inside" in root:
            out["inside"] = root["inside"][...]
        out["attrs"] = dict(root.attrs.items())
        for name, obj in root.items():
            if name in {"X", "Z", "inside"}:
                continue
            if not isinstance(obj, h5py.Group):
                continue
            payload: dict[str, Any] = {}
            for key in ("E", "H", "S"):
                if key in obj:
                    payload[key] = obj[key][...]
            payload["attrs"] = dict(obj.attrs.items())
            out["fields"][name] = payload
    return out


def save_near_field_components_h5(
    path: str | Path,
    *,
    X: np.ndarray,
    Z: np.ndarray,
    fields: Mapping[str, Mapping[str, Any]],
    inside: np.ndarray | None = None,
    group: str = "near_field_components",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
    drop_redundant_total: bool = True,
) -> None:
    """Save multiple near-field families (e.g. initial/scattered/internal/total).

    Example `fields`:
      {
        "initial": {"E": E0, "H": H0},
        "scattered": {"E": Es, "H": Hs, "S": Ss},
        "total": {"E": Et, "H": Ht, "S": St},
      }
    When `drop_redundant_total=True` and both `initial` and `scattered` are
    present, a provided `total` family is omitted to avoid redundant storage.
    """
    fields_to_write = dict(fields)
    omitted_total = False
    if (
        drop_redundant_total
        and "total" in fields_to_write
        and "initial" in fields_to_write
        and "scattered" in fields_to_write
    ):
        fields_to_write.pop("total")
        omitted_total = True

    with h5py.File(_pathlike(path), mode) as h5:
        root = _reset_group(h5, group)
        _write_dataset(root, "X", X, compression=compression)
        _write_dataset(root, "Z", Z, compression=compression)
        if inside is not None:
            _write_dataset(root, "inside", inside, compression=compression)

        for family_name, payload in fields_to_write.items():
            fam = root.create_group(str(family_name))
            for key in ("E", "H", "S"):
                if key in payload:
                    _write_dataset(fam, key, payload[key], compression=compression)

        root.attrs["total_omitted_as_redundant"] = bool(omitted_total)
        _write_attrs(root, attrs)


def _write_pwp(
    group: h5py.Group,
    pwp: Mapping[str, Any],
    *,
    compression: str | None,
) -> None:
    """Write compact PWP payload (`alpha`,`beta`,`coeff`) into an HDF5 group."""
    for key in ("alpha", "beta", "coeff"):
        if key in pwp:
            _write_dataset(group, key, pwp[key], compression=compression)


def _write_mapping_recursive(
    group: h5py.Group, mapping: Mapping[str, Any], *, compression: str | None
) -> None:
    """Recursively serialize nested diagnostics dictionaries into HDF5."""
    for key, value in mapping.items():
        if value is None:
            continue
        name = str(key)
        if isinstance(value, Mapping):
            sub = group.create_group(name)
            _write_mapping_recursive(sub, value, compression=compression)
            continue
        if isinstance(value, (str, bytes, int, float, np.integer, np.floating, np.bool_)):
            group.attrs[name] = value
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S", "O"}:
            # Keep non-numeric scalars as attrs for readability.
            if arr.shape == ():
                group.attrs[name] = str(arr.item())
                continue
        _write_dataset(group, name, arr, compression=compression)


def save_mapping_h5(
    path: str | Path,
    *,
    mapping: Mapping[str, Any],
    group: str = "diagnostics",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Save a nested mapping of diagnostics/metadata in one HDF5 group."""
    with h5py.File(_pathlike(path), mode) as h5:
        root = _reset_group(h5, group)
        _write_mapping_recursive(root, mapping, compression=compression)
        _write_attrs(root, attrs)


def _read_mapping_recursive(group: h5py.Group) -> dict[str, Any]:
    """Recursively read diagnostics mapping from an HDF5 group tree."""
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        out[str(key)] = value
    for key, obj in group.items():
        name = str(key)
        if isinstance(obj, h5py.Group):
            out[name] = _read_mapping_recursive(obj)
        else:
            out[name] = obj[...]
    return out


def load_mapping_h5(path: str | Path, *, group: str = "diagnostics") -> dict[str, Any]:
    """Load nested diagnostics mapping saved via `save_mapping_h5`."""
    with h5py.File(_pathlike(path), "r") as h5:
        if group not in h5:
            return {}
        return _read_mapping_recursive(h5[group])


def save_far_field_h5(
    path: str | Path,
    *,
    patterns: Mapping[str, Mapping[str, Mapping[str, Any]]],
    group: str = "far_field",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
    drop_redundant_total: bool = True,
) -> None:
    """Save far-field PWPs under nested groups.

    Example `patterns`:
      {
        "initial": {"te": pwp_i_te, "tm": pwp_i_tm},
        "scattered": {"te": pwp_s_te, "tm": pwp_s_tm},
        "total": {"te": pwp_t_te, "tm": pwp_t_tm},
      }
    Only ``alpha``, ``beta``, and ``coeff`` are stored per polarization.
    When `drop_redundant_total=True` and both `initial` and `scattered` are
    present, a provided `total` family is omitted to avoid redundant storage.
    """
    patterns_to_write = dict(patterns)
    omitted_total = False
    if (
        drop_redundant_total
        and "total" in patterns_to_write
        and "initial" in patterns_to_write
        and "scattered" in patterns_to_write
    ):
        patterns_to_write.pop("total")
        omitted_total = True

    with h5py.File(_pathlike(path), mode) as h5:
        root = _reset_group(h5, group)
        for family_name, pol_map in patterns_to_write.items():
            fam = root.create_group(str(family_name))
            for pol_name, pwp in pol_map.items():
                polg = fam.create_group(str(pol_name))
                _write_pwp(polg, pwp, compression=compression)
        root.attrs["total_omitted_as_redundant"] = bool(omitted_total)
        _write_attrs(root, attrs)
