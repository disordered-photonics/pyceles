from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pyceles.core.particles import (
    LayeredSphere,
    Particle,
    ParticleCollection,
    PECSphere,
    Sphere,
    Spheroid,
)
from pyceles.core.plane_wave_spectrum import PlaneWaveSpectrum
from pyceles.postprocessing.farfield.patterns import FarFieldPatterns


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


def _write_particle_payload(
    pgroup: h5py.Group,
    particle: Particle,
    *,
    compression: str | None,
) -> None:
    """Persist one position-independent particle archetype."""
    if isinstance(particle, PECSphere):
        pgroup.attrs["type"] = "PECSphere"
        _write_dataset(pgroup, "radius", float(particle.radius), compression=None)
    elif isinstance(particle, Sphere):
        pgroup.attrs["type"] = "Sphere"
        _write_dataset(pgroup, "radius", float(particle.radius), compression=None)
        _write_dataset(
            pgroup,
            "refractive_index",
            np.asarray(complex(particle.refractive_index), dtype=np.complex128),
            compression=None,
        )
    elif isinstance(particle, LayeredSphere):
        pgroup.attrs["type"] = "LayeredSphere"
        _write_dataset(
            pgroup,
            "layer_radii",
            np.asarray(particle.layer_radii, dtype=float),
            compression=compression,
        )
        _write_dataset(
            pgroup,
            "layer_refractive_indices",
            np.asarray(particle.layer_refractive_indices, dtype=np.complex128),
            compression=compression,
        )
    elif isinstance(particle, Spheroid):
        pgroup.attrs["type"] = "Spheroid"
        _write_dataset(
            pgroup,
            "equatorial_radius",
            np.asarray(float(particle.equatorial_radius), dtype=float),
            compression=None,
        )
        _write_dataset(
            pgroup,
            "polar_radius",
            np.asarray(float(particle.polar_radius), dtype=float),
            compression=None,
        )
        _write_dataset(
            pgroup,
            "refractive_index",
            np.asarray(complex(particle.refractive_index), dtype=np.complex128),
            compression=None,
        )
        _write_dataset(
            pgroup,
            "euler_angles",
            np.asarray(particle.euler_angles, dtype=float),
            compression=None,
        )
    else:
        raise TypeError(f"Unsupported particle type {type(particle).__name__!r}.")


def _write_particle_descriptors(
    group: h5py.Group,
    particles: Sequence[Particle],
    *,
    compression: str | None,
) -> None:
    """Persist the canonical instance/archetype particle representation."""
    collection = ParticleCollection.from_particles(particles)
    pg = group.create_group("particles")
    pg.attrs["schema"] = "pyceles.particles.v2"
    pg.attrs["count"] = len(collection)
    pg.attrs["archetype_count"] = collection.n_archetypes
    _write_dataset(pg, "positions", collection.positions, compression=compression)
    _write_dataset(
        pg,
        "archetype_indices",
        collection.archetype_indices,
        compression=compression,
    )
    archetypes_group = pg.create_group("archetypes")
    for index, particle in enumerate(collection.archetypes):
        _write_particle_payload(
            archetypes_group.create_group(str(index)),
            particle,
            compression=compression,
        )


def _load_particle_payload(pgroup: h5py.Group, *, position: tuple[float, float, float]) -> Particle:
    """Load one typed particle payload at the requested position."""
    kind = str(pgroup.attrs.get("type", ""))
    if kind == "PECSphere":
        return PECSphere(
            position=position,
            radius=float(np.asarray(pgroup["radius"][...]).reshape(())),
        )
    if kind == "Sphere":
        return Sphere(
            position=position,
            radius=float(np.asarray(pgroup["radius"][...]).reshape(())),
            refractive_index=complex(np.asarray(pgroup["refractive_index"][...]).reshape(())),
        )
    if kind == "LayeredSphere":
        return LayeredSphere(
            position=position,
            layer_radii=tuple(
                float(value)
                for value in np.asarray(pgroup["layer_radii"][...], dtype=float).reshape(-1)
            ),
            layer_refractive_indices=tuple(
                complex(value)
                for value in np.asarray(
                    pgroup["layer_refractive_indices"][...], dtype=np.complex128
                ).reshape(-1)
            ),
        )
    if kind == "Spheroid":
        euler_arr = np.asarray(pgroup["euler_angles"][...], dtype=float).reshape(3)
        return Spheroid(
            position=position,
            equatorial_radius=float(
                np.asarray(pgroup["equatorial_radius"][...], dtype=float).reshape(())
            ),
            polar_radius=float(np.asarray(pgroup["polar_radius"][...], dtype=float).reshape(())),
            refractive_index=complex(np.asarray(pgroup["refractive_index"][...]).reshape(())),
            euler_angles=(float(euler_arr[0]), float(euler_arr[1]), float(euler_arr[2])),
        )
    raise ValueError(f"Unsupported or missing particle type attribute: {kind!r}.")


def _load_particle_descriptors(group: h5py.Group) -> ParticleCollection:
    """Load particles from the canonical shared-archetype schema."""
    if "particles" not in group:
        raise ValueError(
            "Missing required geometry payload `particles`. "
            "This file does not follow the canonical particle-native geometry schema."
        )
    pg = group["particles"]
    schema = str(pg.attrs.get("schema", ""))
    if schema != "pyceles.particles.v2":
        raise ValueError(f"Unsupported particle schema {schema!r}.")
    positions = np.asarray(pg["positions"][...], dtype=float).reshape(-1, 3)
    archetype_indices = np.asarray(pg["archetype_indices"][...]).reshape(-1)
    archetypes_group = pg["archetypes"]
    keys = sorted(archetypes_group.keys(), key=lambda key: int(key))
    archetypes = tuple(
        _load_particle_payload(
            archetypes_group[key],
            position=(0.0, 0.0, 0.0),
        )
        for key in keys
    )
    return ParticleCollection.from_archetypes(
        positions=positions,
        archetypes=archetypes,
        archetype_indices=archetype_indices,
    )


def save_geometry_h5(
    path: str | Path,
    *,
    particles: Sequence[Particle],
    n_medium: complex,
    wavelength: float,
    lmax: int,
    group: str = "geometry",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Persist canonical particle instances, shared archetypes, and optical metadata."""
    with h5py.File(_pathlike(path), mode) as h5:
        g = _reset_group(h5, group)
        _write_particle_descriptors(g, particles, compression=compression)
        g.attrs["n_medium"] = complex(n_medium)
        g.attrs["wavelength"] = float(wavelength)
        g.attrs["lmax"] = int(lmax)
        _write_attrs(g, attrs)


def load_geometry_h5(path: str | Path, *, group: str = "geometry") -> dict[str, Any]:
    """Load geometry as a canonical ``ParticleCollection`` plus metadata attrs."""
    out: dict[str, Any] = {}
    with h5py.File(_pathlike(path), "r") as h5:
        g = h5[group]
        out["particles"] = _load_particle_descriptors(g)
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


def load_far_field_h5(path: str | Path, *, group: str = "far_field") -> FarFieldPatterns:
    """Load finite far-field patterns saved via `save_far_field_h5`."""
    with h5py.File(_pathlike(path), "r") as h5:
        root = h5[group]
        grid = root["grid"]
        alpha = grid["alpha"][...]
        beta = grid["beta"][...]
        kx = grid["kx"][...]
        ky = grid["ky"][...]
        kz = grid["kz"][...]

        def load_spectrum(name: str) -> PlaneWaveSpectrum:
            spectrum = root[name]
            return PlaneWaveSpectrum(
                alpha,
                beta,
                kx,
                ky,
                kz,
                spectrum["coeff_te"][...],
                spectrum["coeff_tm"][...],
            )

        scattered = load_spectrum("scattered")
        initial = load_spectrum("initial") if "initial" in root else None
    return FarFieldPatterns(initial=initial, scattered=scattered)


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


def _write_mapping_recursive(
    group: h5py.Group, mapping: Mapping[str, Any], *, compression: str | None
) -> None:
    """Recursively serialize nested diagnostics dictionaries into HDF5."""
    for key, value in mapping.items():
        if value is None:
            continue
        name = str(key)
        to_mapping = getattr(value, "to_mapping", None)
        if callable(to_mapping):
            value = to_mapping()
        if isinstance(value, Mapping):
            sub = group.create_group(name)
            _write_mapping_recursive(sub, value, compression=compression)
            continue
        if isinstance(value, (str, bytes, int, float, np.integer, np.floating, np.bool_)):
            group.attrs[name] = value
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S", "O"} and arr.shape == ():
            # Keep non-numeric scalars as attrs for readability.
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


def _periodic_to_mapping(periodic: Any, *, include_power: bool = True) -> dict[str, Any]:
    """Normalize periodic payload objects to a mapping for HDF5 serialization."""
    if isinstance(periodic, Mapping):
        out = dict(periodic)
        if not include_power:
            out.pop("power", None)
        return out
    keys: tuple[str, ...] = (
        "lattice_a1",
        "lattice_a2",
        "unit_cell_area",
        "incident_k_parallel",
        "output_bmax",
        "order_mn",
        "order_k_parallel",
        "order_kz",
        "order_propagating",
        "reflected_amplitudes",
        "transmitted_amplitudes",
        "reflected_flux_per_order",
        "transmitted_flux_per_order",
        "incident_flux",
    )
    if include_power:
        keys = (*keys, "power")
    payload: dict[str, Any] = {}
    for key in keys:
        if hasattr(periodic, key):
            payload[key] = getattr(periodic, key)
    if not payload:
        raise TypeError(
            "Unsupported periodic payload. Expected mapping or object with periodic fields."
        )
    return payload


def save_periodic_h5(
    path: str | Path,
    *,
    periodic: Any,
    group: str = "periodic",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
    include_power: bool = True,
) -> None:
    """Save periodic orders and optionally the nested common power balance."""
    save_mapping_h5(
        path,
        mapping=_periodic_to_mapping(periodic, include_power=include_power),
        group=group,
        mode=mode,
        attrs=attrs,
        compression=compression,
    )


def load_periodic_h5(path: str | Path, *, group: str = "periodic") -> dict[str, Any]:
    """Load periodic diffraction-order payload saved via `save_periodic_h5`."""
    return load_mapping_h5(path, group=group)


def save_far_field_h5(
    path: str | Path,
    *,
    farfield: FarFieldPatterns,
    group: str = "far_field",
    mode: str = "a",
    attrs: Mapping[str, Any] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Save finite far fields with one grid and no derived total copy."""
    reference = farfield.scattered
    if farfield.initial is not None:
        try:
            reference.require_same_grid(farfield.initial)
        except ValueError as exc:
            raise ValueError("Initial and scattered far fields must share one grid.") from exc
    with h5py.File(_pathlike(path), mode) as h5:
        root = _reset_group(h5, group)
        grid = root.create_group("grid")
        for name in ("alpha", "beta", "kx", "ky", "kz"):
            _write_dataset(
                grid,
                name,
                getattr(reference, name),
                compression=compression,
            )

        def save_spectrum(name: str, spectrum: PlaneWaveSpectrum) -> None:
            target = root.create_group(name)
            _write_dataset(target, "coeff_te", spectrum.coeff_te, compression=compression)
            _write_dataset(target, "coeff_tm", spectrum.coeff_tm, compression=compression)

        save_spectrum("scattered", farfield.scattered)
        if farfield.initial is not None:
            save_spectrum("initial", farfield.initial)
        _write_attrs(root, attrs)
