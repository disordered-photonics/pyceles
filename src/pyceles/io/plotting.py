from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from pyceles.core.sources import JonesPolarizedSource


def _is_pure_channel_result(run, channel: str, *, atol: float = 1e-12) -> bool:
    """Return True when `run` already represents one pure TE/TM Jones channel."""
    src = getattr(getattr(run, "config", None), "source", None)
    if src is None:
        return False
    if not isinstance(src, JonesPolarizedSource):
        return False
    if not hasattr(run, "polarization_jones"):
        return False
    if run.polarization_jones is None:
        return False
    a_te, a_tm = run.polarization_jones
    if channel == "te":
        return bool(abs(complex(a_tm)) <= atol and abs(complex(a_te)) > atol)
    if channel == "tm":
        return bool(abs(complex(a_te)) <= atol and abs(complex(a_tm)) > atol)
    return False


def _slice_axis_labels(plane: str) -> tuple[str, str]:
    """Return in-plane coordinate labels for a given slice normal axis."""
    p = str(plane).lower()
    if p == "x":
        return r"$y$", r"$z$"
    if p == "y":
        return r"$x$", r"$z$"
    if p == "z":
        return r"$x$", r"$y$"
    raise ValueError("plane must be one of {'x', 'y', 'z'}")


def _component_panel_title(component_name: str) -> str:
    """Return a math-formatted near-field panel title."""
    c = str(component_name).strip().lower()
    mapping = {
        "real ex": r"$\operatorname{Re} E_x$",
        "real ey": r"$\operatorname{Re} E_y$",
        "real ez": r"$\operatorname{Re} E_z$",
        "real hx": r"$\operatorname{Re} H_x$",
        "real hy": r"$\operatorname{Re} H_y$",
        "real hz": r"$\operatorname{Re} H_z$",
        "abs e": r"$|E|$",
        "abs h": r"$|H|$",
    }
    return mapping.get(c, component_name)


def _imshow_extent_from_center_grids(axis_0: np.ndarray, axis_1: np.ndarray) -> list[float]:
    """Return edge-based imshow extent for center-sampled Cartesian grids.

    Near-field slices store coordinate arrays at pixel centers. Matplotlib's
    `imshow(..., extent=...)` expects outer image edges. Converting center
    coordinates to edge coordinates avoids a visual half-pixel offset between
    field pixels and geometric overlays.
    """

    def _line_edges(values: np.ndarray) -> tuple[float, float]:
        line = np.asarray(values, dtype=float).reshape(-1)
        if line.size == 0:
            raise ValueError("Cannot infer image extent from an empty axis.")
        if line.size == 1:
            span = 1.0
        else:
            d = np.abs(np.diff(line))
            d = d[np.isfinite(d) & (d > 0.0)]
            span = float(np.median(d)) if d.size > 0 else 1.0
        lo = float(np.min(line)) - 0.5 * span
        hi = float(np.max(line)) + 0.5 * span
        return lo, hi

    a0 = np.asarray(axis_0, dtype=float)
    a1 = np.asarray(axis_1, dtype=float)
    a0_line = a0[0, :] if a0.ndim == 2 else a0
    a1_line = a1[:, 0] if a1.ndim == 2 else a1
    x0, x1 = _line_edges(a0_line)
    y0, y1 = _line_edges(a1_line)
    return [x0, x1, y0, y1]


def near_field_component(E: np.ndarray, H: np.ndarray, component: str) -> np.ndarray:
    """Extract a scalar map from vector near fields.

    Supported components:
      - `real Ex`, `real Ey`, `real Ez`
      - `real Hx`, `real Hy`, `real Hz`
      - `abs E`, `abs H`
    """
    c = str(component).strip().lower()
    e = np.asarray(E)
    h = np.asarray(H)
    if c == "real ex":
        return np.real(e[..., 0])
    if c == "real ey":
        return np.real(e[..., 1])
    if c == "real ez":
        return np.real(e[..., 2])
    if c == "real hx":
        return np.real(h[..., 0])
    if c == "real hy":
        return np.real(h[..., 1])
    if c == "real hz":
        return np.real(h[..., 2])
    if c == "abs e":
        return np.sqrt(np.sum(np.abs(e) ** 2, axis=-1))
    if c == "abs h":
        return np.sqrt(np.sum(np.abs(h) ** 2, axis=-1))
    raise ValueError(f"Unsupported component '{component}'.")


def unpolarized_near_field_intensity(E_te: np.ndarray, E_tm: np.ndarray) -> np.ndarray:
    """Return incoherent unpolarized near-field intensity map.

    Definition:
      I_unpol = 0.5 * (|E_TE|^2 + |E_TM|^2)

    This is an intensity-level average; TE/TM are not coherently summed.
    """
    e_te = np.asarray(E_te)
    e_tm = np.asarray(E_tm)
    return 0.5 * (np.sum(np.abs(e_te) ** 2, axis=-1) + np.sum(np.abs(e_tm) ** 2, axis=-1))


def far_field_intensity(pwp_te: dict, pwp_tm: dict) -> np.ndarray:
    """Return combined PWP intensity |g_TE|^2 + |g_TM|^2."""
    return np.abs(np.asarray(pwp_te["coeff"])) ** 2 + np.abs(np.asarray(pwp_tm["coeff"])) ** 2


def far_field_intensity_from_result(run, *, channel: str = "mixed") -> np.ndarray:
    """Return far-field intensity map from a `SimulationResult`.

    Parameters
    ----------
    channel:
        - ``"mixed"``: source-requested polarization state (`run.farfield`)
        - ``"te"`` or ``"tm"``: pure basis channels (`run.farfield_basis`), or
          a pure single-channel result returned by
          `postprocess_sources(solve_sources(...))`
        - ``"unpolarized"``: incoherent average ``0.5*(I_te + I_tm)``
    """
    ch = str(channel).lower()
    if ch == "mixed":
        return far_field_intensity(run.farfield.scattered_te, run.farfield.scattered_tm)
    if ch in {"te", "tm"}:
        if run.farfield_basis is not None and ch in run.farfield_basis:
            ff = run.farfield_basis[ch]
            return far_field_intensity(ff.scattered_te, ff.scattered_tm)
        if _is_pure_channel_result(run, ch):
            return far_field_intensity(run.farfield.scattered_te, run.farfield.scattered_tm)
        raise ValueError(
            "Requested basis far-field channel, but no TE/TM basis payload is available on this run. "
            "Use `solve_polarization_basis=True` with `Simulation.run()`, or use a channel result from "
            "`Simulation.postprocess_sources(Simulation.solve_sources(...))` and query it with "
            "`channel='mixed'`."
        )
    if ch == "unpolarized":
        if (
            run.farfield_basis is None
            or "te" not in run.farfield_basis
            or "tm" not in run.farfield_basis
        ):
            raise ValueError(
                "Requested unpolarized intensity, but TE/TM basis far fields are unavailable. "
                "Use `solve_polarization_basis=True` with `Simulation.run()`, or compute it from "
                "`Simulation.postprocess_sources(Simulation.solve_sources(...))` by averaging "
                "TE/TM channel intensities."
            )
        I_te = far_field_intensity(
            run.farfield_basis["te"].scattered_te, run.farfield_basis["te"].scattered_tm
        )
        I_tm = far_field_intensity(
            run.farfield_basis["tm"].scattered_te, run.farfield_basis["tm"].scattered_tm
        )
        return 0.5 * (I_te + I_tm)
    raise ValueError("channel must be one of {'mixed', 'te', 'tm', 'unpolarized'}.")


def plot_spheres(
    ax,
    positions,
    radii,
    *,
    plane="y",
    plane_value=0.0,
    alpha=0.6,
    color="w",
    linewidth=1.0,
):
    """Plot sphere intersections with a Cartesian slice plane.

    Parameters
    ----------
    plane:
        Slice plane normal axis: "x", "y", or "z".
    plane_value:
        Slice location along `plane`.
    """
    plane = str(plane).lower()
    if plane not in {"x", "y", "z"}:
        raise ValueError("plane must be one of {'x', 'y', 'z'}")

    # Map slice plane to displayed coordinate pair.
    if plane == "x":
        slice_idx = 0
        plot_i, plot_j = 1, 2
    elif plane == "y":
        slice_idx = 1
        plot_i, plot_j = 0, 2
    else:
        slice_idx = 2
        plot_i, plot_j = 0, 1

    pos = np.asarray(positions, float)
    r = np.asarray(radii, float)
    d = pos[:, slice_idx] - float(plane_value)
    mask = np.abs(d) <= r

    for p, rr, dd in zip(pos[mask], r[mask], d[mask]):
        rp = np.sqrt(max(0.0, rr * rr - dd * dd))
        ax.add_patch(
            plt.Circle(
                (p[plot_i], p[plot_j]),
                rp,
                edgecolor=color,
                fill=False,
                alpha=alpha,
                linewidth=linewidth,
            )
        )
    return ax


def plot_field_component(
    ax,
    axis_0,
    axis_1,
    F,
    title="",
    cmap=None,
    vmin=None,
    vmax=None,
    *,
    axis_0_label: str = "axis 0",
    axis_1_label: str = "axis 1",
):
    """Plot one scalar field map over a 2D slice grid with equal aspect."""
    extent = _imshow_extent_from_center_grids(axis_0, axis_1)
    im = ax.imshow(
        F,
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xlabel(axis_0_label)
    ax.set_ylabel(axis_1_label)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return im


def plot_intensity(
    ax,
    axis_0,
    axis_1,
    I,
    title="Intensity |E|^2",
    *,
    axis_0_label: str = "axis 0",
    axis_1_label: str = "axis 1",
):
    """Convenience wrapper to plot scalar intensity-like maps."""
    return plot_field_component(
        ax,
        axis_0,
        axis_1,
        I,
        title=title,
        axis_0_label=axis_0_label,
        axis_1_label=axis_1_label,
    )


def plot_poynting(
    ax,
    axis_0,
    axis_1,
    Sx,
    Sz,
    *,
    intensity=None,
    intensity_cmap="inferno",
    stride=10,
    quiver_color="w",
    title="Poynting vector",
    axis_0_label: str = "axis 0",
    axis_1_label: str = "axis 1",
):
    """Plot 2D Poynting-vector quiver, optionally over intensity background."""
    if intensity is not None:
        extent = _imshow_extent_from_center_grids(axis_0, axis_1)
        ax.imshow(
            intensity,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap=intensity_cmap,
        )
    ax.quiver(
        axis_0[::stride, ::stride],
        axis_1[::stride, ::stride],
        Sx[::stride, ::stride],
        Sz[::stride, ::stride],
        color=quiver_color,
    )
    ax.set_xlabel(axis_0_label)
    ax.set_ylabel(axis_1_label)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_farfield_hemispheres(
    polar_angles,
    azimuthal_angles,
    intensity,
    *,
    cmap="viridis",
    independent_scales: bool = False,
    vmin_zero: bool = True,
    title=r"Far-field plane-wave pattern intensity $|g(\alpha,\beta)|^2$",
):
    """Plot two polar panels: forward (+z) and backward (-z) hemispheres."""
    beta = np.asarray(polar_angles, dtype=float)
    alpha = np.asarray(azimuthal_angles, dtype=float)
    I = np.asarray(intensity)
    if I.shape != (alpha.size, beta.size):
        raise ValueError("intensity must have shape (len(azimuthal_angles), len(polar_angles))")

    fwd = beta <= (0.5 * np.pi)
    bwd = beta >= (0.5 * np.pi)

    Af, Bf = np.meshgrid(alpha, beta[fwd], indexing="ij")
    Ab, Bb = np.meshgrid(alpha, np.pi - beta[bwd], indexing="ij")

    if independent_scales:
        vmin_f = 0.0 if vmin_zero else float(np.min(I[:, fwd]))
        vmax_f = float(np.max(I[:, fwd]))
        vmin_b = 0.0 if vmin_zero else float(np.min(I[:, bwd]))
        vmax_b = float(np.max(I[:, bwd]))
    else:
        vmin_f = vmin_b = 0.0 if vmin_zero else float(np.min(I))
        vmax_f = vmax_b = float(np.max(I))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
        subplot_kw={"projection": "polar"},
    )
    im0 = axes[0].pcolormesh(Af, Bf, I[:, fwd], shading="auto", cmap=cmap, vmin=vmin_f, vmax=vmax_f)
    axes[0].set_title(r"Forward hemisphere ($+z$)")
    axes[0].set_ylim(0.0, 0.5 * np.pi)
    axes[0].set_ylabel(r"$\beta$")

    im1 = axes[1].pcolormesh(Ab, Bb, I[:, bwd], shading="auto", cmap=cmap, vmin=vmin_b, vmax=vmax_b)
    axes[1].set_title(r"Backward hemisphere ($-z$)")
    axes[1].set_ylim(0.0, 0.5 * np.pi)
    axes[1].set_ylabel(r"$\pi-\beta$")

    fig.suptitle(title)
    if independent_scales:
        fig.colorbar(im0, ax=axes[0], shrink=0.9, label=r"$|g(\alpha,\beta)|^2$ (fwd)")
        fig.colorbar(im1, ax=axes[1], shrink=0.9, label=r"$|g(\alpha,\beta)|^2$ (bwd)")
    else:
        fig.colorbar(im0, ax=axes, shrink=0.9, label=r"$|g(\alpha,\beta)|^2$")
    return fig, axes


def _nearfield_panel_component_names() -> tuple[str, ...]:
    """Canonical 8-panel near-field component order."""
    return (
        "real Ex",
        "real Ey",
        "real Ez",
        "abs E",
        "real Hx",
        "real Hy",
        "real Hz",
        "abs H",
    )


def _plot_nearfield_panel_block(
    fig,
    axes_block: np.ndarray,
    *,
    axis_0: np.ndarray,
    axis_1: np.ndarray,
    E: np.ndarray,
    H: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    plane: str,
    plane_value: float,
    axis_0_label: str,
    axis_1_label: str,
    real_limits: tuple[float, float],
    abs_limits: tuple[float, float],
    title_prefix: str = "",
) -> None:
    """Render one 2x4 near-field panel block on preallocated axes."""
    components = [
        (name, near_field_component(E, H, name)) for name in _nearfield_panel_component_names()
    ]
    for ax, (name, F) in zip(np.asarray(axes_block).flat, components):
        title = _component_panel_title(name)
        if title_prefix:
            title = f"{title_prefix}: {title}"
        im = plot_field_component(
            ax,
            axis_0,
            axis_1,
            F,
            title=title,
            axis_0_label=axis_0_label,
            axis_1_label=axis_1_label,
        )

        if name.startswith("real"):
            im.set_cmap("RdBu")
            im.set_clim(real_limits[0], real_limits[1])
            sphere_color = "k"
        else:
            im.set_cmap("inferno")
            im.set_clim(abs_limits[0], abs_limits[1])
            sphere_color = "w"

        plot_spheres(
            ax,
            positions,
            radii,
            plane=plane,
            plane_value=plane_value,
            alpha=0.7,
            color=sphere_color,
        )
        fig.colorbar(im, ax=ax, shrink=0.85)


def plot_nearfield_panels(
    axis_0: np.ndarray,
    axis_1: np.ndarray,
    E: np.ndarray,
    H: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    plane: str = "y",
    plane_value: float = 0.0,
    real_limits: tuple[float, float] = (-2.0, 2.0),
    abs_limits: tuple[float, float] = (0.0, 2.0),
):
    """Plot 8 near-field panels with sphere overlays."""
    axis_0_label, axis_1_label = _slice_axis_labels(plane)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)
    _plot_nearfield_panel_block(
        fig,
        axes,
        axis_0=axis_0,
        axis_1=axis_1,
        E=E,
        H=H,
        positions=positions,
        radii=radii,
        plane=plane,
        plane_value=plane_value,
        axis_0_label=axis_0_label,
        axis_1_label=axis_1_label,
        real_limits=real_limits,
        abs_limits=abs_limits,
    )
    return fig, axes


def plot_nearfield_panels_channels(
    axis_0: np.ndarray,
    axis_1: np.ndarray,
    channel_fields: dict[str, tuple[np.ndarray, np.ndarray]],
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    plane: str = "y",
    plane_value: float = 0.0,
    real_limits: tuple[float, float] = (-2.0, 2.0),
    abs_limits: tuple[float, float] = (0.0, 2.0),
):
    """Plot near-field 8-panel sets for multiple polarization channels.

    Parameters
    ----------
    channel_fields:
        Mapping ``channel_name -> (E, H)``. Each channel contributes one 8-panel
        set (2 rows x 4 columns). For two channels (e.g. TE/TM) this produces
        16 panels.
    """
    if not channel_fields:
        raise ValueError("`channel_fields` must not be empty.")

    channels = list(channel_fields.items())
    axis_0_label, axis_1_label = _slice_axis_labels(plane)

    n_channels = len(channels)
    fig, axes = plt.subplots(
        2 * n_channels, 4, figsize=(18, 7 * n_channels), constrained_layout=True
    )
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape(2, 4)

    for i, (ch_name, (E, H)) in enumerate(channels):
        block_axes = axes_arr[(2 * i) : (2 * i + 2), :]
        _plot_nearfield_panel_block(
            fig,
            block_axes,
            axis_0=axis_0,
            axis_1=axis_1,
            E=E,
            H=H,
            positions=positions,
            radii=radii,
            plane=plane,
            plane_value=plane_value,
            axis_0_label=axis_0_label,
            axis_1_label=axis_1_label,
            real_limits=real_limits,
            abs_limits=abs_limits,
            title_prefix=str(ch_name),
        )

    return fig, axes_arr


def plot_source_showcase_slices(
    run,
    *,
    field_component: Literal["initial", "scattered", "internal", "total"] = "initial",
    channel: Literal["mixed", "te", "tm"] = "mixed",
    plane_values: tuple[float, float, float] = (0.0, 0.0, 0.0),
    phase_component: Literal["Ex", "Ey", "Ez"] = "Ex",
    phase_cmap: str = "twilight_shifted",
    axis_0_min: float = -4000.0,
    axis_0_max: float = 4000.0,
    axis_1_min: float = -3000.0,
    axis_1_max: float = 5000.0,
    dx: float = 40.0,
    real_limits: tuple[float, float] | None = None,
    abs_limits: tuple[float, float] | None = None,
    rowwise_percentile: float = 99.5,
    show_progress: bool = True,
    force_general_initial_field: bool | None = None,
    center_pixel_policy: Literal["none", "interpolate"] = "interpolate",
):
    """Plot source-centric near-field showcase on x=const, y=const, z=const slices.

    Layout is 3 rows x 5 columns:
    - rows: planes x/y/z (in that order),
    - columns 1..3: ``Re(Ex)``, ``Re(Ey)``, ``Re(Ez)`` on each slice,
    - columns 4..5: ``|E|`` and one complex phase map on every slice.

    Normalization policy:
    - by default (`real_limits=None`, `abs_limits=None`) each row uses robust
      percentile-based limits to reduce saturation while preserving per-row
      consistency across columns;
    - explicit limits override that auto-scaling.
    """
    from pyceles.postprocessing.workflows import compute_near_field_slice

    family = str(field_component).lower()
    if family not in {"initial", "scattered", "internal", "total"}:
        raise ValueError(
            "`field_component` must be one of {'initial', 'scattered', 'internal', 'total'}."
        )

    phase_name = str(phase_component).strip().lower()
    phase_idx_map = {"ex": 0, "ey": 1, "ez": 2}
    if phase_name not in phase_idx_map:
        raise ValueError("`phase_component` must be one of {'Ex', 'Ey', 'Ez'}.")
    phase_idx = phase_idx_map[phase_name]
    if not (0.0 < float(rowwise_percentile) <= 100.0):
        raise ValueError("`rowwise_percentile` must lie in (0, 100].")

    planes = ("x", "y", "z")
    if len(tuple(plane_values)) != 3:
        raise ValueError("`plane_values` must contain exactly three entries for x/y/z.")

    slices = []
    for plane, plane_value in zip(planes, plane_values):
        slices.append(
            compute_near_field_slice(
                run,
                axis_0_min=float(axis_0_min),
                axis_0_max=float(axis_0_max),
                axis_1_min=float(axis_1_min),
                axis_1_max=float(axis_1_max),
                dx=float(dx),
                plane=plane,
                plane_value=float(plane_value),
                channel=channel,
                show_progress=bool(show_progress),
                force_general_initial_field=force_general_initial_field,
                center_pixel_policy=center_pixel_policy,
            )
        )

    fig, axes = plt.subplots(3, 5, figsize=(24, 13), constrained_layout=True)

    for row, (plane, slc) in enumerate(zip(planes, slices)):
        E, H = slc.field_maps[family]
        E_arr = np.asarray(E)

        if real_limits is None:
            real_vals = np.abs(np.real(E_arr[..., :3])).reshape(-1)
            real_vals = real_vals[np.isfinite(real_vals)]
            if real_vals.size > 0:
                real_peak = float(np.percentile(real_vals, float(rowwise_percentile)))
            else:
                real_peak = 1e-12
            real_peak = max(real_peak, 1e-12)
            row_real_limits = (-real_peak, real_peak)
        else:
            row_real_limits = (float(real_limits[0]), float(real_limits[1]))

        abs_e = near_field_component(E, H, "abs E")
        if abs_limits is None:
            abs_vals = np.asarray(abs_e, dtype=float).reshape(-1)
            abs_vals = abs_vals[np.isfinite(abs_vals)]
            if abs_vals.size > 0:
                abs_peak = float(np.percentile(abs_vals, float(rowwise_percentile)))
            else:
                abs_peak = 1e-12
            abs_peak = max(abs_peak, 1e-12)
            row_abs_limits = (0.0, abs_peak)
        else:
            row_abs_limits = (float(abs_limits[0]), float(abs_limits[1]))

        for col, comp in enumerate(("real Ex", "real Ey", "real Ez")):
            F = near_field_component(E, H, comp)
            title = f"{plane}= {float(slc.plane_value):.3g}: {_component_panel_title(comp)}"
            im = plot_field_component(
                axes[row, col],
                slc.axis_0,
                slc.axis_1,
                F,
                title=title,
                axis_0_label=slc.axis_0_label,
                axis_1_label=slc.axis_1_label,
            )
            im.set_cmap("RdBu")
            im.set_clim(row_real_limits[0], row_real_limits[1])
            plot_spheres(
                axes[row, col],
                run.positions,
                run.radii,
                plane=plane,
                plane_value=float(slc.plane_value),
                alpha=0.7,
                color="k",
            )
            fig.colorbar(im, ax=axes[row, col], shrink=0.85)

        im_abs = plot_field_component(
            axes[row, 3],
            slc.axis_0,
            slc.axis_1,
            abs_e,
            title=f"{plane}= {float(slc.plane_value):.3g}: $|E|$",
            axis_0_label=slc.axis_0_label,
            axis_1_label=slc.axis_1_label,
        )
        im_abs.set_cmap("inferno")
        im_abs.set_clim(row_abs_limits[0], row_abs_limits[1])
        plot_spheres(
            axes[row, 3],
            run.positions,
            run.radii,
            plane=plane,
            plane_value=float(slc.plane_value),
            alpha=0.7,
            color="w",
        )
        fig.colorbar(im_abs, ax=axes[row, 3], shrink=0.85)

        phase = np.angle(E_arr[..., phase_idx])
        phase_label = ("Ex", "Ey", "Ez")[phase_idx]
        im_phase = plot_field_component(
            axes[row, 4],
            slc.axis_0,
            slc.axis_1,
            phase,
            title=f"{plane}= {float(slc.plane_value):.3g}: $\\arg({phase_label})$",
            axis_0_label=slc.axis_0_label,
            axis_1_label=slc.axis_1_label,
        )
        im_phase.set_cmap(str(phase_cmap))
        im_phase.set_clim(-np.pi, np.pi)
        plot_spheres(
            axes[row, 4],
            run.positions,
            run.radii,
            plane=plane,
            plane_value=float(slc.plane_value),
            alpha=0.7,
            color="w",
        )
        fig.colorbar(im_phase, ax=axes[row, 4], shrink=0.85)

    return fig, axes


def plot_nearfield_poynting_overlay(
    axis_0: np.ndarray,
    axis_1: np.ndarray,
    E: np.ndarray,
    H: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    plane: str = "y",
    plane_value: float = 0.0,
    stride: int = 10,
    quiver_color: str = "w",
    title: str = "Poynting vector on |E|^2",
):
    """Plot Poynting quiver on top of |E|^2 map."""
    axis_0_label, axis_1_label = _slice_axis_labels(plane)
    S = 0.5 * np.real(np.cross(E, np.conj(H)))
    absE = near_field_component(E, H, "abs E")
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    plot_poynting(
        ax,
        axis_0,
        axis_1,
        S[..., 0],
        S[..., 2],
        intensity=absE,
        intensity_cmap="inferno",
        stride=stride,
        quiver_color=quiver_color,
        title=title,
        axis_0_label=axis_0_label,
        axis_1_label=axis_1_label,
    )
    plot_spheres(ax, positions, radii, plane=plane, plane_value=plane_value, alpha=0.7, color="w")
    ax.set_xlabel(axis_0_label)
    ax.set_ylabel(axis_1_label)
    return fig, ax
