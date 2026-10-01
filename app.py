"""Streamlit web UI for the Peclet-normalized cutting thermal model (Ti-6Al-4V).

One page: process inputs in the sidebar, key results as metrics, then
  1D:  normalized (Peclet) surface shape, flash-temperature convergence,
       actual surface temperature
  2D:  subsurface temperature field with residual-stress contours overlaid,
       a zoom on the iso-RS region, and residual stress vs depth in µm
       zoomed on the yielded layer
  Machining: Cutting force and flash temperature vs speed, critical speed vs feed

Run with: uv run python app.py (or uv run streamlit run app.py).
"""

# An editor's Run button executes this file as plain Python. Hand that
# invocation to Streamlit before making any UI calls. Streamlit executes
# the file again with a ScriptRunContext, so this does not launch recursively.
if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is None:
        import sys
        from pathlib import Path

        from streamlit.web import cli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        raise SystemExit(cli.main())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from model import (CALIBRATION, DATA_H, DATA_T_C, DATA_V_M_MIN, KIENZLE_TI64, TI64,
                   critical_speed, critical_speed_fit, flash_vs_speed, force_vs_feed,
                   kienzle_force, residual_stress, solve, solve_2d)

# --- Palette (dataviz skill reference instance, light mode) -----------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
FEED_COLORS = [BLUE, ORANGE, "#1a9850"]
RS_COLORS = [BLUE, "#1a9850", "#8e44ad", "#0f4c81"]  # one per RS contour level

# Sequential light->dark ramp in the orange family for the temperature field.
TEMP_CMAP = LinearSegmentedColormap.from_list("cutting_temp", [SURFACE, ORANGE, "#7a2e0e"])

X_OVER_B = np.linspace(-3.0, 5.0, 801)  # contains x/b = 1 (flank) exactly
# Quadratic spacing: fine near the surface, where the (shallow) RS lives.
Z_OVER_B = 4.0 * np.linspace(0.0, 1.0, 200) ** 2
T0 = 20.0  # ambient temperature (deg C), fixed
DEFAULT_SWEEP_FEEDS = (0.01, 0.05, 0.10)  # mm, one flash-vs-speed line each


# ------------------------------------------------------------------ helpers
def _axes(title, xlabel, ylabel, figsize=(5.0, 3.6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(True, color=GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=INK_MUTED)
    return fig, ax


def _show(column, fig):
    fig.tight_layout()
    column.pyplot(fig)
    plt.close(fig)


def _crop(result2d, frac=0.02):
    """x/b range and max depth where the field is noticeably above ambient,
    so the heatmap isn't mostly empty far-field."""
    rise = result2d.T_field_C - result2d.T_field_C.min()
    hot = rise > frac * rise.max() if rise.max() > 0 else np.ones_like(rise, bool)
    x_hot = result2d.x_over_b[hot.any(axis=0)]
    z_hot = result2d.z_over_b[hot.any(axis=1)]
    pad = max(0.3 * (x_hot.max() - x_hot.min()), 0.5)
    return (max(x_hot.min() - pad, X_OVER_B[0]), min(x_hot.max() + pad, X_OVER_B[-1]),
            min(max(1.3 * z_hot.max(), 0.15), Z_OVER_B[-1]))


# -------------------------------------------------------------------- inputs
def _inputs():
    sb = st.sidebar
    sb.header("Inputs (Ti-6Al-4V)")
    v = sb.number_input("Cutting speed v (m/min)", 1.0, 600.0, 60.0, step=5.0)
    h = sb.number_input("Feed h (mm)", 0.005, 0.5, 0.05, step=0.01, format="%.3f")
    w = sb.number_input("Width of cut w (mm)", 0.1, 20.0, 3.0, step=0.5)
    sb.subheader("Feeds to plot (mm)")
    feeds = tuple(
        sb.number_input(f"Feed {i}", 0.005, 0.5, default, step=0.005, format="%.3f")
        for i, default in enumerate(DEFAULT_SWEEP_FEEDS, start=1))
    return v, h, w, feeds


def _values_table(rows):
    """Name / value (with units) / kind table replacing the old metric cards."""
    st.dataframe(
        [{"Name": label, "Value": value, "Type": kind} for label, value, kind in rows],
        hide_index=True, use_container_width=True)


# --------------------------------------------------------------------- plots
def _plot_shape(r):
    fig, ax = _axes(f"Normalized surface shape (Pe = {r.Pe:.2f})", "x / b", "(T − T₀) / ΔT_flash")
    ax.plot(r.x_over_b, r.shape, color=BLUE, linewidth=2)
    ax.plot(r.peak_x_over_b, 1.0, "o", color=BLUE)
    ax.set_xlim(r.x_over_b[0], r.x_over_b[-1])
    ax.set_ylim(0, 1.1)
    return fig


def _plot_convergence(r):
    fig, ax = _axes("Flash temperature convergence", "Iteration", "ΔT_flash (K)")
    it = np.arange(len(r.history))  # 0 = initial guess
    ax.plot(it, r.history, "o-", color=BLUE, linewidth=1.5, markersize=4)
    ax.axhline(r.T_flash, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def _plot_surface(r):
    fig, ax = _axes("Actual surface temperature", "x / b", "Temperature (°C)")
    ax.plot(r.x_over_b, r.T_actual_C, color=ORANGE, linewidth=2)
    ax.plot(r.peak_x_over_b, T0 + r.T_flash, "o", color=ORANGE)
    ax.set_xlim(r.x_over_b[0], r.x_over_b[-1])
    return fig


def _plot_field(r2, limits=None, title="Subsurface temperature with residual stress"):
    """Temperature field with iso-RS contours. limits = (x_lo, x_hi, z_hi) zooms
    the view (default: crop to the visibly heated region)."""
    x_lo, x_hi, z_hi = _crop(r2) if limits is None else limits
    Tc = r2.T_critical_C
    fig, ax = _axes(title, "x / b", "z / b (depth)", figsize=(6.6, 3.6))
    mesh = ax.pcolormesh(r2.x_over_b, r2.z_over_b, r2.T_field_C, cmap=TEMP_CMAP, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)
    cbar.outline.set_visible(False)

    rs = residual_stress(r2.T_field_C, Tc)
    if rs.max() > 0:
        # Yield boundary (T = T_crit), then one coloured contour per RS level.
        ax.contour(r2.x_over_b, r2.z_over_b, r2.T_field_C, levels=[Tc],
                   colors=[INK_PRIMARY], linewidths=1.2, linestyles="dashed")
        ax.plot([], [], "--", color=INK_PRIMARY, linewidth=1.2, label=f"T = T_crit ({Tc:.0f} °C)")
        levels = MaxNLocator(4).tick_values(0, rs.max())[1:-1]
        for color, level in zip(RS_COLORS, levels):
            ax.contour(r2.x_over_b, r2.z_over_b, rs, levels=[level], colors=[color], linewidths=1.3)
            ax.plot([], [], color=color, linewidth=1.3, label=f"RS = {level:.0f} MPa")
    else:
        ax.set_title(f"{title.split(' with ')[0]} (no yielding, RS = 0)",
                     color=INK_PRIMARY, fontsize=11, loc="left")
    ax.axvline(r2.flank_x_over_b, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.plot([], [], ":", color=INK_MUTED, label="Flank x/b = 1")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.3, 1.0),
              labelcolor=INK_SECONDARY)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_hi, 0)  # surface at top
    return fig


def _plot_field_zoom(r2, v, Fc, w, h):
    """Same plot as _plot_field, zoomed on the yielded region (the iso-RS lines),
    re-solved on a fine depth grid."""
    rs = residual_stress(r2.T_field_C, r2.T_critical_C)
    title = "Zoom on the iso-RS region"
    if rs.max() <= 0:
        return _plot_field(r2, title=f"{title} with RS")
    cols = r2.x_over_b[rs.any(axis=0)]
    z_hi = 1.25 * r2.z_over_b[rs.any(axis=1)].max()
    pad = 0.15 * (cols.max() - cols.min()) + 0.05
    z = np.linspace(0.0, z_hi, 120)
    rz = solve_2d(v, Fc, w, None, T0, x_over_b=X_OVER_B, z_over_b=z, h_mm=h)
    return _plot_field(rz, (cols.min() - pad, cols.max() + pad, z_hi), title)


def _plot_rs_zoom(r2, v, Fc, w, h):
    """Residual stress vs depth in µm, zoomed to the yielded layer (the
    z/b = 0-1 plot shows it as a sliver), re-solved on a fine depth grid."""
    yielded = r2.z_over_b[r2.residual_stress_MPa > 0]
    fig, ax = _axes("Residual stress near the surface (zoom)", "Depth z (µm)",
                    "Residual stress (MPa)")
    if not yielded.size:
        ax.text(0.5, 0.5, "No yielding (RS = 0)", transform=ax.transAxes, ha="center",
                color=INK_MUTED)
        return fig
    z_max = 1.25 * yielded.max()
    z = np.linspace(0.0, z_max, 200)
    rz = solve_2d(v, Fc, w, None, T0, x_over_b=X_OVER_B, z_over_b=z, h_mm=h)
    ax.plot(z * rz.b_um, rz.residual_stress_MPa, color=ORANGE, linewidth=2)
    ax.set_xlim(0.0, z_max * rz.b_um)
    ax.set_ylim(bottom=0.0)
    return fig


@st.cache_data(show_spinner=False)
def _critical_speed(h, w):
    return critical_speed(h, w, None, T0)


@st.cache_data(show_spinner="Computing machining sweeps…")
def _machining(w, feeds):
    v = 0.1 * np.arange(1, 51)  # 0.1-5 m/s (6-300 m/min), as the workbook sweeps
    T_lines = [flash_vs_speed(v, hf, w, None, T0) for hf in feeds]
    hc = np.linspace(0.01, 0.12, 23)
    F = force_vs_feed(hc, w)
    vc = np.array([critical_speed(x, w, None, T0) for x in hc]) * 60.0  # m/min
    return v, T_lines, hc, F, vc


def _plot_machining(w, feeds):
    """Force vs feed, flash T vs speed (workbook axes: 0-300 m/min)
    and critical speed vs feed (m/min, h from 0)."""
    v, T_lines, hc, F, vc = _machining(w, feeds)

    fig_f, ax = _axes(f"Cutting force vs feed (Kienzle fit, w = {w:g} mm)", "Feed h (mm)", "Cutting force Fc (N)")
    C, n = KIENZLE_TI64
    ax.plot(hc, F, color=BLUE, linewidth=2, label=f"Fc = {C:.4g}·h^{n + 1.0:.4g}·w")
    ax.set_xlim(0.0, 0.125)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)

    fig_t, ax = _axes("Peak temperature vs cutting speed", "Cutting speed (m/min)", "Peak temperature (°C)")
    for color, hf, T in zip(FEED_COLORS, feeds, T_lines):
        ax.plot(v * 60.0, T, color=color, linewidth=2, label=f"h = {hf:g} mm")
    ax.plot(DATA_V_M_MIN, DATA_T_C, "o", color=INK_PRIMARY, markersize=5,
            label=f"Data (h = {DATA_H:g} mm)")
    ax.set_xlim(0.0, 300.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)

    fig_c, ax = _axes("Critical speed for tensile surface RS", "Feed h (mm)", "Critical speed (m/min)")
    ok = np.isfinite(vc)
    fit = None
    if ok.sum() >= 2:
        a, n = critical_speed_fit(hc, vc)
        fit = (a, n)
        ax.plot(hc[ok], a * hc[ok] ** n, color=INK_MUTED, linestyle="--",
                label=f"v = {a:.3g}·h^{n:.3f}")
    ax.plot(hc[ok], vc[ok], "o", color=ORANGE, label="model")
    ax.set_xlim(0.0, 0.125)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=8)
    return (fig_f, fig_t, fig_c), fit


# ---------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Ti-6Al-4V Cutting Thermal Model", layout="wide")
    v, h, w, feeds = _inputs()
    Fc = float(kienzle_force(h, w))
    r = solve(v, Fc, w, None, T0, h_mm=h)
    r2 = solve_2d(v, Fc, w, None, T0, x_over_b=X_OVER_B, z_over_b=Z_OVER_B, h_mm=h)
    b = r.b_um  # contact half-width from the shear-plane model
    figs, fit = _plot_machining(w, feeds)
    v_crit = _critical_speed(h, w)

    st.title("Ti-6Al-4V Cutting Thermal Model")
    if not r.converged:
        st.error(f"Flash temperature did not converge in {r.iterations} iterations.")

    yielded = r2.z_over_b[r2.residual_stress_MPa > 0]
    _values_table([
        # -- inputs (sidebar) --
        ("Cutting speed v", f"{v:.1f} m/min", "input"),
        ("Feed h", f"{h:.3f} mm", "input"),
        ("Width of cut w", f"{w:.2f} mm", "input"),
        ("Sweep feeds (machining plots)", ", ".join(f"{x:g}" for x in feeds) + " mm", "input"),
        # -- hardcoded constants --
        ("Ambient temperature T₀", f"{T0:.1f} °C", "hardcoded"),
        ("Density ρ", f"{TI64['rho']:g} kg/m³", "hardcoded"),
        ("Melting point T_melt", f"{TI64['T_melt']:g} °C", "hardcoded"),
        ("Poisson's ratio ν", f"{TI64['poisson']:g}", "hardcoded"),
        # -- fitted to reference/measured data --
        ("Kienzle coefficients (C, n)", f"{KIENZLE_TI64[0]:g}, {KIENZLE_TI64[1]:g}", "fit"),
        ("RS fit (slope, intercept)", f"{TI64['rs_fit'][0]:g}, {TI64['rs_fit'][1]:g}", "fit"),
        ("Flash-T calibration (a_low, a_high)",
         f"{CALIBRATION['a_low']:.4f}, {CALIBRATION['a_high']:.4f}", "fit"),
        ("Critical speed fit (a, n)",
         f"v = {fit[0]:.4g}·h^{fit[1]:.3f}" if fit else "n/a", "fit"),
        # -- calculated from the above each run --
        ("Cutting force Fc (Kienzle)", f"{Fc:.1f} N", "calculated"),
        ("Contact half-width b (shear-plane)", f"{b:.1f} µm", "calculated"),
        ("Heat partition R (shear-plane)", f"{r.R:.2f}", "calculated"),
        ("Peak surface temperature", f"{T0 + r.T_flash:.1f} °C", "calculated"),
        ("Flash temperature rise ΔT_flash", f"{r.T_flash:.1f} K", "calculated"),
        ("Peclet number Pe", f"{r.Pe:.3f}", "calculated"),
        ("Conductivity k at convergence", f"{r.k:.2f} W/(m·K)", "calculated"),
        ("Specific heat cp at convergence", f"{r.cp:.1f} J/(kg·K)", "calculated"),
        ("Initial guess (T₀ + T_melt)/2", f"{r.history[0]:.0f} °C", "calculated"),
        ("Iterations to converge", f"{r.iterations}", "calculated"),
        ("Critical (yield) temperature T_crit", f"{r2.T_critical_C:.1f} °C", "calculated"),
        ("Surface RS at flank", f"{r2.residual_stress_MPa[0]:.1f} MPa", "calculated"),
        ("Yielded depth", f"{yielded.max() * b:.1f} µm" if yielded.size else "0 µm", "calculated"),
        ("Critical speed at this h",
         f"{v_crit * 60:.1f} m/min" if np.isfinite(v_crit) else "> 600 m/min", "calculated"),
    ])

    st.subheader("Surface (1D)")
    c = st.columns(3)
    _show(c[0], _plot_shape(r))
    _show(c[1], _plot_convergence(r))
    _show(c[2], _plot_surface(r))

    st.subheader("Subsurface (2D)")
    c = st.columns([1.3, 1, 1])
    _show(c[0], _plot_field(r2))
    _show(c[1], _plot_field_zoom(r2, v, Fc, w, h))
    _show(c[2], _plot_rs_zoom(r2, v, Fc, w, h))

    st.subheader("Machining sweeps")
    c = st.columns(3)
    for col, fig in zip(c, figs):
        _show(col, fig)


if __name__ == "__main__":
    main()
