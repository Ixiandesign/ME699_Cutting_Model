"""Streamlit web UI for the Peclet-normalized cutting thermal model (Ti-6Al-4V).

One page: process inputs in the sidebar, key results as metrics, then
  1D:  normalized (Peclet) surface shape, flash-temperature chain convergence,
       actual surface temperature
  2D:  subsurface temperature field with residual-stress contours overlaid,
       residual stress vs depth at the flank, thermoelastic stress vs yield
  Machining: force vs feed, Pe vs speed, flash T vs speed (with measured data),
       critical speed vs feed

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

from model import (RHO, SHEET_DATA, SHEET_DATA_H, SHEET_FEEDS, T_CRIT_SPEED, TI64,
                   Constants, critical_speed_fit, cutting_force, feed_sweep,
                   initial_guess, residual_stress, solve, thermoelastic_stress)

# --- Palette (dataviz skill reference instance, light mode) -----------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1a9850"
SERIES = [BLUE, ORANGE, GREEN, "#8e44ad", "#c0392b", "#7f8c8d"]  # categorical
RS_COLORS = [BLUE, GREEN, "#8e44ad", "#0f4c81"]  # one per RS contour level

# Sequential light->dark ramp in the orange family for the temperature field.
TEMP_CMAP = LinearSegmentedColormap.from_list("cutting_temp", [SURFACE, ORANGE, "#7a2e0e"])


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


def _crop(s, frac=0.02):
    """x/b range and max depth where the field is noticeably above ambient,
    so the heatmap isn't mostly empty far-field."""
    rise = s.T_field_C - s.T_field_C.min()
    hot = rise > frac * rise.max() if rise.max() > 0 else np.ones_like(rise, bool)
    x_hot = s.x_over_b[hot.any(axis=0)]
    z_hot = s.z_over_b[hot.any(axis=1)]
    pad = max(0.3 * (x_hot.max() - x_hot.min()), 0.5)
    return (max(x_hot.min() - pad, s.x_over_b[0]), min(x_hot.max() + pad, s.x_over_b[-1]),
            min(max(1.3 * z_hot.max(), 0.15), s.z_over_b[-1]))


# -------------------------------------------------------------------- inputs
def _inputs():
    sb = st.sidebar
    sb.header("Inputs (Ti-6Al-4V)")
    # Ranges are the sheet's (v 0.1-5 m/s, h up to 0.12 mm); the low-Pe polynomial
    # C4(Pe) diverges beyond them.
    v = sb.number_input("Cutting speed v (m/min)", 6.0, 300.0, 120.0, step=5.0)
    h = sb.number_input("Feed h (mm)", 0.01, 0.12, 0.05, step=0.005, format="%.3f")
    w = sb.number_input("Width of cut w (mm)", 0.1, 20.0, 3.0, step=0.5)
    with sb.expander("Sheet constants"):
        shear = st.number_input("Shear angle (°)", 5.0, 80.0, 35.0, step=1.0)
        rake = st.number_input("Rake angle (°)", -30.0, 30.0, 0.0, step=1.0)
        clearance = st.number_input("Clearance angle (°)", 1.0, 30.0, 5.0, step=1.0)
        edge = st.number_input("Cutting edge radius (µm)", 1.0, 100.0, 10.0, step=1.0)
        a_low = st.number_input("Calibration factor, Pe < 5", 0.1, 2.0, 0.95, step=0.05)
        a_high = st.number_input("Calibration factor, Pe > 5", 0.1, 2.0, 0.90, step=0.05)
        T_init = st.number_input("Initial temperature guess (°C)", 20.0, 1660.0,
                                 initial_guess(), step=10.0)
        T_crit_speed = st.number_input("Tc for critical speed (°C)", 100.0, 1000.0,
                                       T_CRIT_SPEED, step=10.0)
    return (v, h, w, Constants(shear, rake, clearance, edge, a_low, a_high),
            T_init, T_crit_speed)


def _values_row(items):
    """Secondary values under the main metrics, in a smaller font."""
    cells = "".join(f'<div><div class="lbl">{k}</div><div class="val">{v}</div></div>'
                    for k, v in items)
    st.markdown(
        "<style>.vals{display:flex;flex-wrap:wrap;gap:0.6rem 2rem;margin:-0.5rem 0 0.5rem}"
        ".vals .lbl{font-size:0.8rem;opacity:0.75}.vals .val{font-size:1.15rem}</style>"
        f'<div class="vals">{cells}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------- plots
def _plot_shape(s):
    fig, ax = _axes(f"Normalized surface shape (Pe = {s.row.Pe:.2f})", "x / b", "(T − T₀) / ΔT_flash")
    ax.plot(s.x_over_b, s.shape, color=BLUE, linewidth=2)
    ax.plot(s.peak_x_over_b, 1.0, "o", color=BLUE)
    ax.set_xlim(s.x_over_b[0], s.x_over_b[-1])
    ax.set_ylim(0, 1.1)
    return fig


def _plot_convergence(row):
    fig, ax = _axes("Flash temperature convergence", "Pass", "Flash temperature (°C)")
    passes = np.arange(1, 6)
    used_high = row.Pe > 5.0
    ax.plot(passes, row.low, "o-", color=BLUE, linewidth=1.5, markersize=4,
            label="Pe < 5 chain" + ("" if used_high else " (used)"))
    ax.plot(passes, row.high, "o-", color=ORANGE, linewidth=1.5, markersize=4,
            label="Pe > 5 chain" + (" (used)" if used_high else ""))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8)
    return fig


def _plot_surface(s):
    fig, ax = _axes("Actual surface temperature", "x / b", "Temperature (°C)")
    ax.plot(s.x_over_b, s.T_surface_C, color=ORANGE, linewidth=2)
    ax.plot(s.peak_x_over_b, s.peak_surface_C, "o", color=ORANGE)
    ax.set_xlim(s.x_over_b[0], s.x_over_b[-1])
    return fig


def _plot_field(s):
    x_lo, x_hi, z_hi = _crop(s)
    Tc = s.T_critical_C
    fig, ax = _axes("Subsurface temperature with residual stress", "x / b", "z / b (depth)",
                    figsize=(6.6, 3.6))
    mesh = ax.pcolormesh(s.x_over_b, s.z_over_b, s.T_field_C, cmap=TEMP_CMAP, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Temperature (°C)", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)
    cbar.outline.set_visible(False)

    rs = residual_stress(s.T_field_C, Tc)
    if rs.max() > 0:
        # Yield boundary (T = T_crit), then one coloured contour per RS level.
        ax.contour(s.x_over_b, s.z_over_b, s.T_field_C, levels=[Tc],
                   colors=[INK_PRIMARY], linewidths=1.2, linestyles="dashed")
        ax.plot([], [], "--", color=INK_PRIMARY, linewidth=1.2, label=f"T = T_crit ({Tc:.0f} °C)")
        levels = MaxNLocator(4).tick_values(0, rs.max())[1:-1]
        for color, level in zip(RS_COLORS, levels):
            ax.contour(s.x_over_b, s.z_over_b, rs, levels=[level], colors=[color], linewidths=1.3)
            ax.plot([], [], color=color, linewidth=1.3, label=f"RS = {level:.0f} MPa")
    else:
        ax.set_title("Subsurface temperature (no yielding, RS = 0)",
                     color=INK_PRIMARY, fontsize=11, loc="left")
    ax.axvline(s.flank_x_over_b, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.plot([], [], ":", color=INK_MUTED, label=f"Flank x/b = {s.flank_x_over_b:g}")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.3, 1.0),
              labelcolor=INK_SECONDARY)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_hi, 0)  # surface at top
    return fig


def _plot_rs(s):
    fig, ax = _axes(f"Residual stress at flank (x/b = {s.flank_x_over_b:g})",
                    "z / b (depth)", "Residual stress (MPa)")
    ax.axhline(0.0, color=AXIS_COLOR, linewidth=1)
    ax.plot(s.z_over_b, s.residual_stress_MPa, color=ORANGE, linewidth=2)
    yielded = s.z_over_b[s.residual_stress_MPa > 0]
    ax.set_xlim(0, 1.5 * yielded.max() if yielded.size else _crop(s)[2])
    return fig


def _plot_stress_vs_T(Tc):
    T = np.arange(20.0, 1001.0)
    yield_strength, thermal = TI64["sigma_y"](T), thermoelastic_stress(T)
    fig, ax = _axes("Thermoelastic stress vs yield strength", "Temperature (°C)", "Stress (MPa)")
    ax.plot(T, yield_strength, color=BLUE, linewidth=2, label="Yield strength")
    ax.plot(T, thermal, color=ORANGE, linewidth=2, label="Thermoelastic stress")
    ax.plot(T, np.maximum(thermal - yield_strength, 0.0), color=GREEN, linewidth=2,
            label="Excess over yield")
    ax.axvline(Tc, color=INK_PRIMARY, linestyle="--", linewidth=1, label=f"T_crit = {Tc:.0f} °C")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return fig


def _plot_force(w, h_now):
    h = np.linspace(0.01, 0.12, 50)
    fig, ax = _axes(f"Cutting force vs feed (w = {w:g} mm)", "Feed h (mm)", "Cutting force Fc (N)")
    ax.plot(h, cutting_force(h, w), color=BLUE, linewidth=2)
    ax.plot(h_now, cutting_force(h_now, w), "o", color=BLUE)
    return fig


def _plot_pe_speed(sweeps):
    fig, ax = _axes("Peclet number vs cutting speed", "Cutting speed (m/s)", "Pe (average)")
    for color, fs in zip(SERIES, sweeps):
        ax.plot(fs.v_m_min / 60.0, fs.Pe, color=color, linewidth=2, label=f"h = {fs.h_mm:g} mm")
    ax.legend(frameon=False, fontsize=8)
    return fig


def _plot_T_speed(sweeps):
    fig, ax = _axes("Flash temperature vs cutting speed", "Cutting speed (m/min)", "Temperature (°C)")
    for color, fs in zip(SERIES, sweeps):
        ax.plot(fs.v_m_min, fs.T_flash, color=color, linewidth=2, label=f"h = {fs.h_mm:g} mm")
    v, T = zip(*SHEET_DATA)
    ax.plot(v, T, "o", color=INK_PRIMARY, markersize=5, label=f"Data (h = {SHEET_DATA_H:g} mm)")
    ax.legend(frameon=False, fontsize=8)
    return fig


def _plot_vcrit(sweeps, Tc):
    h = np.array([fs.h_mm for fs in sweeps])
    fig, ax = _axes(f"Critical speed for T = {Tc:g} °C", "Feed h (mm)", "Critical speed (m/min)")
    ax.plot(h, [fs.v_crit_power for fs in sweeps], "o", color=BLUE, label="Power-law fit")
    ax.plot(h, [fs.v_crit_log for fs in sweeps], "o", color=ORANGE, label="Log fit")
    ax.plot(h, [fs.v_crit_avg for fs in sweeps], "o", color=GREEN, label="Average")
    a, n = critical_speed_fit(h, [fs.v_crit_avg for fs in sweeps])
    hh = np.linspace(h.min(), h.max(), 100)
    ax.plot(hh, a * hh**n, "--", color=INK_MUTED, label=f"Average fit: v = {a:.3g}·h^{n:.3f}")
    ax.legend(frameon=False, fontsize=8)
    return fig


# ---------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Ti-6Al-4V Cutting Thermal Model", layout="wide")
    v, h, w, consts, T_init, T_crit_speed = _inputs()
    s = solve(v, h, w, consts, T_init)
    r = s.row
    feeds = sorted({*SHEET_FEEDS, round(h, 6)})
    sweeps = [feed_sweep(x, w, T_init, consts, T_crit_speed) for x in feeds]
    here = sweeps[feeds.index(round(h, 6))]

    st.title("Ti-6Al-4V Cutting Thermal Model")

    m = st.columns(6)
    m[0].metric("Peak surface T", f"{s.peak_surface_C:.1f} °C")
    m[1].metric("Flash temperature ΔT", f"{r.T_flash:.1f} °C")
    m[2].metric("Peclet number", f"{r.Pe:.3f}")
    m[3].metric("Critical T (yield)", f"{s.T_critical_C:.1f} °C")
    m[4].metric("Surface RS at flank", f"{s.residual_stress_MPa[0]:.1f} MPa")
    yielded = s.z_over_b[s.residual_stress_MPa > 0]
    m[5].metric("Yielded depth", f"{yielded.max() * r.b * 1e6:.1f} µm" if yielded.size else "0 µm")
    _values_row([
        ("Cutting force Fc", f"{r.Fc:.1f} N"),
        ("Contact half-width b", f"{r.b * 1e6:.1f} µm"),
        ("Contact length 2b", f"{2 * r.b * 1e6:.1f} µm"),
        ("Heat partition R", f"{r.partition:.3f}"),
        ("Pe (shear plane)", f"{r.Pe_shear:.3f}"),
        ("k at initial T", f"{r.k:.2f} W/(m·K)"),
        ("cp at initial T", f"{r.cp:.1f} J/(kg·K)"),
        ("Density ρ", f"{RHO:g} kg/m³"),
        ("Initial guess", f"{r.T_initial:.0f} °C"),
        ("Flash chain used", "Pe > 5" if r.Pe > 5 else "Pe < 5"),
        ("Critical speed, power law", f"{here.v_crit_power:.1f} m/min"),
        ("Critical speed, log fit", f"{here.v_crit_log:.1f} m/min"),
        ("Critical speed, average", f"{here.v_crit_avg:.1f} m/min"),
    ])

    st.subheader("Surface (1D)")
    c = st.columns(3)
    _show(c[0], _plot_shape(s))
    _show(c[1], _plot_convergence(r))
    _show(c[2], _plot_surface(s))

    st.subheader("Subsurface (2D)")
    c = st.columns([1.4, 1, 1])
    _show(c[0], _plot_field(s))
    _show(c[1], _plot_rs(s))
    _show(c[2], _plot_stress_vs_T(s.T_critical_C))

    st.subheader("Machining sweeps")
    c = st.columns(4)
    _show(c[0], _plot_force(w, h))
    _show(c[1], _plot_pe_speed(sweeps))
    _show(c[2], _plot_T_speed(sweeps))
    _show(c[3], _plot_vcrit(sweeps, T_crit_speed))


if __name__ == "__main__":
    main()
