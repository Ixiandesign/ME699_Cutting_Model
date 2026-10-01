"""Streamlit web UI for the Peclet-normalized cutting thermal model (Ti-6Al-4V).

One page: process inputs in the sidebar, key results as metrics, then
  1D:  normalized (Peclet) surface shape, flash-temperature convergence,
       actual surface temperature
  2D:  subsurface temperature field with residual-stress contours overlaid,
       residual stress vs depth at the flank (x/b = 1)
  Machining: force vs feed, flash temperature vs speed, critical speed vs feed

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

from model import (TI64, critical_speed, critical_speed_fit, flash_vs_speed,
                   force_vs_feed, kienzle_force, residual_stress, solve, solve_2d)

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

X_OVER_B = np.linspace(-3.0, 5.0, 300)
# Quadratic spacing: fine near the surface, where the (shallow) RS lives.
Z_OVER_B = 4.0 * np.linspace(0.0, 1.0, 200) ** 2
SWEEP_FEEDS = (0.01, 0.05, 0.10)  # mm, one flash-vs-speed line each


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
    b = sb.number_input("Contact half-width b (µm)", 5.0, 1000.0, 200.0, step=10.0)
    w = sb.number_input("Width of cut w (mm)", 0.1, 20.0, 3.0, step=0.5)
    T0 = sb.number_input("Ambient temperature T₀ (°C)", -50.0, 200.0, 20.0, step=5.0)
    return v, h, b, w, T0


def _values_row(items):
    """Secondary values under the main metrics, in a smaller font."""
    cells = "".join(f'<div><div class="lbl">{k}</div><div class="val">{v}</div></div>'
                    for k, v in items)
    st.markdown(
        "<style>.vals{display:flex;flex-wrap:wrap;gap:0.6rem 2rem;margin:-0.5rem 0 0.5rem}"
        ".vals .lbl{font-size:0.8rem;opacity:0.75}.vals .val{font-size:1.15rem}</style>"
        f'<div class="vals">{cells}</div>', unsafe_allow_html=True)


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


def _plot_surface(r, T0):
    fig, ax = _axes("Actual surface temperature", "x / b", "Temperature (°C)")
    ax.plot(r.x_over_b, r.T_actual_C, color=ORANGE, linewidth=2)
    ax.plot(r.peak_x_over_b, T0 + r.T_flash, "o", color=ORANGE)
    ax.set_xlim(r.x_over_b[0], r.x_over_b[-1])
    return fig


def _plot_field(r2):
    x_lo, x_hi, z_hi = _crop(r2)
    Tc = r2.T_critical_C
    fig, ax = _axes("Subsurface temperature with residual stress", "x / b", "z / b (depth)",
                    figsize=(6.6, 3.6))
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
        ax.set_title("Subsurface temperature (no yielding, RS = 0)",
                     color=INK_PRIMARY, fontsize=11, loc="left")
    ax.axvline(r2.flank_x_over_b, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.plot([], [], ":", color=INK_MUTED, label="Flank x/b = 1")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.3, 1.0),
              labelcolor=INK_SECONDARY)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_hi, 0)  # surface at top
    return fig


def _plot_rs(r2):
    fig, ax = _axes("Residual stress at flank (x/b = 1)", "z / b (depth)", "Residual stress (MPa)")
    ax.axhline(0.0, color=AXIS_COLOR, linewidth=1)
    ax.plot(r2.z_over_b, r2.residual_stress_MPa, color=ORANGE, linewidth=2)
    yielded = r2.z_over_b[r2.residual_stress_MPa > 0]
    ax.set_xlim(0, 1.5 * yielded.max() if yielded.size else _crop(r2)[2])
    return fig


@st.cache_data(show_spinner=False)
def _critical_speed(h, w, b, T0):
    return critical_speed(h, w, b, T0)


@st.cache_data(show_spinner="Computing machining sweeps…")
def _machining(w, b, T0):
    h = np.linspace(0.01, 0.10, 50)
    v = np.geomspace(0.1, 10.0, 40)
    T_lines = [flash_vs_speed(v, hf, w, b, T0) for hf in SWEEP_FEEDS]
    hc = np.linspace(0.01, 0.10, 19)
    vc = np.array([critical_speed(x, w, b, T0) for x in hc])
    return h, force_vs_feed(h, w), v, T_lines, hc, vc


def _plot_machining(w, b, T0):
    h, F, v, T_lines, hc, vc = _machining(w, b, T0)

    fig_f, ax = _axes(f"Cutting force vs feed (w = {w:g} mm)", "Feed h (mm)", "Cutting force Fc (N)")
    ax.plot(h, F, color=BLUE, linewidth=2)

    fig_t, ax = _axes("Peak temperature vs cutting speed", "Cutting speed (m/s)", "Peak temperature (°C)")
    ax.set_xscale("log")
    for color, hf, T in zip(FEED_COLORS, SWEEP_FEEDS, T_lines):
        ax.plot(v, T, color=color, linewidth=2, label=f"h = {hf:g} mm")
    ax.legend(frameon=False)

    fig_c, ax = _axes("Critical speed for tensile surface RS", "Feed h (mm)", "Critical speed (m/s)")
    ok = np.isfinite(vc)
    fit = None
    if ok.sum() >= 2:
        a, n = critical_speed_fit(hc, vc)
        fit = (a, n)
        ax.plot(hc[ok], a * hc[ok] ** n, color=INK_MUTED, linestyle="--",
                label=f"v = {a:.3g}·h^{n:.3f}")
    ax.plot(hc[ok], vc[ok], "o", color=ORANGE, label="model")
    ax.legend(frameon=False)
    return (fig_f, fig_t, fig_c), fit


# ---------------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Ti-6Al-4V Cutting Thermal Model", layout="wide")
    v, h, b, w, T0 = _inputs()
    Fc = float(kienzle_force(h, w))
    r = solve(v, Fc, w, b, T0)
    r2 = solve_2d(v, Fc, w, b, T0, x_over_b=X_OVER_B, z_over_b=Z_OVER_B)
    figs, fit = _plot_machining(w, b, T0)
    v_crit = _critical_speed(h, w, b, T0)

    st.title("Ti-6Al-4V Cutting Thermal Model")
    if not r.converged:
        st.error(f"Flash temperature did not converge in {r.iterations} iterations.")

    m = st.columns(6)
    m[0].metric("Peak surface T", f"{T0 + r.T_flash:.1f} °C")
    m[1].metric("Flash rise ΔT", f"{r.T_flash:.1f} K")
    m[2].metric("Peclet number", f"{r.Pe:.3f}")
    m[3].metric("Critical T (yield)", f"{r2.T_critical_C:.1f} °C")
    m[4].metric("Surface RS at flank", f"{r2.residual_stress_MPa[0]:.1f} MPa")
    yielded = r2.z_over_b[r2.residual_stress_MPa > 0]
    m[5].metric("Yielded depth", f"{yielded.max() * b:.1f} µm" if yielded.size else "0 µm")
    _values_row([
        ("Cutting force Fc (Kienzle)", f"{Fc:.1f} N"),
        ("Contact length 2b", f"{2 * b:g} µm"),
        ("Density ρ", f"{TI64['rho']:g} kg/m³"),
        ("k at convergence", f"{r.k:.2f} W/(m·K)"),
        ("cp at convergence", f"{r.cp:.1f} J/(kg·K)"),
        ("Initial guess (T₀ + T_melt)/2", f"{r.history[0]:.0f} °C"),
        ("Iterations", f"{r.iterations}"),
        ("Critical speed at this h", f"{v_crit:.3f} m/s" if np.isfinite(v_crit) else "> 10 m/s"),
        ("Critical speed fit", f"v = {fit[0]:.4g}·h^{fit[1]:.3f}" if fit else "n/a"),
    ])

    st.subheader("Surface (1D)")
    c = st.columns(3)
    _show(c[0], _plot_shape(r))
    _show(c[1], _plot_convergence(r))
    _show(c[2], _plot_surface(r, T0))

    st.subheader("Subsurface (2D)")
    c = st.columns([1.3, 1])
    _show(c[0], _plot_field(r2))
    _show(c[1], _plot_rs(r2))

    st.subheader("Machining sweeps")
    c = st.columns(3)
    for col, fig in zip(c, figs):
        _show(col, fig)


if __name__ == "__main__":
    main()
