"""Streamlit web UI for the Peclet-normalized cutting thermal model.

Runs entirely through the browser (no Tk/desktop GUI backend needed --
matplotlib is forced onto the non-interactive Agg backend below and
figures are handed to Streamlit as static images via st.pyplot).

Two views, picked via tabs:
- **1D view**: two single-axis panels side by side, sharing the x/b axis
  -- the normalized surface shape (peak always 1.0) and that same curve
  scaled by the peak ("flash") temperature into actual degrees C. A dual
  y-axis was deliberately avoided (see the dataviz skill's "one axis"
  rule) since it invites misreading a coincidental crossing/slope match
  between two differently-scaled series as meaningful.
- **2D view**: the subsurface temperature field T(x/b, z/b) as a
  cutaway heatmap (z=0, the surface, at top), and the residual-stress-
  vs-depth profile at the flank/tool-exit location (x/b=1).

Run with: uv run streamlit run app.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from cutting_model.forces import estimate_cutting_force
from cutting_model.materials import AA6061, AA7050, DEFAULTS, SS304, TI64
from cutting_model.model import CuttingThermalModel

# --- Palette (dataviz skill reference instance, light mode) -----------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SERIES_NORM = "#2a78d6"  # categorical slot 1 (blue)
SERIES_ACTUAL = "#eb6834"  # categorical slot 2 (orange)

# Sequential (single-hue, light->dark) ramp for the 2D temperature
# field, in the same orange family as SERIES_ACTUAL.
TEMP_CMAP = LinearSegmentedColormap.from_list(
    "cutting_temp", [SURFACE, SERIES_ACTUAL, "#7a2e0e"]
)

MATERIAL_ORDER = [TI64, AA7050, SS304, AA6061]
MATERIAL_NAMES = [m.name for m in MATERIAL_ORDER]
MATERIAL_BY_NAME = {m.name: m for m in MATERIAL_ORDER}

APP_X_OVER_B = np.linspace(-3.0, 5.0, 300)
APP_Z_OVER_B = np.linspace(0.0, 4.0, 150)

SLIDER_SPECS = [
    ("v_m_min", "Cutting speed vc (m/min)", 1.0, 300.0),
    ("Fc_N", "Cutting force Fc (N)", 10.0, 1000.0),
    ("b_um", "Contact half-width b (µm)", 10.0, 500.0),
    ("w_mm", "Width of cut w (mm)", 1.0, 10.0),
]

st.set_page_config(page_title="Cutting Thermal Model", layout="wide")


def _style_axes(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(True, color=GRIDLINE, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=INK_MUTED)


def _new_figure():
    fig, ax = plt.subplots(figsize=(6.0, 4.5), facecolor=SURFACE)
    return fig, ax


def _apply_material_defaults():
    """on_change callback for the material radio: reset the process
    sliders to that material's defaults before the widgets re-render."""
    defaults = DEFAULTS[st.session_state["material"]]
    st.session_state["v_m_min"] = defaults["v_m_min"]
    st.session_state["Fc_N"] = defaults["Fc_N"]
    st.session_state["b_um"] = defaults["b_um"]
    st.session_state["w_mm"] = defaults["w_mm"]


def _init_state():
    if "material" in st.session_state:
        return
    st.session_state["material"] = TI64.name
    defaults = DEFAULTS[TI64.name]
    st.session_state["v_m_min"] = defaults["v_m_min"]
    st.session_state["Fc_N"] = defaults["Fc_N"]
    st.session_state["b_um"] = defaults["b_um"]
    st.session_state["w_mm"] = defaults["w_mm"]
    st.session_state["h_mm"] = 0.10
    st.session_state["use_kienzle"] = False


def _build_sidebar():
    st.sidebar.radio(
        "Material", MATERIAL_NAMES, key="material", on_change=_apply_material_defaults
    )
    for key, label, lo, hi in SLIDER_SPECS:
        st.sidebar.slider(label, lo, hi, key=key)
    st.sidebar.checkbox(
        "Estimate Fc from feed h (Kienzle fit)", key="use_kienzle"
    )
    st.sidebar.slider(
        "Feed h (mm) [Kienzle, Ti-6Al-4V only]", 0.01, 0.30, key="h_mm"
    )


def _effective_force(material):
    if st.session_state["use_kienzle"] and material is TI64:
        return estimate_cutting_force(material, st.session_state["h_mm"], st.session_state["w_mm"])
    return st.session_state["Fc_N"]


def _draw_1d(result):
    fig_norm, ax_norm = _new_figure()
    _style_axes(ax_norm, "Normalized surface shape (peak = 1)", "T / T_peak")
    ax_norm.set_xlabel("x / b", color=INK_SECONDARY)
    ax_norm.plot(result.x_over_b, result.shape, color=SERIES_NORM, linewidth=2)
    ax_norm.plot(
        [result.peak_x_over_b],
        [1.0],
        marker="o",
        color=SERIES_NORM,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=5,
    )
    ax_norm.annotate(
        f"1.00 @ x/b={result.peak_x_over_b:.2f}",
        xy=(result.peak_x_over_b, 1.0),
        xytext=(8, 6),
        textcoords="offset points",
        color=INK_PRIMARY,
        fontsize=10,
    )
    ax_norm.set_xlim(result.x_over_b.min(), result.x_over_b.max())
    ax_norm.set_ylim(0, 1.15)

    fig_actual, ax_actual = _new_figure()
    _style_axes(ax_actual, "Actual surface temperature", "Temperature (°C)")
    ax_actual.set_xlabel("x / b", color=INK_SECONDARY)
    T_actual_max = result.T_actual_C.max()
    ax_actual.plot(result.x_over_b, result.T_actual_C, color=SERIES_ACTUAL, linewidth=2)
    ax_actual.plot(
        [result.peak_x_over_b],
        [T_actual_max],
        marker="o",
        color=SERIES_ACTUAL,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=5,
    )
    ax_actual.annotate(
        f"{T_actual_max:.1f}°C @ x/b={result.peak_x_over_b:.2f}",
        xy=(result.peak_x_over_b, T_actual_max),
        xytext=(8, 6),
        textcoords="offset points",
        color=INK_PRIMARY,
        fontsize=10,
    )
    ax_actual.set_xlim(result.x_over_b.min(), result.x_over_b.max())
    y_lo = min(20.0, result.T_actual_C.min()) * 0.95
    ax_actual.set_ylim(y_lo, T_actual_max * 1.15)

    return fig_norm, fig_actual


def _draw_2d(result2d):
    fig_field, ax_field = _new_figure()
    _style_axes(
        ax_field, "Subsurface temperature field (flank cutaway)", "z / b (depth)"
    )
    ax_field.set_xlabel("x / b", color=INK_SECONDARY)
    ax_field.invert_yaxis()  # surface (z=0) at top, like a cutaway
    mesh = ax_field.pcolormesh(
        result2d.x_over_b,
        result2d.z_over_b,
        result2d.T_field_C,
        cmap=TEMP_CMAP,
        shading="auto",
    )
    colorbar = fig_field.colorbar(mesh, ax=ax_field, pad=0.02)
    colorbar.ax.tick_params(colors=INK_MUTED)
    colorbar.set_label("Temperature (°C)", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)

    T_crit = result2d.T_critical_C
    field_max = result2d.T_field_C.max()
    if np.isfinite(T_crit) and field_max > T_crit > result2d.T_field_C.min():
        ax_field.contour(
            result2d.x_over_b,
            result2d.z_over_b,
            result2d.T_field_C,
            levels=[T_crit],
            colors=[INK_PRIMARY],
            linewidths=1.0,
            linestyles="dashed",
        )
    ax_field.axvline(
        result2d.flank_x_over_b, color=INK_PRIMARY, linewidth=1.0, linestyle=":"
    )

    fig_rs, ax_rs = _new_figure()
    _style_axes(
        ax_rs, "Residual stress near flank (x/b ≈ 1)", "Residual stress (MPa)"
    )
    ax_rs.set_xlabel("z / b (depth)", color=INK_SECONDARY)
    ax_rs.axhline(0.0, color=AXIS_COLOR, linewidth=1.0)
    rs = result2d.residual_stress_MPa
    ax_rs.plot(result2d.z_over_b, rs, color=SERIES_ACTUAL, linewidth=2)
    ax_rs.set_xlim(result2d.z_over_b.min(), result2d.z_over_b.max())
    rs_max = max(rs.max(), 0.0)
    rs_min = min(rs.min(), 0.0)
    pad = max(rs_max - rs_min, 1.0) * 0.15
    ax_rs.set_ylim(rs_min - pad, rs_max + pad)

    if np.isfinite(T_crit) and np.any(result2d.T_flank_profile_C > T_crit):
        affected = result2d.z_over_b[result2d.T_flank_profile_C > T_crit]
        depth = affected.max()
        ax_rs.axvline(depth, color=INK_MUTED, linewidth=1.0, linestyle="--")
        ax_rs.annotate(
            f"yielded to z/b={depth:.2f}",
            xy=(depth, rs_max + pad * 0.5),
            xytext=(6, 6),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8,
        )

    return fig_field, fig_rs


def main():
    _init_state()
    _build_sidebar()

    material = MATERIAL_BY_NAME[st.session_state["material"]]
    Fc_N = _effective_force(material)
    model = CuttingThermalModel(
        material=material,
        v_m_min=st.session_state["v_m_min"],
        Fc_N=Fc_N,
        w_mm=st.session_state["w_mm"],
        b_um=st.session_state["b_um"],
    )
    result = model.solve()

    force_note = (
        f"Fc = {Fc_N:.1f} N (from feed h={st.session_state['h_mm']:.3f} mm via Kienzle fit)"
        if st.session_state["use_kienzle"] and material is TI64
        else f"Fc = {Fc_N:.1f} N (direct input)"
    )
    if st.session_state["use_kienzle"] and material is not TI64:
        force_note += "  [Kienzle fit only available for Ti-6Al-4V; using direct Fc]"

    st.title("Peclet-Normalized Cutting Thermal Model")
    st.markdown(
        f"**{material.name}**   |   Pe = {result.Pe:.3f}   |   "
        f"T_flash = {result.T_flash:.1f}°C"
        + ("" if result.converged else "   :warning: **not converged**")
    )
    st.caption(force_note)

    tab_1d, tab_2d = st.tabs(["1D view", "2D view"])

    with tab_1d:
        fig_norm, fig_actual = _draw_1d(result)
        col1, col2 = st.columns(2)
        col1.pyplot(fig_norm, clear_figure=True)
        col2.pyplot(fig_actual, clear_figure=True)

    with tab_2d:
        result2d = model.solve_2d(x_over_b=APP_X_OVER_B, z_over_b=APP_Z_OVER_B)
        if np.isfinite(result2d.T_critical_C):
            yielded = np.any(result2d.T_flank_profile_C > result2d.T_critical_C)
            if yielded:
                st.caption(
                    f"T_critical = {result2d.T_critical_C:.0f}°C   |   "
                    f"residual stress at flank: {result2d.residual_stress_MPa.min():.0f} to "
                    f"{result2d.residual_stress_MPa.max():.0f} MPa"
                )
            else:
                st.caption(
                    f"T_critical = {result2d.T_critical_C:.0f}°C   |   "
                    "flash temperature stays below it -- no residual stress"
                )
        else:
            st.caption(f"{material.name} never reaches its thermal-yield critical temperature")

        fig_field, fig_rs = _draw_2d(result2d)
        col1, col2 = st.columns(2)
        col1.pyplot(fig_field, clear_figure=True)
        col2.pyplot(fig_rs, clear_figure=True)


if __name__ == "__main__":
    main()
