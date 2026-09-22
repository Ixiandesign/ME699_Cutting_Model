"""Interactive matplotlib desktop UI for the Peclet-normalized cutting
thermal model, with a toggle between two views:

- **1D view**: two single-axis panels side by side, sharing the x/b axis
  -- the normalized surface shape (peak always 1.0) and that same curve
  scaled by the peak ("flash") temperature into actual degrees C. A dual
  y-axis was deliberately avoided (see the dataviz skill's "one axis"
  rule) since it invites misreading a coincidental crossing/slope match
  between two differently-scaled series as meaningful.
- **2D view**: the subsurface temperature field T(x/b, z/b) as a
  cutaway heatmap (z=0, the surface, at top), and the residual-stress-
  vs-depth profile at the flank/tool-exit location (x/b=1), both driven
  by the same sliders and material selector as the 1D view.

Run with: python app.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

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
# field, in the same orange family as SERIES_ACTUAL (the 1D "actual
# temperature" series), running from the light surface color through
# to a dark burnt-orange at the hot end.
TEMP_CMAP = LinearSegmentedColormap.from_list(
    "cutting_temp", [SURFACE, SERIES_ACTUAL, "#7a2e0e"]
)

MATERIAL_ORDER = [TI64, AA7050, SS304, AA6061]
MATERIAL_NAMES = [m.name for m in MATERIAL_ORDER]
MATERIAL_BY_NAME = {m.name: m for m in MATERIAL_ORDER}

# Coarser than cutting_model.model's defaults: this grid is re-evaluated
# on every slider tick and rendered as a 2D mesh, so resolution is
# traded for interactivity.
APP_X_OVER_B = np.linspace(-3.0, 5.0, 300)
APP_Z_OVER_B = np.linspace(0.0, 4.0, 150)


class App:
    def __init__(self):
        self.material = TI64
        defaults = DEFAULTS[self.material.name]
        self.v_m_min = defaults["v_m_min"]
        self.Fc_N = defaults["Fc_N"]
        self.w_mm = defaults["w_mm"]
        self.b_um = defaults["b_um"]
        self.h_mm = 0.10
        self.use_kienzle = False
        self.view = "1D"
        self._updating = False  # guard against recursive slider callbacks

        self.fig = plt.figure(figsize=(12, 8.5), facecolor=SURFACE)
        self.fig.canvas.manager.set_window_title(
            "Peclet-Normalized Cutting Thermal Model"
        )

        self._build_1d_axes()
        self._build_2d_axes()

        self.info_text = self.fig.text(
            0.08, 0.965, "", fontsize=12, color=INK_PRIMARY, fontweight="bold"
        )
        self.note_text = self.fig.text(0.08, 0.940, "", fontsize=9, color=INK_MUTED)
        self.rs_text = self.fig.text(0.08, 0.918, "", fontsize=9, color=INK_MUTED)

        self._build_controls()
        self._set_view("1D")
        self._recompute_and_draw()

    # -- styling ---------------------------------------------------------

    def _style_axes(self, ax, title, ylabel):
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

    # -- panel construction -------------------------------------------------

    def _build_1d_axes(self):
        self.ax_norm = self.fig.add_axes((0.08, 0.46, 0.40, 0.42))
        self.ax_actual = self.fig.add_axes(
            (0.56, 0.46, 0.40, 0.42), sharex=self.ax_norm
        )
        self._style_axes(self.ax_norm, "Normalized surface shape (peak = 1)", "T / T_peak")
        self._style_axes(
            self.ax_actual, "Actual surface temperature", "Temperature (°C)"
        )
        self.ax_norm.set_xlabel("x / b", color=INK_SECONDARY)
        self.ax_actual.set_xlabel("x / b", color=INK_SECONDARY)

        (self.line_norm,) = self.ax_norm.plot([], [], color=SERIES_NORM, linewidth=2)
        (self.line_actual,) = self.ax_actual.plot(
            [], [], color=SERIES_ACTUAL, linewidth=2
        )
        (self.peak_dot_norm,) = self.ax_norm.plot(
            [],
            [],
            marker="o",
            linestyle="none",
            color=SERIES_NORM,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=5,
        )
        (self.peak_dot_actual,) = self.ax_actual.plot(
            [],
            [],
            marker="o",
            linestyle="none",
            color=SERIES_ACTUAL,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=5,
        )
        self.label_norm = self.ax_norm.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 6),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=10,
        )
        self.label_actual = self.ax_actual.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 6),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=10,
        )
        self.axes_1d = (self.ax_norm, self.ax_actual)

    def _build_2d_axes(self):
        self.ax_field = self.fig.add_axes((0.08, 0.46, 0.40, 0.42))
        self.ax_rs = self.fig.add_axes((0.56, 0.46, 0.40, 0.42))
        self._style_axes(
            self.ax_field, "Subsurface temperature field (flank cutaway)", "z / b (depth)"
        )
        self._style_axes(
            self.ax_rs, "Residual stress near flank (x/b ≈ 1)", "Residual stress (MPa)"
        )
        self.ax_field.set_xlabel("x / b", color=INK_SECONDARY)
        self.ax_rs.set_xlabel("z / b (depth)", color=INK_SECONDARY)
        self.ax_field.invert_yaxis()  # surface (z=0) at top, like a cutaway

        self.quadmesh = None  # created on first draw (needs data shape)
        self.field_colorbar = None
        self.critical_contour = None
        self.flank_marker = None

        (self.line_rs,) = self.ax_rs.plot([], [], color=SERIES_ACTUAL, linewidth=2)
        self.zero_line_rs = self.ax_rs.axhline(0.0, color=AXIS_COLOR, linewidth=1.0)
        self.critical_depth_line = self.ax_rs.axvline(
            np.nan, color=INK_MUTED, linewidth=1.0, linestyle="--"
        )
        self.critical_depth_label = self.ax_rs.annotate(
            "",
            xy=(0, 0),
            xytext=(6, 6),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8,
        )
        self.axes_2d = (self.ax_field, self.ax_rs)

    # -- controls ----------------------------------------------------------

    def _build_controls(self):
        view_ax = self.fig.add_axes((0.03, 0.36, 0.16, 0.09))
        view_ax.set_facecolor(SURFACE)
        view_ax.set_title("View", color=INK_PRIMARY, fontsize=10, loc="left")
        self.view_radio = RadioButtons(
            view_ax, ["1D view", "2D view"], active=0, activecolor=SERIES_NORM
        )
        for text in self.view_radio.labels:
            text.set_color(INK_SECONDARY)
            text.set_fontsize(9)
        self.view_radio.on_clicked(self._on_view_change)

        radio_ax = self.fig.add_axes((0.03, 0.04, 0.16, 0.28))
        radio_ax.set_facecolor(SURFACE)
        radio_ax.set_title("Material", color=INK_PRIMARY, fontsize=10, loc="left")
        self.radio = RadioButtons(
            radio_ax, MATERIAL_NAMES, active=0, activecolor=SERIES_NORM
        )
        for text in self.radio.labels:
            text.set_color(INK_SECONDARY)
            text.set_fontsize(9)
        self.radio.on_clicked(self._on_material_change)

        slider_specs = [
            ("v", "Cutting speed vc (m/min)", 1.0, 300.0, self.v_m_min),
            ("Fc", "Cutting force Fc (N)", 10.0, 1000.0, self.Fc_N),
            ("b", "Contact half-width b (µm)", 10.0, 500.0, self.b_um),
            ("w", "Width of cut w (mm)", 1.0, 10.0, self.w_mm),
            ("h", "Feed h (mm) [Kienzle, Ti-6Al-4V only]", 0.01, 0.30, self.h_mm),
        ]
        self.sliders = {}
        left, width = 0.30, 0.62
        top, step, height = 0.30, 0.062, 0.03
        for i, (key, label, vmin, vmax, vinit) in enumerate(slider_specs):
            ax = self.fig.add_axes((left, top - i * step, width, height))
            ax.set_facecolor(SURFACE)
            slider = Slider(
                ax,
                label,
                vmin,
                vmax,
                valinit=vinit,
                color=SERIES_NORM,
                initcolor="none",
            )
            slider.label.set_color(INK_SECONDARY)
            slider.label.set_fontsize(9)
            slider.valtext.set_color(INK_MUTED)
            slider.on_changed(self._on_slider_change(key))
            self.sliders[key] = slider

        check_ax = self.fig.add_axes((0.30, 0.003, 0.35, 0.03))
        check_ax.set_facecolor(SURFACE)
        check_ax.axis("off")
        self.check = CheckButtons(
            check_ax,
            ["Estimate Fc from feed h (Kienzle fit)"],
            [self.use_kienzle],
        )
        for text in self.check.labels:
            text.set_color(INK_SECONDARY)
            text.set_fontsize(9)
        self.check.on_clicked(self._on_kienzle_toggle)

    def _on_slider_change(self, key):
        def handler(val):
            if self._updating:
                return
            if key == "v":
                self.v_m_min = val
            elif key == "Fc":
                self.Fc_N = val
            elif key == "b":
                self.b_um = val
            elif key == "w":
                self.w_mm = val
            elif key == "h":
                self.h_mm = val
            self._recompute_and_draw()

        return handler

    def _on_kienzle_toggle(self, label):
        self.use_kienzle = not self.use_kienzle
        self._recompute_and_draw()

    def _on_material_change(self, label):
        self.material = MATERIAL_BY_NAME[label]
        defaults = DEFAULTS[self.material.name]
        self._updating = True
        self.sliders["v"].set_val(defaults["v_m_min"])
        self.sliders["Fc"].set_val(defaults["Fc_N"])
        self.sliders["b"].set_val(defaults["b_um"])
        self.sliders["w"].set_val(defaults["w_mm"])
        self._updating = False
        self.v_m_min = defaults["v_m_min"]
        self.Fc_N = defaults["Fc_N"]
        self.b_um = defaults["b_um"]
        self.w_mm = defaults["w_mm"]
        self._recompute_and_draw()

    def _on_view_change(self, label):
        self._set_view("1D" if label == "1D view" else "2D")
        # Switching into 2D view needs a recompute -- its panels aren't
        # kept up to date while hidden (see _recompute_and_draw).
        self._recompute_and_draw()

    def _set_view(self, view):
        self.view = view
        for ax in self.axes_1d:
            ax.set_visible(view == "1D")
        for ax in self.axes_2d:
            ax.set_visible(view == "2D")
        if self.field_colorbar is not None:
            self.field_colorbar.ax.set_visible(view == "2D")

    # -- compute + redraw ---------------------------------------------------

    def _effective_force(self):
        if self.use_kienzle and self.material is TI64:
            return estimate_cutting_force(self.material, self.h_mm, self.w_mm)
        return self.Fc_N

    def _recompute_and_draw(self):
        Fc_N = self._effective_force()
        model = CuttingThermalModel(
            material=self.material,
            v_m_min=self.v_m_min,
            Fc_N=Fc_N,
            w_mm=self.w_mm,
            b_um=self.b_um,
        )
        result = model.solve()
        self._draw_1d(model, result)

        result2d = None
        if self.view == "2D":
            # Only pay for the 45000-point field evaluation and the
            # colormesh/contour rebuild while the 2D panels are actually
            # visible; _on_view_change() triggers one more recompute at
            # the moment the user switches into this view.
            result2d = model.solve_2d(x_over_b=APP_X_OVER_B, z_over_b=APP_Z_OVER_B)
            self._draw_2d(result2d)

        force_note = (
            f"Fc = {Fc_N:.1f} N (from feed h={self.h_mm:.3f} mm via Kienzle fit)"
            if self.use_kienzle and self.material is TI64
            else f"Fc = {Fc_N:.1f} N (direct input)"
        )
        if self.use_kienzle and self.material is not TI64:
            force_note += "  [Kienzle fit only available for Ti-6Al-4V; using direct Fc]"

        self.info_text.set_text(
            f"{self.material.name}   |   Pe = {result.Pe:.3f}   |   "
            f"T_flash = {result.T_flash:.1f}°C"
            + ("" if result.converged else "   [WARNING: not converged]")
        )
        self.note_text.set_text(force_note)

        self.rs_text.set_visible(self.view == "2D")
        if result2d is not None:
            if np.isfinite(result2d.T_critical_C):
                # Whether the flank ever exceeds T_critical, not whether
                # the resulting RS(T) is positive -- for materials whose
                # rs_fit is reused from a different material (see
                # residual_stress.py's docstring), RS(T) can come out
                # negative just above T_critical, so `RS.max() > 0` is
                # not a reliable proxy for "did it yield."
                yielded = np.any(result2d.T_flank_profile_C > result2d.T_critical_C)
                if yielded:
                    self.rs_text.set_text(
                        f"T_critical = {result2d.T_critical_C:.0f}°C   |   "
                        f"residual stress at flank: {result2d.residual_stress_MPa.min():.0f} to "
                        f"{result2d.residual_stress_MPa.max():.0f} MPa"
                    )
                else:
                    self.rs_text.set_text(
                        f"T_critical = {result2d.T_critical_C:.0f}°C   |   "
                        "flash temperature stays below it -- no residual stress"
                    )
            else:
                self.rs_text.set_text(
                    f"{self.material.name} never reaches its thermal-yield critical temperature"
                )

        self.fig.canvas.draw_idle()

    def _draw_1d(self, model, result):
        T_ambient = model.T_ambient_C
        T_actual_max = result.T_actual_C.max()

        self.line_norm.set_data(result.x_over_b, result.shape)
        self.line_actual.set_data(result.x_over_b, result.T_actual_C)
        self.peak_dot_norm.set_data([result.peak_x_over_b], [1.0])
        self.peak_dot_actual.set_data([result.peak_x_over_b], [T_actual_max])

        self.label_norm.xy = (result.peak_x_over_b, 1.0)
        self.label_norm.set_text(f"1.00 @ x/b={result.peak_x_over_b:.2f}")
        self.label_actual.xy = (result.peak_x_over_b, T_actual_max)
        self.label_actual.set_text(
            f"{T_actual_max:.1f}°C @ x/b={result.peak_x_over_b:.2f}"
        )

        self.ax_norm.set_xlim(result.x_over_b.min(), result.x_over_b.max())
        self.ax_norm.set_ylim(0, 1.15)
        y_lo = min(T_ambient, result.T_actual_C.min()) * 0.95
        self.ax_actual.set_ylim(y_lo, T_actual_max * 1.15)

    def _draw_2d(self, result2d):
        if self.quadmesh is not None:
            self.quadmesh.remove()
        self.quadmesh = self.ax_field.pcolormesh(
            result2d.x_over_b,
            result2d.z_over_b,
            result2d.T_field_C,
            cmap=TEMP_CMAP,
            shading="auto",
        )
        if self.field_colorbar is None:
            self.field_colorbar = self.fig.colorbar(
                self.quadmesh, ax=self.ax_field, pad=0.02
            )
            self.field_colorbar.ax.tick_params(colors=INK_MUTED)
            self.field_colorbar.set_label("Temperature (°C)", color=INK_SECONDARY)
            self.field_colorbar.outline.set_visible(False)
        else:
            self.field_colorbar.update_normal(self.quadmesh)
        self.field_colorbar.ax.set_visible(self.view == "2D")

        if self.critical_contour is not None:
            self.critical_contour.remove()
            self.critical_contour = None
        T_crit = result2d.T_critical_C
        field_max = result2d.T_field_C.max()
        if np.isfinite(T_crit) and field_max > T_crit > result2d.T_field_C.min():
            self.critical_contour = self.ax_field.contour(
                result2d.x_over_b,
                result2d.z_over_b,
                result2d.T_field_C,
                levels=[T_crit],
                colors=[INK_PRIMARY],
                linewidths=1.0,
                linestyles="dashed",
            )
        if self.flank_marker is not None:
            self.flank_marker.remove()
        self.flank_marker = self.ax_field.axvline(
            result2d.flank_x_over_b, color=INK_PRIMARY, linewidth=1.0, linestyle=":"
        )

        self.line_rs.set_data(result2d.z_over_b, result2d.residual_stress_MPa)
        self.ax_rs.set_xlim(result2d.z_over_b.min(), result2d.z_over_b.max())
        rs = result2d.residual_stress_MPa
        rs_max = max(rs.max(), 0.0)
        rs_min = min(rs.min(), 0.0)
        pad = max(rs_max - rs_min, 1.0) * 0.15
        self.ax_rs.set_ylim(rs_min - pad, rs_max + pad)

        if np.isfinite(result2d.T_critical_C) and np.any(
            result2d.T_flank_profile_C > result2d.T_critical_C
        ):
            affected = result2d.z_over_b[
                result2d.T_flank_profile_C > result2d.T_critical_C
            ]
            depth = affected.max()
            self.critical_depth_line.set_xdata([depth, depth])
            self.critical_depth_label.xy = (depth, rs_max + pad * 0.5)
            self.critical_depth_label.set_text(f"yielded to z/b={depth:.2f}")
        else:
            self.critical_depth_line.set_xdata([np.nan, np.nan])
            self.critical_depth_label.set_text("")

    def show(self):
        plt.show()


if __name__ == "__main__":
    App().show()
