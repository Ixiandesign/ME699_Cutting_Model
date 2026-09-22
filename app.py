"""Interactive matplotlib desktop UI for the 1D Peclet-normalized cutting
thermal model.

Shows two single-axis panels side by side, sharing the x/b axis, rather
than one plot with a dual y-axis: the normalized shape (peak always 1.0)
and that same curve scaled by the peak ("flash") temperature into actual
degrees C. A dual y-axis was deliberately avoided (see the dataviz
skill's "one axis" rule) since it invites misreading a coincidental
crossing/slope match between two differently-scaled series as meaningful.

Run with: python app.py
"""

import matplotlib.pyplot as plt
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

MATERIAL_ORDER = [TI64, AA7050, SS304, AA6061]
MATERIAL_NAMES = [m.name for m in MATERIAL_ORDER]
MATERIAL_BY_NAME = {m.name: m for m in MATERIAL_ORDER}


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
        self._updating = False  # guard against recursive slider callbacks

        self.fig = plt.figure(figsize=(12, 7.5), facecolor=SURFACE)
        self.fig.canvas.manager.set_window_title(
            "1D Peclet-Normalized Cutting Temperature Model"
        )

        self.ax_norm = self.fig.add_axes((0.08, 0.46, 0.40, 0.40))
        self.ax_actual = self.fig.add_axes(
            (0.56, 0.46, 0.40, 0.40), sharex=self.ax_norm
        )
        self._style_axes(self.ax_norm, "Normalized shape (peak = 1)", "T / T_peak")
        self._style_axes(
            self.ax_actual, "Scaled to actual temperature", "Temperature (°C)"
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

        self.info_text = self.fig.text(
            0.08, 0.955, "", fontsize=12, color=INK_PRIMARY, fontweight="bold"
        )
        self.note_text = self.fig.text(0.08, 0.925, "", fontsize=9, color=INK_MUTED)

        self._build_controls()
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

    # -- controls ----------------------------------------------------------

    def _build_controls(self):
        radio_ax = self.fig.add_axes((0.03, 0.04, 0.16, 0.30))
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

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    App().show()
