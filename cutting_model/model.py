"""High-level orchestration: bundles a material and process parameters,
and solves the flash temperature plus the normalized/actual temperature
profile in one call, for use by the UI layer."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .flash_temperature import solve_flash_temperature
from .materials import Material
from .residual_stress import critical_temperature, residual_stress
from .shape_function import normalized_shape
from .subsurface import subsurface_temperature

# x/b in [-3, 5]: wide/fine enough to capture the peak across the Pe
# range of interest (it shifts toward x/b = 1 at high Pe, per Liu et al.
# Fig. 5-7 and the Peclet paper's Fig. 12) and to show the far-field
# decay on both sides.
DEFAULT_X_OVER_B = np.linspace(-3.0, 5.0, 2000)

# z/b in [0, 4]: covers the reference workbook's finely-sampled range
# (it goes to z/b=5, coarsening past 0.5); a coarser 2D grid than
# DEFAULT_X_OVER_B is used here since it's evaluated as an outer
# product with x for every slider change in the UI.
DEFAULT_Z_OVER_B = np.linspace(0.0, 4.0, 200)


@dataclass(frozen=True)
class ProfileResult:
    x_over_b: np.ndarray
    shape: np.ndarray  # normalized, peak = 1.0
    T_actual_C: np.ndarray
    Pe: float
    T_flash: float
    peak_x_over_b: float
    converged: bool


@dataclass(frozen=True)
class TwoDResult:
    x_over_b: np.ndarray
    z_over_b: np.ndarray
    T_field_C: np.ndarray  # shape (len(z_over_b), len(x_over_b))
    T_flank_profile_C: np.ndarray  # shape (len(z_over_b),), at x/b = flank_x_over_b
    residual_stress_MPa: np.ndarray  # shape (len(z_over_b),)
    T_critical_C: float
    flank_x_over_b: float
    Pe: float
    T_flash: float
    converged: bool


@dataclass
class CuttingThermalModel:
    material: Material
    v_m_min: float  # cutting speed, m/min (UI-friendly unit)
    Fc_N: float  # cutting force, N
    w_mm: float  # width of cut, mm
    b_um: float  # tool-chip/flank contact half-width, micrometers
    T_ambient_C: float = 20.0

    def solve(self, x_over_b: Optional[np.ndarray] = None) -> ProfileResult:
        if x_over_b is None:
            x_over_b = DEFAULT_X_OVER_B
        x_over_b = np.asarray(x_over_b, dtype=float)

        v_m_s = self.v_m_min / 60.0
        w_m = self.w_mm / 1000.0
        b_m = self.b_um * 1e-6

        flash = solve_flash_temperature(
            self.material, v_m_s, self.Fc_N, w_m, b_m, T_ambient=self.T_ambient_C
        )
        shape = normalized_shape(x_over_b, flash.Pe)
        T_actual = self.T_ambient_C + flash.T_flash * shape
        peak_x = float(x_over_b[np.argmax(shape)])

        return ProfileResult(
            x_over_b=x_over_b,
            shape=shape,
            T_actual_C=T_actual,
            Pe=flash.Pe,
            T_flash=flash.T_flash,
            peak_x_over_b=peak_x,
            converged=flash.converged,
        )

    def solve_2d(
        self,
        x_over_b: Optional[np.ndarray] = None,
        z_over_b: Optional[np.ndarray] = None,
        flank_x_over_b: float = 1.0,
    ) -> TwoDResult:
        """Subsurface temperature field plus residual-stress-vs-depth at
        the flank/tool-exit location (`x/b = flank_x_over_b`, default 1.0,
        matching the reference workbook's row 44 / MATLAB's `flank`
        index)."""
        if x_over_b is None:
            x_over_b = DEFAULT_X_OVER_B
        if z_over_b is None:
            z_over_b = DEFAULT_Z_OVER_B
        x_over_b = np.asarray(x_over_b, dtype=float)
        z_over_b = np.asarray(z_over_b, dtype=float)

        v_m_s = self.v_m_min / 60.0
        w_m = self.w_mm / 1000.0
        b_m = self.b_um * 1e-6

        flash = solve_flash_temperature(
            self.material, v_m_s, self.Fc_N, w_m, b_m, T_ambient=self.T_ambient_C
        )

        T_field = subsurface_temperature(
            x_over_b,
            z_over_b,
            flash.Pe,
            flash.T_flash,
            self.T_ambient_C,
            flash.k,
            flash.cp,
            self.material.rho,
            v_m_s,
            b_m,
        )

        flank_idx = int(np.argmin(np.abs(x_over_b - flank_x_over_b)))
        T_flank_profile = T_field[:, flank_idx]

        T_crit = critical_temperature(self.material, T_ambient=self.T_ambient_C)
        RS = residual_stress(self.material, T_flank_profile, T_crit)

        return TwoDResult(
            x_over_b=x_over_b,
            z_over_b=z_over_b,
            T_field_C=T_field,
            T_flank_profile_C=T_flank_profile,
            residual_stress_MPa=RS,
            T_critical_C=T_crit,
            flank_x_over_b=float(x_over_b[flank_idx]),
            Pe=flash.Pe,
            T_flash=flash.T_flash,
            converged=flash.converged,
        )
